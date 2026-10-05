"""
Bug generation pipeline for PDB (Precise Debugging Benchmarking).

Injects bugs into correct solutions using an LLM, composes multi-bug variants,
and validates/samples the result set for downstream evaluation.

Pipeline stages (see `gen_main`):
    1. `bug_generate`   -- LLM proposes single-block bugs; each is verified
                           against unit tests and checked for atomicity.
    2. `bug_compose`    -- combines k non-adjacent verified blocks into
                           multi-bug variants for k = 2..max_bugs.
    3. `validate_and_sample` -- final round-trip check (apply gt_diff to
                           buggy_code recovers gt_solution) + bucketed sampling.

Two modes flow through the whole pipeline:
    single -- one edited line per block, stride=2 between blocks.
    multi  -- MIN_MULTILINES..MAX_MULTILINES contiguous lines per block, stride=4.

Diff direction convention used throughout:
    `diff`    -- forward edit (gt -> buggy). Used to compose bugs onto gt.
    `gt_diff` -- reverse edit (buggy -> gt). The target a correct fix must produce.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import datetime
import numpy as np
import random
import tqdm
import json
import copy
import dspy
import argparse
import logging
import itertools
from dataset import get_handler
from concurrent.futures import ThreadPoolExecutor
from utils import file_diff, apply_diff, verify_block_single_diff, verify_block_diff, verify_span_diff, parse_diff_to_blocks, single_diff_to_str, str_to_single_diff, rstrip_lines, set_block_merge_mode
from module import BugInjector, MultilineBugInjector
from examples import odc_categories, odc_category_probs, bug_type_examples, multiline_bug_type_examples
from api_config import resolve_api_key
from config import MIN_MULTILINES, MAX_MULTILINES
from test_adequacy import run_test_adequacy, apply_coverage_filter
from collections import defaultdict, Counter


def _build_contiguous_ranges(lines, min_size=2, max_size=4):
    """
    Build all contiguous ranges of min_size to max_size from a list of
    (line_no, line_content) tuples. Lines are contiguous if their line
    numbers are consecutive.
    """
    # NOTE: [design thought] multi mode asks the LLM to edit a *contiguous*
    # block. We hand it a menu of valid ranges rather than trusting it to
    # pick one itself -- LLMs routinely pick non-contiguous lines otherwise.
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda x: x[0])
    ranges = []
    # Find all maximal contiguous runs
    runs = []
    current_run = [sorted_lines[0]]
    for i in range(1, len(sorted_lines)):
        if sorted_lines[i][0] == current_run[-1][0] + 1:
            current_run.append(sorted_lines[i])
        else:
            runs.append(current_run)
            current_run = [sorted_lines[i]]
    runs.append(current_run)

    # Extract all sub-ranges of valid size from each run
    for run in runs:
        for size in range(min_size, min(max_size, len(run)) + 1):
            for start in range(len(run) - size + 1):
                ranges.append(run[start:start + size])
    return ranges


def bug_generate(data, handler, log_file_prefix, bug_per_example, ic_size=4, mode="single",
                 max_lines_per_block=MAX_MULTILINES, multi_validation="block", n_workers_llm=1,
                 n_workers_validation=4, atomicity_full_max_lines=MAX_MULTILINES):
    """
    Generate buggy code from correct solutions.

    Runs `bug_per_example` LLM passes over `data`; each pass samples one ODC
    bug category per example, prompts the injector, validates the resulting
    single-block edit, and collects it. After the LLM phase, every candidate
    is unit-test-verified and checked for atomicity in one batched sandbox call.

    :param mode: "single" for single-line bugs (original), "multi" for multiline block bugs
    :param max_lines_per_block: max lines per bug block in multiline mode
        (default `config.MAX_MULTILINES`)
    :param multi_validation: "block" -- a multiline bug must be exactly one contiguous
        block (original PDB rule); "span" -- all edits within a max_lines_per_block
        window, which also admits code-reorder bugs (used for SWE-smith)
    :param n_workers_llm: parallel LLM calls per pass
    :param n_workers_validation: parallelism hint for handler.verify_unit_test
    :param atomicity_full_max_lines: bugs with at most this many edited lines are
        checked against every proper partial repair; larger bugs only against
        single-line reverts (2^k - 2 partial programs is intractable for large k)
    """
    results = []
    if mode == "multi":
        bug_gen = MultilineBugInjector(max_lines_per_block=max_lines_per_block)
        action_examples = multiline_bug_type_examples
    else:
        bug_gen = BugInjector()
        action_examples = bug_type_examples

    def _generate_one(item, count):
        """One LLM attempt for one example; returns the log entry or None."""
        task_id = item.get("task_id") + f"_{count}"
        gt_solution = item.get("gt_solution")
        task_prompt = item.get("task_prompt")
        log_entry = copy.deepcopy(item)
        log_entry["task_id"] = task_id

        # NOTE: [design thought] category sampled per-example per-pass so
        # the final dataset's category mix matches odc_category_probs in
        # expectation even when some generations fail verification.
        bug_type = np.random.choice(list(odc_category_probs.keys()), p=list(odc_category_probs.values()))
        bug_def = odc_categories[bug_type]["Definition"]
        if len(odc_categories[bug_type]["Examples"]) <= ic_size:
            bug_examples = list(odc_categories[bug_type]["Examples"].items())
        else:
            bug_examples_idx = np.random.choice(range(len(odc_categories[bug_type]["Examples"].keys())), ic_size,
                                                replace=False)
            all_examples = list(odc_categories[bug_type]["Examples"].items())
            bug_examples = [all_examples[i] for i in bug_examples_idx]

        # NOTE: [design thought] sample 2 extra subtypes from OTHER ODC
        # categories to inspire cross-category blended multiline bugs. Only used
        # in multi mode.
        cross_category_examples = []
        if mode == "multi":
            other_cats = [c for c in odc_categories.keys() if c != bug_type]
            picked_cats = random.sample(other_cats, min(2, len(other_cats)))
            for oc in picked_cats:
                subs = list(odc_categories[oc]["Examples"].items())
                sub, expl = subs[random.randint(0, len(subs) - 1)]
                # Use just the explanation, drop the multi-line 'Bug Example:' code block.
                expl_short = expl.split("\nBug Example:")[0].strip()
                cross_category_examples.append((f"{oc} / {sub}", expl_short))
        bug_type_sum = [bug_type, bug_def, bug_examples, cross_category_examples]

        if bug_type == "Algorithm":
            action_type = action_examples[random.randint(0, len(action_examples) - 1)]
        else:
            action_type = action_examples[-1]

        # NOTE: [edge case callout] deletion is restricted to lines whose
        # removal still leaves syntactically valid code (see handler.mark_editable_lines).
        if action_type.startswith("Delete"):
            candidate_lines = log_entry["deletable_lines"]
        else:
            candidate_lines = log_entry["editable_lines"]
        if not candidate_lines:
            return None

        if mode == "multi":
            # Build contiguous ranges of MIN_MULTILINES..max_lines_per_block lines
            # from the candidate lines, then present these ranges to the LLM.
            # This ensures the LLM picks from valid contiguous regions.
            contiguous_ranges = _build_contiguous_ranges(
                candidate_lines, min_size=MIN_MULTILINES, max_size=max_lines_per_block)
            if not contiguous_ranges:
                return None
            # Select up to 5 random ranges to present
            num_ranges = min(5, len(contiguous_ranges))
            selected_ranges = random.sample(contiguous_ranges, num_ranges)
            action_on_lines = [action_type, selected_ranges]
        else:
            selected_lines_idx = np.random.choice(range(len(candidate_lines)),
                                                  len(candidate_lines) // 2, replace=False)
            selected_lines = [candidate_lines[i] for i in selected_lines_idx]
            action_on_lines = [action_type, selected_lines]

        try:
            response = bug_gen(task_prompt=task_prompt, gt_solution=gt_solution, bug_type=bug_type_sum,
                               action_on_lines=action_on_lines)
            log_entry["bug_type"] = bug_type
            log_entry["bug_subtype"] = response.subtype
            log_entry["buggy_code"] = response.buggy_code
            # NOTE: [pedagogical] file_diff(gt, buggy) returns the FORWARD diff
            # gt -> buggy. verify_block_diff asserts it forms exactly one block
            # of the right shape; anything weirder (multi-block, wrong size) is dropped.
            _, _, json_diff = file_diff(gt_solution, log_entry["buggy_code"].strip())
            if json_diff is None:
                print("JSON diff wrong format from the response.")
                return None
            if mode == "multi" and multi_validation == "span":
                # Span mode: all edits within a max_lines_per_block window; the
                # edits may form several blocks (e.g. a code move).
                diff_ok = verify_span_diff(json_diff, max_span=max_lines_per_block,
                                           min_changes=MIN_MULTILINES, max_changes=max_lines_per_block)[0]
            else:
                # Single mode: exactly 1 block with 1 line.
                # Multi mode: exactly 1 block with MIN_MULTILINES..max_lines_per_block lines.
                lines_limit = max_lines_per_block if mode == "multi" else 1
                min_limit = MIN_MULTILINES if mode == "multi" else 1
                diff_ok = verify_block_diff(json_diff, block_count=1, max_lines_per_block=lines_limit,
                                            min_lines_per_block=min_limit)[0]
            if not diff_ok:
                print("Block validation failed: ", json.dumps(json_diff, indent=2))
                return None
            edited_line_no = min(int(str(k).strip()) for k in json_diff.keys())
            # NOTE: [edge case callout] frozen_lines are the prompt/signature
            # prefix -- editing them would change the task itself, not the solution.
            if edited_line_no <= log_entry["frozen_lines"]:
                print("Edit frozen lines:", edited_line_no)
                return None
            if mode == "multi":
                allowed_line_no = [l[0] for r in selected_ranges for l in r]
            else:
                allowed_line_no = [l[0] for l in selected_lines]
            if edited_line_no not in allowed_line_no:
                print(f"Allowed lines {allowed_line_no}, but the edit is: ",
                      json.dumps(json_diff, indent=2))
            log_entry["diff"] = json_diff
            # NOTE: [design thought] bug_count counts blocks; a span-mode bug whose
            # edits form several blocks therefore has bug_count > 1.
            log_entry["bug_count"] = len(parse_diff_to_blocks(json_diff))
            return log_entry
        except Exception as e:
            print(f"Error processing task_id {task_id}: {e}")
            return None

    # NOTE: [performance improvement] LLM calls are I/O bound; dspy + litellm are
    # thread-safe via the module-level settings, so each pass fans out over a pool.
    for count in range(bug_per_example):
        print(f"Generating buggy code step {count} (mode={mode})")
        with ThreadPoolExecutor(max_workers=max(1, min(n_workers_llm, len(data)))) as pool:
            futures = [pool.submit(_generate_one, item, count) for item in data]
            for fut in tqdm.tqdm(futures, total=len(futures), desc=f"step {count}"):
                entry = fut.result()
                if entry is not None:
                    results.append(entry)

    # NOTE: [design thought] Checkpoint the raw generated bugs before the
    # atomicity/verify phase. The verify step shells out to the dataset
    # sandbox (bigcodebench.evaluate / lcb_runner) and can take hours on
    # large runs; if the parent process dies there we'd otherwise lose every
    # bug generated during the preceding step loop.
    preverify_path = log_file_prefix + "_bug_preverify.json"
    with open(preverify_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pre-verify checkpoint: {len(results)} bugs saved to {preverify_path}")

    # NOTE: [design thought] Atomicity check batched into main verify.
    # For each k-line multiline bug (k>=2), build 2^k-2 partial-fix programs by
    # reverting a proper non-empty subset of edits back to GT. Batch them into
    # the SAME verify call. If any partial-fix PASSES tests, the bug is
    # non-atomic and we reject it.
    # NOTE: [pedagogical] this enforces the "compound-independent" property:
    # a multi-line bug is atomic iff fixing only some of its lines does NOT
    # already satisfy the tests -- otherwise the lines aren't really one bug.
    # NOTE: [performance improvement] 2^k-2 grows fast; in practice k<=4 so
    # at most 14 extra entries per bug. If MAX_MULTILINES were raised, a
    # random subset of proper subsets would keep this tractable.
    import itertools as _itertools

    atomicity_entries = []
    atom_to_orig = {}

    for entry in results:
        diff = entry.get("diff")
        if not diff or len(diff) < 2:
            continue  # single-line bugs have no proper subsets

        gt_solution = entry["gt_solution"]
        buggy_code = entry["buggy_code"]
        edits = sorted(diff.items(), key=lambda kv: int(kv[0].strip()))
        k = len(edits)

        # NOTE: [pedagogical] range(1, k) gives proper non-empty subsets only:
        # size 0 = buggy code itself (already verified), size k = full revert = GT.
        # Bugs larger than atomicity_full_max_lines are only probed with
        # single-line reverts to keep the number of partial programs linear in k.
        max_revert = k if k <= atomicity_full_max_lines else 2
        for revert_size in range(1, max_revert):
            for revert_combo in _itertools.combinations(range(k), revert_size):
                # Invert each reverted edit (Add <-> Delete; swap original/modified).
                revert_diff = {}
                for idx in revert_combo:
                    line_no, edit = edits[idx]
                    inv_type = edit["type"]
                    if inv_type == "Add":
                        inv_type = "Delete"
                    elif inv_type == "Delete":
                        inv_type = "Add"
                    revert_diff[line_no] = {
                        "type": inv_type,
                        "original": edit["modified"],
                        "modified": edit["original"],
                    }
                partial_code = apply_diff(buggy_code, revert_diff)
                if partial_code.strip() == gt_solution.strip():
                    continue  # full revert == GT, not a partial fix
                atom_tid = f"{entry['task_id']}__atom__{revert_size}_{'_'.join(map(str, revert_combo))}"
                atomicity_entries.append({
                    "task_id": atom_tid,
                    "solution": partial_code,
                    "buggy_code": partial_code,
                })
                atom_to_orig[atom_tid] = entry["task_id"]

    combined_entries = [dict(e) for e in results] + atomicity_entries

    # NOTE: [performance improvement] one batched verify_unit_test call amortizes
    # the sandbox spin-up cost across bugs + all their partial-fix siblings.
    # Verify buggy (and partial fixes) in ONE batch call.
    formatted_gt = handler.save_formatted_gt(log_file_prefix + "_bug_gen_gt", combined_entries)
    verify_file = handler.build_verify_unit_test(log_file_prefix + "_bug_verify", combined_entries, sol_field="buggy_code")
    try:
        fail_ids, correct_ids, verify_info = handler.verify_unit_test(
            verify_file, gt_file=formatted_gt, timeout=1800, n_workers=n_workers_validation)
    except Exception as e:
        print(f"Error verifying. Save first.")
        with open(log_file_prefix + "_bug.json", "w") as f:
            json.dump(results, f, indent=2)
        return results, []

    # Atomicity entry in correct_ids => partial-subset revert still passed tests
    # => the bug is non-atomic.
    non_atomic_orig_ids = set()
    atomicity_reject_counts = defaultdict(int)
    for atom_tid, orig_tid in atom_to_orig.items():
        if atom_tid in correct_ids:
            non_atomic_orig_ids.add(orig_tid)
            atomicity_reject_counts[orig_tid] += 1

    new_data = []
    for entry in results:
        entry["editable_lines"] = len(entry["editable_lines"])
        entry["deletable_lines"] = len(entry["deletable_lines"])
        if entry["task_id"] in non_atomic_orig_ids:
            entry["is_buggy"] = False
            entry["atomicity_violation"] = True
            entry["atomicity_reject_count"] = atomicity_reject_counts[entry["task_id"]]
            print(f"Non-atomic bug rejected: {entry['task_id']} "
                  f"({atomicity_reject_counts[entry['task_id']]} partial fixes passed)")
        elif entry["task_id"] in fail_ids:
            # NOTE: [design thought] strip the trailing "_<pass_index>" suffix so
            # downstream composition can merge multiple distinct bugs per original task.
            entry["is_buggy"] = True
            # Handlers that report which tests a bug breaks (SWE-smith) return a
            # {task_id: {"FAIL_TO_PASS": [...], "PASS_TO_PASS": [...]}} mapping.
            f2p = verify_info.get(entry["task_id"]) if isinstance(verify_info, dict) else None
            if isinstance(f2p, dict):
                entry["FAIL_TO_PASS"] = f2p.get("FAIL_TO_PASS", [])
                entry["PASS_TO_PASS"] = f2p.get("PASS_TO_PASS", [])
            entry["task_id"] = entry["task_id"].rsplit("_", 1)[0]
            new_data.append(entry)
        elif entry["task_id"] in correct_ids:
            entry["is_buggy"] = False

    print("Total buggy code generated: {} out of {}".format(len(new_data), len(results)))
    with open(log_file_prefix + "_bug.json", "w") as f:
        json.dump(results, f, indent=2)
    return results, new_data


def compose_and_apply_diff(gt_solution, k, diff_blocks, stride=2, max_lines_per_block=1, max_try=100):
    """
    Pick k non-adjacent diff blocks from diff_blocks and apply them.

    Each diff_block is a dict of line-level edits forming one contiguous block.
    For single-line mode, each block has 1 entry. For multiline, up to max_lines_per_block.
    """
    # NOTE: [design thought] stride=2 (single) / stride=4 (multi) keeps composed
    # bugs far enough apart that a fix for one can't accidentally overlap another;
    # this preserves block-count semantics for evaluation.
    if len(diff_blocks) < k:
        return None, None

    # Each diff_block is a dict {line_no: {type, original, modified}}
    # Compute block_start for stride checking
    block_info = []
    for block in diff_blocks:
        line_nos = [int(l) for l in block.keys()]
        block_info.append({
            "start": min(line_nos),
            "end": max(line_nos),
            "diff": block,
        })
    block_info.sort(key=lambda x: x["start"])

    for _ in range(max_try):
        chosen = random.sample(block_info, k)
        chosen.sort(key=lambda x: x["start"])

        # Enforce stride between block boundaries
        valid = True
        for i in range(1, len(chosen)):
            if chosen[i]["start"] - chosen[i - 1]["end"] <= stride:
                valid = False
                break
        if not valid:
            continue

        # Merge chosen blocks into a single diff and apply
        merged_diff = {}
        for c in chosen:
            merged_diff.update(c["diff"])
        buggy_code = apply_diff(gt_solution, merged_diff, with_delta=False)
        # NOTE: [pedagogical] the COMPOSED diff is gt->buggy, but we persist
        # gt_diff = buggy->gt because that's the direction a fix must produce.
        gt_diff = file_diff(buggy_code, gt_solution, cleaned=True)[2]
        ver, err = verify_block_diff(gt_diff, block_count=k, stride=stride,
                                     max_lines_per_block=max_lines_per_block)
        if ver:
            return gt_diff, buggy_code
        else:
            print("One of bug compositions failed:", err)

    return None, None


def diff_block_to_str(diff_block):
    """Convert a diff block (possibly multiline) to a hashable string for dedup."""
    parts = []
    for line_no in sorted(diff_block.keys(), key=lambda x: int(x)):
        v = diff_block[line_no]
        parts.append(f"{line_no}: {v['original']} --> {v['modified']}")
    return " | ".join(parts)


# Fields that describe one generated bug (or its single-bug bookkeeping) rather
# than the task; they are not copied onto composed multi-bug entries.
PER_BUG_FIELDS = {
    "task_id", "diff", "gt_diff", "buggy_code", "bug_count", "bug_type", "bug_subtype",
    "is_buggy", "atomicity_violation", "atomicity_reject_count", "FAIL_TO_PASS", "PASS_TO_PASS",
    "frozen_lines", "gt_length", "editable_lines", "deletable_lines",
}


def bug_compose(buggy_data, max_bugs, compose_per_example, stride=2, max_lines_per_block=1,
                multi_validation="block"):
    """
    Compose multi-bug variants from single-bug data.

    Groups verified single-bug entries by original task_id, deduplicates the
    per-task block pool, then for each k in 2..max_bugs samples k-tuples of
    non-adjacent blocks and applies them jointly to produce bug_count=k items.

    Composed entries carry the task-level fields of their source entries
    (e.g. `test`, or `repo`/`target_file` for SWE-smith) but not per-bug fields
    such as `bug_type` or `FAIL_TO_PASS`, which would describe only one of the
    composed bugs.

    :param max_lines_per_block: max lines per block (1 for single-line, >1 for multiline)
    :param multi_validation: "block" or "span" (see bug_generate)
    """
    merged = defaultdict(lambda: {
        "task_id": None,
        "gt_solution": None,
        "task_prompt": None,
        "diff_blocks": [],
        "task_fields": {},
    })

    filtered_buggy_data = []
    for entry in buggy_data:
        tid = entry["task_id"]
        if merged[tid]["task_id"] is None:
            merged[tid]["task_id"] = tid
            merged[tid]["gt_solution"] = entry["gt_solution"]
            merged[tid]["task_prompt"] = entry["task_prompt"]
            merged[tid]["task_fields"] = {"test": entry.get("test"),
                                          **{k: v for k, v in entry.items()
                                             if k not in PER_BUG_FIELDS
                                             and k not in ("gt_solution", "task_prompt")}}

        # For multiline, entry["diff"] is a full block dict (possibly multi-line)
        # NOTE: [design thought] dedup by diff string -- two LLM passes sometimes
        # produce the exact same bug; keeping both would bias composed samples.
        diff_str = diff_block_to_str(entry["diff"])
        existing_strs = [diff_block_to_str(b) for b in merged[tid]["diff_blocks"]]
        if diff_str not in existing_strs:
            gt_diff = file_diff(entry["buggy_code"], entry["gt_solution"], cleaned=True)[2]
            if multi_validation == "span" and max_lines_per_block > 1:
                ver, err = verify_span_diff(gt_diff, max_span=max_lines_per_block,
                                            min_changes=1, max_changes=max_lines_per_block)
            else:
                ver, err = verify_block_diff(gt_diff, block_count=1, max_lines_per_block=max_lines_per_block)
            if ver:
                # NOTE: [design thought] only single-block bugs enter the composition
                # pool; span-mode bugs spanning several blocks are kept as k=1 items.
                if len(parse_diff_to_blocks(entry["diff"])) == 1:
                    merged[tid]["diff_blocks"].append(entry["diff"])
                del entry["diff"]
                entry["gt_diff"] = gt_diff
                filtered_buggy_data.append(entry)
                print(f"new {diff_str} in {tid}")
            else:
                print(f"failed: {diff_str} for diff but {err} on gt diff in {tid}")
        else:
            print(f"same {diff_str} in {tid}")

    # NOTE: [design thought] k=1 reuses the single-bug pool directly; k>=2
    # is where composition actually happens. Each k gets its own bucket so
    # downstream sampling can balance across bug counts.
    all_bugs = {1: filtered_buggy_data}
    for k in range(2, max_bugs + 1):
        composed_buggy_data = copy.deepcopy(merged)
        for tid, composed_buggy_item in composed_buggy_data.items():
            new_gt_diff = []
            new_buggy_code = []
            gt_solution = composed_buggy_item["gt_solution"]
            diff_blocks = composed_buggy_item["diff_blocks"]

            for j in range(compose_per_example):
                gt_diff, buggy_code = compose_and_apply_diff(
                    gt_solution, k, diff_blocks, stride=stride,
                    max_lines_per_block=max_lines_per_block)
                if buggy_code and buggy_code not in new_buggy_code:
                    new_gt_diff.append(gt_diff)
                    new_buggy_code.append(buggy_code)

            composed_buggy_item["gt_diff"] = new_gt_diff
            composed_buggy_item["buggy_code"] = new_buggy_code

        k_bug_data = []
        for tid, item in composed_buggy_data.items():
            for gt_diff, buggy_code in zip(item["gt_diff"], item["buggy_code"]):
                # NOTE: [design thought] bug_count is the number of blocks, not lines.
                # For multiline, each block may span multiple lines but counts as 1 bug.
                blocks = parse_diff_to_blocks(gt_diff)
                entry = {
                    "task_id": item["task_id"],
                    "gt_solution": item["gt_solution"],
                    "task_prompt": item["task_prompt"],
                    "bug_count": len(blocks),
                    "gt_diff": gt_diff,
                    "buggy_code": buggy_code,
                    **copy.deepcopy(item["task_fields"]),
                }
                k_bug_data.append(entry)

        all_bugs[k] = k_bug_data

    print("Total buggy code generated: ", [len(all_bugs[i]) for i in range(1, max_bugs + 1)])

    id_counter = defaultdict(lambda: 0)
    all_bug_flatten = []
    for all_bug in all_bugs.values():
        for item in all_bug:
            task_id = item["task_id"] + f"_{id_counter[item['task_id']]}"
            id_counter[item["task_id"]] += 1
            item["task_id"] = task_id
            all_bug_flatten.append(item)

    return all_bug_flatten


def validate_and_sample(buggy_data, max_bugs, max_gen_per_bin=-1):
    """Final round-trip check + per-(task, bug_count) bucket sampling."""
    validated_buggy_data = []
    for d in buggy_data:
        d["buggy_code"] = rstrip_lines(d["buggy_code"])
        d["gt_solution"] = rstrip_lines(d["gt_solution"])
        # NOTE: [pedagogical] the invariant: applying gt_diff to buggy_code must
        # recover gt_solution exactly. If not, the diff is inconsistent and we drop it.
        if len(file_diff(apply_diff(d["buggy_code"], d["gt_diff"]), d["gt_solution"])[2]) > 0:
            print(f"Fail on data id: {d['task_id']}")
        else:
            validated_buggy_data.append(d)

    if max_gen_per_bin < 0:
        return validated_buggy_data

    # NOTE: [design thought] bucket by (original_task, bug_count) so every task
    # contributes a comparable number of examples at each difficulty level --
    # prevents a few prolific tasks from dominating the benchmark.
    orig_id_bins = defaultdict(lambda: {i: [] for i in range(1, max_bugs + 1)})
    for d in validated_buggy_data:
        orig_id = d["task_id"].rsplit("_", 1)[0]
        # span-mode bugs can have more blocks than max_bugs; give them their own bin
        orig_id_bins[orig_id].setdefault(d["bug_count"], []).append(d)

    orig_id_bins_filtered = defaultdict(list)
    for orig_id, all_examples in orig_id_bins.items():
        for bug_count, examples in all_examples.items():
            if len(examples) > max_gen_per_bin:
                selected_ids = sorted(np.random.choice(range(len(examples)), max_gen_per_bin, replace=False))
                orig_id_bins_filtered[orig_id] += [examples[i] for i in selected_ids]
            else:
                orig_id_bins_filtered[orig_id] += examples

    validated_buggy_data = list(itertools.chain.from_iterable(orig_id_bins_filtered.values()))
    bug_count_counter = Counter(d["bug_count"] for d in validated_buggy_data)

    print("Bug count distribution:")
    for k in sorted(bug_count_counter):
        print(f"  bug_count {k}: {bug_count_counter[k]}")

    return list(itertools.chain.from_iterable(orig_id_bins_filtered.values()))


def gen_main(args):
    """End-to-end orchestrator: load dataset -> generate -> compose -> sample -> save."""
    data_dir = os.path.join("dataset", args.dataset_name, "data")
    log_dir = os.path.join("results", args.dataset_name, "bug_data", "log")
    output_dir = os.path.join("results", args.dataset_name, "bug_data")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Add datetime
    time_to_add = datetime.datetime.now().strftime("%m%d-%H%M")

    log_file_prefix = os.path.join(log_dir, args.log_prefix) + "_" + time_to_add
    if args.rewrite:
        log_file_prefix += "_re"
    output_file = os.path.join(output_dir, args.output_prefix) + "_" + time_to_add + ".json"

    # Load the dataset
    if len(args.input_file) == 1:
        input_file = os.path.join(data_dir, args.input_file[0])
        raw_data = json.load(open(input_file, "r"))
    else:
        input_files = [os.path.join(data_dir, args.input_file[i]) for i in range(len(args.input_file))]
        raw_data_list = [json.load(open(input_file, "r")) for input_file in input_files]
        raw_data = raw_data_list[0]
        for d in raw_data_list[1:]:
            raw_data.extend(d)

    handler = get_handler(args.dataset_name)
    set_block_merge_mode(handler.merge_adjacent_add_blocks)
    print("Preprocessing data...")
    raw_data = handler.preprocess(raw_data)

    # Load the model
    if getattr(args, 'dry_run', False):
        # NOTE: [design thought] dspy >= 3.x requires the LM to be a dspy.BaseLM.
        # We use the library-provided DummyLM fed a canned "bug" response that
        # just wraps the original solution in a code fence. This lets us
        # exercise the generation pipeline end-to-end without API calls.
        from dspy.utils import DummyLM
        canned = "```python\n# mock dry-run response\npass\n```"
        mock_lm = DummyLM([{"subtype": "Others", "buggy_solution": canned}] * 10000)
        dspy.settings.configure(lm=mock_lm)
        print("DRY RUN: using DummyLM, no API calls will be made")
    else:
        api_key = resolve_api_key(args.model_name, args.model_api_file)
        lm_kwargs = dict(temperature=args.temperature, max_tokens=args.max_tokens)
        if api_key is not None:  # None means ambient credentials (e.g. vertex_ai/)
            lm_kwargs["api_key"] = api_key
        if args.reasoning_effort:
            lm_kwargs["reasoning_effort"] = args.reasoning_effort
        generator = dspy.LM(args.model_name, **lm_kwargs)
        dspy.settings.configure(lm=generator)

    # Generate bugs
    # NOTE: [design thought] `mode` is the master switch: single -> 1 line/block,
    # multi -> MIN_MULTILINES..MAX_MULTILINES lines/block. max_lpb caps the
    # per-block size and flows into bug_generate, bug_compose, and verification.
    mode = getattr(args, 'mode', 'single')
    max_lpb = getattr(args, 'max_lines_per_block', MAX_MULTILINES if mode == 'multi' else 1)
    print(f"Bug generation with model: {args.model_name} (mode={mode}, max_lines_per_block={max_lpb})")
    handler.mark_editable_lines(raw_data)

    # Optional test-adequacy gate (paper §3.3): restrict injection to lines the
    # suite executes, optionally augmenting the suite of gate-failing tasks first.
    # Inputs already annotated by src/test_adequacy.py are filtered the same way.
    if args.coverage_gate:
        lm_ctx = dspy.settings.lm
        if args.augment_tests and args.augment_model_name:
            aug_key = resolve_api_key(args.augment_model_name, None)
            dspy.settings.configure(lm=dspy.LM(args.augment_model_name, max_tokens=args.max_tokens,
                                               **({"api_key": aug_key} if aug_key else {})))
        run_test_adequacy(raw_data, handler, augment=args.augment_tests,
                          property_based=args.property_based, n_workers=args.n_workers_llm)
        dspy.settings.configure(lm=lm_ctx)
    else:
        excluded = apply_coverage_filter(raw_data)
        if excluded:
            print(f"Coverage annotations found: excluded {excluded} unexecuted editable lines")

    _, buggy_data = bug_generate(raw_data, handler, log_file_prefix, args.bug_per_time,
                                 mode=mode, max_lines_per_block=max_lpb,
                                 multi_validation=args.multi_validation,
                                 n_workers_llm=args.n_workers_llm,
                                 n_workers_validation=args.n_workers_validation,
                                 atomicity_full_max_lines=args.atomicity_full_max_lines)

    # Compose bugs
    composed_buggy_data = bug_compose(buggy_data, args.max_bugs, args.bug_per_time,
                                      stride=args.stride, max_lines_per_block=max_lpb,
                                      multi_validation=args.multi_validation)
    validated_buggy_data = validate_and_sample(composed_buggy_data, args.max_bugs, args.max_gen_per_bin)

    # Save the buggy code
    print("Saving validated composed buggy code to:", output_file)
    with open(output_file, "w") as f:
        json.dump(validated_buggy_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Dataset name", required=True)
    parser.add_argument("--model_name", type=str, help="Bug generation model name", required=True)
    parser.add_argument("--model_api_file", type=str, default=None,
                        help="Model API file under keys/ (optional, auto-resolved from model name)")
    parser.add_argument("--input_file", nargs='+', help="Input file path, under dataset/{dataset_name}/data",
                        default=["full_data.json"])
    parser.add_argument("--log_prefix", type=str, help="Log file prefix", default="log")
    parser.add_argument("--output_prefix", type=str, help="Output file prefix", default="buggy_code")
    parser.add_argument("--rewrite", action="store_true", help="Whether to rewrite the code")
    parser.add_argument("--bug_per_time", type=int, default=20, help="Number of bugs to add per iteration")
    parser.add_argument("--max_bugs", type=int, default=4, help="Max number of bugs to compose")
    parser.add_argument("--max_gen_per_bin", type=int, default=5,
                        help="Max number of generated buggy code per example per bug count")
    parser.add_argument("--stride", type=int, default=2,
                        help="Minimum line distance between composed bugs")
    parser.add_argument("--max_tokens", type=int, default=32000, help="Maximum number of tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the generator")
    parser.add_argument("--dry_run", action="store_true",
                        help="Replace LLM with a mock (no API credit used)")
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                        help="Bug mode: 'single' for single-line bugs, 'multi' for multiline blocks")
    parser.add_argument("--max_lines_per_block", type=int, default=MAX_MULTILINES,
                        help=f"Max lines per bug block in multiline mode (default {MAX_MULTILINES})")
    parser.add_argument("--multi_validation", choices=["block", "span"], default="block",
                        help="Multiline bug shape check: 'block' = one contiguous block (default); "
                             "'span' = all edits within a max_lines_per_block window (SWE-smith)")
    parser.add_argument("--atomicity_full_max_lines", type=int, default=MAX_MULTILINES,
                        help="Bugs with at most this many edited lines are checked against every "
                             "partial repair; larger bugs only against single-line reverts")
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        choices=["none", "minimal", "low", "medium", "high", "xhigh", "default"],
                        help="Reasoning effort for models that support it (e.g. 'high')")
    parser.add_argument("--n_workers_llm", type=int, default=1,
                        help="Parallel LLM calls during generation")
    parser.add_argument("--n_workers_validation", type=int, default=4,
                        help="Parallel validation workers (used by the SWE-smith Docker harness)")
    # Optional test-adequacy gate (src/test_adequacy.py)
    parser.add_argument("--coverage_gate", action="store_true",
                        help="Exclude editable lines never executed by the test suite from injection")
    parser.add_argument("--augment_tests", action="store_true",
                        help="With --coverage_gate: LLM-generate tests for tasks that fail the gate")
    parser.add_argument("--augment_model_name", type=str, default=None,
                        help="LLM for test augmentation (default: --model_name)")
    parser.add_argument("--property_based", action="store_true",
                        help="With --augment_tests: ask for Hypothesis property-based tests")

    logging.getLogger().setLevel(logging.ERROR)

    args = parser.parse_args()
    gen_main(args)
