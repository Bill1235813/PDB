"""
Evaluation pipeline for PDB (Precise Debugging Benchmarking).

Computes two metrics for model debugging results:
  1. Unit score (F_U): binary pass/fail on the model's predicted solution.
  2. Symbolic block scores: edit-level precision, bug-level recall, and F1,
     computed by matching predicted edits against ground-truth edits using a
     multi-pass symbolic alignment algorithm (see paper Section 3).

The key insight is that conventional unit-test accuracy cannot distinguish a
precise one-line fix from a full program rewrite. Our precision/recall metrics
operate on the *diff* between buggy code and the model's output, measuring how
closely the model's edits align with the ground-truth minimal corrections.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import copy

from dataset import get_handler
from utils import file_diff, parse_diff_to_blocks, verify_block_single_diff, verify_block_diff, expand_blocks_to_diff, rstrip_lines, \
    apply_diff, set_block_merge_mode
from config import DEFAULT_TOLERANCE_MULTILINE, DEFAULT_TOLERANCE_SINGLELINE, EVAL_MAX_LINES_PER_BLOCK
from collections import defaultdict
from argparse import ArgumentParser


class Evaluator:
    def __init__(self, args, results=None):
        self.dataset = args.dataset_name
        self.handler = get_handler(args.dataset_name)
        set_block_merge_mode(self.handler.merge_adjacent_add_blocks)
        self.output_dir = os.path.join(args.eval_result_dir, args.dataset_name, "eval_results")
        self.log_dir = os.path.join(self.output_dir, "log")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_name = args.eval_model_name
        self.eval_set_name = args.eval_set_name
        self.stride = args.stride
        # `tolerance=N` means each matched GT block tolerates N extra predicted
        # lines for full precision credit. Default: 1 for multiline mode,
        # 2 for single-line mode.
        self.tolerance = args.tolerance
        self.round = 0
        assert self.tolerance >= 0, "tolerance must be >= 0"
        self.gt, self.buggy_code, self.pred, self.gt_diff, self.pred_diff, self.eval_ids, self.eval_results = [], [], [], [], [], [], []

        self.metrics = {
            "Unit score": self.unit_score,
            "Symbolic block scores": self.symbolic_block_score,
        }
        self.scores = {metrics: {} for metrics in self.metrics}
        self.results = []
        self.error_msg = None

        if results:
            self.result_formatting(results)

    def result_formatting(self, results):
        """
        Parse raw debugging results into parallel lists for evaluation.

        For each result entry, extracts and validates:
          - gt_diff: the ground-truth diff (buggy_code -> gt_solution)
          - buggy_code, gt_solution, predicted solution
          - pred_diff: the predicted diff (buggy_code -> model's output)

        The gt_diff is validated against the expected bug_count and stride
        to ensure the evaluation input is well-formed.
        """
        self.gt, self.buggy_code, self.pred, self.gt_diff, self.pred_diff = [], [], [], [], []
        final_results = {}
        for s in results:
            if "gt_diff" in s:
                gt_diff = s["gt_diff"]
            else:
                _, _, gt_diff = file_diff(s["buggy_code"], s["gt_solution"], cleaned=True)
            # Validate: correct block count + per-block line cap.
            # NOTE: [design thought] Failures become a skip-with-warning. The stride
            # gap is not re-checked: span-mode multi-line bugs (code moves) may have
            # blocks closer than the composition stride, which generation enforces.
            ver_gt_diff, err_str = verify_block_diff(gt_diff, block_count=s["bug_count"],
                                                     stride=0,
                                                     max_lines_per_block=EVAL_MAX_LINES_PER_BLOCK)
            # NOTE: [edge case callout] results are keyed by task_id; a duplicate id
            # would silently replace an earlier example and shrink the eval set.
            if s["task_id"] in final_results:
                raise ValueError(f"Duplicate task_id {s['task_id']!r} in evaluation input; "
                                 f"task ids must be unique.")
            if ver_gt_diff:
                final_results[s["task_id"]] = s
                self.gt_diff.append(gt_diff)
            else:
                print(f"[skip] {s.get('task_id', '?')}: gt_diff failed verify_block_diff: {err_str}")
                continue

            if "gt_solution" in s:
                self.gt.append(rstrip_lines(s["gt_solution"]))
            else:
                raise KeyError("No GT solution found.")
            if "buggy_code" in s:
                self.buggy_code.append(rstrip_lines(s["buggy_code"]))
            else:
                raise KeyError("No buggy code found.")
            if "debug_results" in s and "solution" in s["debug_results"]:
                self.pred.append(rstrip_lines(s["debug_results"]["solution"]))
            else:
                raise KeyError("No predicted solution found.")

            if "debug_results" in s and "pred_diff" in s["debug_results"]:
                self.pred_diff.append(s["debug_results"]["pred_diff"])
            else:
                _, _, pred_diff = file_diff(s["buggy_code"], s["debug_results"]["solution"], cleaned=True)
                self.pred_diff.append(pred_diff)

        self.eval_ids = list(final_results.keys())
        self.eval_results = list(final_results.values())
        if self.results:
            self.results = [final_results.get(item["task_id"], item) for item in self.results]
        else:
            self.results = list(final_results.values())

    def run_evaluation(self, results=None, round=None):
        if results:
            self.result_formatting(results)
        if not self.results:
            print("No task to evaluate!")
            return self.scores
        else:
            if round:
                self.round = round
            else:
                self.round = self.results[0]["round"]

            for name, metric in self.metrics.items():
                print(f"{name}:", metric(name))
            self.save_results()
            self.print_summary()
            return self.scores

    def success_unit(self, task_id):
        return self.scores["Unit score"].get(task_id, 0) == 1

    def unit_score(self, metric_name):
        """
        Compute unit-test accuracy: F_U(C) -> {0, 1}.

        Runs the model's predicted solution against all unit tests via the
        dataset handler. A score of 1 means the solution passes every test.

        NOTE: [pedagogical] This is the conventional evaluation metric for code
        generation. It tells us *whether* the model fixed the bugs, but not
        *how* — a full rewrite and a minimal one-line fix both score 1.
        """
        print("Compute unit test score:")
        verify_file = self.handler.build_verify_unit_test(
            self.log_dir + f"/{self.model_name}_unit_test_verify",
            [{"task_id": idx, "solution": pred} for idx, pred in zip(self.eval_ids, self.pred)])
        formatted_gt = self.handler.save_formatted_gt(
            self.log_dir + f"/{self.model_name}_unit_test_gt",
            self.eval_results)
        fail_ids, correct_ids, fail_feedback = self.handler.verify_unit_test(
            verify_file, gt_file=formatted_gt, timeout=1800)
        # NOTE: [edge case callout] pair ids with feedback while both are still
        # ordered lists; zipping after set() would attach messages to wrong tasks.
        if isinstance(fail_feedback, dict):
            self.error_msg = {idx: fail_feedback.get(idx, "") for idx in fail_ids}
        else:
            self.error_msg = {ids: feedback for ids, feedback in zip(fail_ids, fail_feedback)}
        fail_ids, correct_ids = set(fail_ids), set(correct_ids)
        for idx in self.eval_ids:
            self.scores[metric_name][idx] = 1 if idx in correct_ids else 0
        return metric_name, sum(self.scores[metric_name].values()) / len(self.results)

    def _is_equal(self, pred_edit, gt_edit):
        """Check if a predicted edit is the equal to a ground truth edit."""
        return (pred_edit['original'] == gt_edit['original'] and
                pred_edit['modified'] == gt_edit['modified'])

    def _line_set_match(self, pred_lines, gt_lines, top_btm_line):
        """
        Check if two sets of surrounding code lines share enough context.

        Used by Pass 2.2 (near match) to determine if a predicted block and a
        GT block occupy the same local code region, even if their line numbers
        differ slightly. Two blocks are considered "near" if both their
        preceding and following context lines overlap.

        NOTE: [edge case callout] Blank/whitespace-only lines are stripped from
        both sets before intersecting. Otherwise a single empty line (which
        appears many times in any Python source) would trivially satisfy the
        intersection test, causing Pass 2.2 to spuriously match pred blocks
        that are actually far from the GT block.
        """
        pred_lines = {ln for ln in pred_lines if ln and ln.strip()}
        gt_lines = {ln for ln in gt_lines if ln and ln.strip()}
        if gt_lines & pred_lines:
            return True
        elif not gt_lines and not pred_lines:
            return True
        elif not gt_lines and top_btm_line in pred_lines:
            return True
        elif not pred_lines and top_btm_line in gt_lines:
            return True
        else:
            return False

    def symbolic_block_score(self, metric_name):
        """
        Compute edit-level precision, bug-level recall, and F1 via symbolic
        block matching (the map function from the paper).

        The algorithm aligns predicted edits (Ê) against ground-truth edits
        (E_gt) using three progressively more flexible matching passes,
        followed by unit-test-based semantic verification and a deep
        redundancy check.

        -----------------------------------------------------------------------
        Overview of the three-pass matching algorithm (map function):
        -----------------------------------------------------------------------

        Pass 1 — Exact Match (EM):
          For each predicted edit at line ℓ with content v, if the GT diff has
          the exact same edit at line ℓ, it's a trivial match. No unit tests
          needed. These are removed from further consideration.

        Pass 2 — Block-Level Matching (three sub-strategies):
          Remaining edits are grouped into contiguous blocks. For each
          predicted block, we try three strategies in order:

          2.1 Wrap Match: Does the predicted block structurally contain (wrap)
              a GT block? i.e., pred_start <= gt_start <= pred_end.

          2.2 Near Match: Do the predicted and GT blocks share the same local
              code context? We compare the lines *before* and *after* each
              block. If both sides overlap, the blocks target the same region.

          2.3 Distant-But-Identical: For single-edit predicted blocks, is there
              any GT block with identical edit content, regardless of location?

          All Pass 2 candidates require unit-test verification: we construct a
          test program by applying all GT fixes *except* the matched GT block,
          plus the predicted block. If F_U(test_program) = 1, the match is
          semantically verified.

        -----------------------------------------------------------------------
        Precision and Recall:
        -----------------------------------------------------------------------

        For each bug i with ground-truth edits E_i, construct a pseudo-revision:
          Ĉ_i = apply((E_gt \ E_i) ∪ map(E_i), C_buggy)
        If F_U(Ĉ_i) = 1, bug i is considered "resolved".

        Recall = (1/k) × Σ F_U(Ĉ_i)
          Bug-level: fraction of bugs correctly fixed, regardless of edit count.

        Precision = (1/|Ê|) × Σ F_U(Ĉ_i) × (|Ê_i|)_ε
          Edit-level: for each resolved bug, count how many of the predicted
          edits were essential (up to tolerance ε). Divided by total predicted
          edits. Penalizes unnecessary edits — a full rewrite yields ~0 precision.

        -----------------------------------------------------------------------
        Deep Redundancy Check (ess_U, tolerance / epsilon):
        -----------------------------------------------------------------------

        When a predicted block is larger than the GT block it matches (e.g.,
        model edited 4 lines to fix a 1-line bug), we check whether a smaller
        contiguous sub-sequence of those edits would still pass tests. This
        finds the *minimal essential edit size* within each predicted block.

        The `tolerance` parameter sets the maximum number of extra predicted
        lines per matched GT block that receive full precision credit. If the
        model's essential edits are within `tolerance` lines of the GT size,
        precision is not penalized.
        """
        print("Compute precision and recall:")
        equ_test = []
        matched_blocks = defaultdict(dict)
        unmatched_gt = defaultdict(dict)
        unmatched_pred = defaultdict(dict)
        all_gt_blocks = {task_id: parse_diff_to_blocks(gt_diff) for task_id, gt_diff in
                         zip(self.eval_ids, self.gt_diff)}
        all_buggy = {task_id: buggy for task_id, buggy in zip(self.eval_ids, self.buggy_code)}

        for task_id, gt_diff, pred_diff in zip(self.eval_ids, self.gt_diff, self.pred_diff):
            remain_gt_diff = copy.deepcopy(gt_diff)
            remain_pred_diff = copy.deepcopy(pred_diff)
            remain_gt_blocks = parse_diff_to_blocks(remain_gt_diff)[::-1]
            buggy = all_buggy[task_id]

            # ---- Pass 1: Exact BLOCK-level matches (EM) ----
            # A pred block is EM-matched only if there is a GT block with the
            # SAME line range AND identical per-line edit content. This avoids
            # the single-line/multi-line ambiguity where a 1-line predicted fix
            # inside a 2-line GT block used to (incorrectly) claim the whole GT
            # block as matched. For single-line bugs every block is 1 line
            # wide, so this is equivalent to the old line-level EM.
            count_em = 0
            pred_blocks_pass1 = parse_diff_to_blocks(remain_pred_diff)[::-1]
            for pred_block in pred_blocks_pass1:
                match = None
                for gt_block in remain_gt_blocks:
                    if (pred_block["block_start"] == gt_block["block_start"]
                            and pred_block["block_end"] == gt_block["block_end"]
                            and len(pred_block["diff"]) == len(gt_block["diff"])
                            and all(
                                k in gt_block["diff"]
                                and self._is_equal(pred_block["diff"][k], gt_block["diff"][k])
                                for k in pred_block["diff"]
                            )):
                        match = gt_block
                        break
                if match is None:
                    continue
                matched_blocks[task_id][f"{task_id}_em_{count_em}"] = {
                    "block_start": pred_block["block_start"],
                    "block_end": pred_block["block_end"],
                    "diff": dict(pred_block["diff"]),
                    "block_id": -1,
                    "success": True,
                    "gt_match_count": 1,
                    "tolerance": 0
                }
                count_em += 1
                for k in pred_block["diff"]:
                    remain_gt_diff.pop(k, None)
                    remain_pred_diff.pop(k, None)
                remain_gt_blocks = [b for b in remain_gt_blocks if b is not match]

            # ---- Pass 2: Block-level matching ----
            # Group remaining edits into contiguous blocks and try three
            # alignment strategies in order of decreasing strictness.
            remain_pred_blocks = parse_diff_to_blocks(remain_pred_diff)[::-1]
            buggy_lines = buggy.splitlines()
            for b_no, pred_block in enumerate(remain_pred_blocks):

                # 2.1 Wrap Match: check if a predicted block structurally
                # contains a GT block (pred_start <= gt_start <= pred_end).
                matched_gt_block = []
                for gt_block in remain_gt_blocks:
                    if pred_block["block_start"] <= gt_block["block_start"] <= pred_block["block_end"]:
                        matched_gt_block.append(gt_block)

                # 2.2 Near Match: compare surrounding code context. Extract
                # the set of code lines before and after each block, then check
                # if both sides overlap. This handles cases where the model
                # edits the right region but at a slightly different line offset.
                if not len(matched_gt_block):
                    if b_no < len(remain_pred_blocks) - 1:
                        pre_pred_lines = set(
                            [buggy_lines[idx - 1] for idx in
                             range(remain_pred_blocks[b_no + 1]["block_end"] + 1, pred_block["block_start"])])
                    else:
                        pre_pred_lines = set([buggy_lines[idx - 1] for idx in range(0, pred_block["block_start"])])
                    if b_no > 0:
                        post_pred_lines = set(
                            [buggy_lines[idx - 1] for idx in
                             range(pred_block["block_end"] + 1, remain_pred_blocks[b_no - 1]["block_start"])])
                    else:
                        post_pred_lines = set(
                            [buggy_lines[idx - 1] for idx in range(pred_block["block_end"] + 1, len(buggy_lines))])
                    for gt_block in remain_gt_blocks:
                        gt_pre_stride = gt_block["stride_before"]
                        gt_post_stride = gt_block["stride_after"] if gt_block["stride_after"] \
                            else len(buggy_lines) - gt_block["block_start"]
                        pre_gt_lines = set(
                            [buggy_lines[gt_block["block_start"] - idx - 1] for idx in range(1, gt_pre_stride) if
                             gt_block["block_start"] - idx > 0])
                        post_gt_lines = set(
                            [buggy_lines[gt_block["block_start"] + idx - 1] for idx in range(1, gt_post_stride) if
                             gt_block["block_start"] + idx <= len(buggy_lines)])
                        if (self._line_set_match(pre_pred_lines, pre_gt_lines, buggy_lines[0]) and
                                self._line_set_match(post_pred_lines, post_gt_lines, buggy_lines[-1])):
                            matched_gt_block.append(gt_block)
                            break

                # 2.3 Distant-But-Identical: check if any GT block has
                # identical edit content regardless of line number. For
                # single-edit blocks, compare the one edit. For multiline
                # blocks, compare all edits in order.
                if not len(matched_gt_block):
                    pred_edits = list(pred_block["diff"].values())
                    for gt_block in remain_gt_blocks:
                        gt_edits = list(gt_block["diff"].values())
                        if len(pred_edits) == len(gt_edits) and all(
                                self._is_equal(p, g) for p, g in zip(pred_edits, gt_edits)):
                            matched_gt_block.append(gt_block)
                            break

                # ---- Semantic verification via unit tests ----
                # For each candidate match from Pass 2, construct a test program:
                # apply all GT fixes EXCEPT the matched GT block(s), then apply
                # the predicted block instead. If the test program passes all
                # unit tests, the match is semantically verified.
                if len(matched_gt_block):
                    matched_gt_block_ids = [b["block_id"] for b in matched_gt_block]
                    test_block = ([b for b in all_gt_blocks[task_id] if b["block_id"] not in matched_gt_block_ids] +
                                  [pred_block])
                    test_diff = expand_blocks_to_diff(test_block, ordered=False)
                    test_solution = apply_diff(buggy, test_diff)
                    equ_test.append({
                        "task_id": f"{task_id}_{b_no}",
                        "solution": test_solution
                    })
                    remain_gt_blocks = [b for b in remain_gt_blocks if b not in matched_gt_block]

                    # NOTE: [design thought]
                    # Tolerance is a PER-BLOCK constant (self.tolerance), scaled by the
                    # number of GT blocks this prediction wraps. A 3-line GT block
                    # matched alone gets the SAME tolerance as a 1-line GT block.
                    # Only when a single predicted block covers multiple GT blocks
                    # does tolerance scale up.
                    # Example: tolerance=1, GT=3 lines, pred=4 lines -> 1 extra allowed
                    # -> full precision. Pred wrapping 2 GT blocks -> 2 extra allowed.
                    num_matched_blocks = len(matched_gt_block)
                    gt_lines_total = sum(len(gb["diff"]) for gb in matched_gt_block)
                    matched_blocks[task_id][f"{task_id}_{b_no}"] = {
                        "pred_block": pred_block,
                        "gt_blocks": matched_gt_block,
                        "gt_match_ids": matched_gt_block_ids,
                        "gt_match_count": num_matched_blocks,
                        "tolerance": max(min(
                            self.tolerance * num_matched_blocks,
                            len(pred_block["diff"]) - gt_lines_total),
                            0)
                    }
                else:
                    # No GT block matched — these are spurious predicted edits
                    # that will penalize precision.
                    unmatched_pred[task_id] |= expand_blocks_to_diff([pred_block])

            # Any GT blocks still unmatched count against recall.
            unmatched_gt[task_id] = expand_blocks_to_diff(remain_gt_blocks)

        # ---- Semantic equivalence check ----
        # Run unit tests on all candidate matches from Pass 2 to verify that
        # replacing the GT fix with the predicted fix still produces a correct
        # program. This is the F_U(Ĉ_i) test from the paper.
        if len(equ_test) > 0:
            print("Semantic equivalence check:")
            verify_file = self.handler.build_verify_unit_test(
                self.log_dir + f"/{self.model_name}_equ_test_verify",
                equ_test,
                sol_field="solution")
            temp_gt = self.handler.save_formatted_gt(
                self.log_dir + f"/{self.model_name}_equ_test_gt",
                equ_test)
            fail_ids, correct_ids, _ = self.handler.verify_unit_test(verify_file, gt_file=temp_gt, timeout=1800)

            # ---- Deep redundancy check (ess_U from the paper) ----
            # For each verified match where the predicted block is larger than
            # the GT block, enumerate all contiguous sub-sequences of the
            # predicted edits and test each one. This finds the *minimal
            # essential edit size*: the fewest contiguous lines within the
            # predicted block that still fix the bug.
            #
            # Example: model predicted a 4-line edit to fix a 1-line bug.
            # We test all sub-sequences: lines [1], [2], [3], [4], [1,2],
            # [2,3], [3,4], etc. If line [2] alone passes, the essential
            # size is 1, and precision is not penalized.
            redun_test = []
            for correct_id in correct_ids:
                try:
                    task_id = correct_id.rsplit('_', 1)[0]
                    matched_blocks[task_id][correct_id]["success"] = True
                    if matched_blocks[task_id][correct_id]["tolerance"] > 0:
                        gt_match_ids = matched_blocks[task_id][correct_id]
                        pred_block = copy.deepcopy(matched_blocks[task_id][correct_id]["pred_block"])
                        pred_diff = list(pred_block["diff"].items())
                        # Enumerate contiguous sub-sequences of increasing size
                        # (from 1 line up to tolerance lines)
                        for tol in range(matched_blocks[task_id][correct_id]["tolerance"]):
                            for test_count in range(len(pred_diff) - tol):
                                pred_block["diff"] = dict(pred_diff[test_count:test_count + tol + 1])
                                pred_block["block_start"] = min([int(k) for k in pred_block["diff"].keys()])
                                pred_block["block_end"] = max([int(k) for k in pred_block["diff"].keys()])
                                test_block = ([b for b in all_gt_blocks[task_id] if b["block_id"] not in gt_match_ids] +
                                              [pred_block])
                                test_diff = expand_blocks_to_diff(test_block, ordered=False)
                                test_solution = apply_diff(all_buggy[task_id], test_diff)
                                redun_test.append({
                                    "task_id": f"{correct_id}_{tol}_{test_count}",
                                    "solution": test_solution
                                })
                except ValueError:
                    print(f"Warning: Could not parse task_id and line from '{correct_id}'")

            # Failed semantic checks: the predicted block does NOT fix the bug,
            # so this match is revoked — contributes 0 to both precision and recall.
            for fail_id in fail_ids:
                try:
                    task_id = fail_id.rsplit('_', 1)[0]
                    matched_blocks[task_id][fail_id]["success"] = False
                    matched_blocks[task_id][fail_id]["tolerance"] = 0
                    matched_blocks[task_id][fail_id]["gt_match_count"] = 0
                except ValueError:
                    print(f"Warning: Could not parse task_id and line from '{fail_id}'")

            # Run the deep redundancy tests and find the minimum valid sub-edit
            if len(redun_test) > 0:
                print("Deep redundancy check:")
                verify_file = self.handler.build_verify_unit_test(
                    self.log_dir + f"/{self.model_name}_redun_test_verify",
                    redun_test,
                    sol_field="solution")
                temp_gt = self.handler.save_formatted_gt(
                    self.log_dir + f"/{self.model_name}_redun_test_gt",
                    redun_test)
                # NOTE: [edge case callout] Redundancy candidates scale with the
                # model's over-edit rate. GPT-5.1-Codex on LiveCodeBench produces
                # ~260 candidates whose unit-test verification can exceed 30 min;
                # we raise the timeout to 60 min here while keeping the equivalence
                # and initial-verify calls at the default 1800s.
                fail_ids, correct_ids, _ = self.handler.verify_unit_test(
                    verify_file, gt_file=temp_gt, timeout=3600)

                # For each match, find the smallest sub-sequence size (τ*) that
                # passes unit tests. This becomes the essential edit size used
                # in the ε-relaxed precision formula.
                min_tol_by_prefix = {}
                for correct_id in correct_ids:
                    prefix, tol_str, test_count = correct_id.rsplit('_', 2)
                    tol = int(tol_str)
                    if prefix not in min_tol_by_prefix or tol < min_tol_by_prefix[prefix][0]:
                        min_tol_by_prefix[prefix] = (tol, test_count)
                for prefix in min_tol_by_prefix:
                    task_id = prefix.rsplit('_', 1)[0]
                    matched_blocks[task_id][prefix]["tolerance"] = min_tol_by_prefix[prefix][0]
                    matched_blocks[task_id][prefix]["effective_starter"] = min_tol_by_prefix[prefix][1]

        # ---- Compute final precision, recall, F1 per task ----
        #
        # Recall = (1/k) × Σ F_U(Ĉ_i)
        #   = (number of bugs with successful matches) / (total bugs)
        #   where gt_match_count tracks how many GT bugs each match resolves.
        #
        # Precision = (1/|Ê|) × Σ F_U(Ĉ_i) × (|Ê_i|)_ε
        #   = (matched GT edits + tolerance credit) / (total predicted edits)
        #   where tolerance credit accounts for the essential edit size found
        #   by the deep redundancy check.
        for task_id, gt_diff, pred_diff in zip(self.eval_ids, self.gt_diff, self.pred_diff):

            if not gt_diff:
                raise ValueError(f"Ground truth empty for {task_id}")
            elif not pred_diff:
                precision, recall, f1 = 0.0, 0.0, 0.0
            else:
                # NOTE: [design thought] count recall at the
                # block level. The EM pass creates one entry per matched LINE,
                # but recall should count matched GT BLOCKS. We track which GT
                # blocks have been matched (by block_start) and count unique ones.
                gt_blocks = parse_diff_to_blocks(gt_diff)
                actual_pos_blocks = len(gt_blocks)
                predicted_pos_lines = len(pred_diff)
                tolerance = sum([match["tolerance"] for _, match in matched_blocks[task_id].items()])

                # Map each EM-matched line to its parent GT block. Two EM
                # lines in the same GT block should count as 1 block match.
                matched_gt_block_ids = set()
                matched_gt_lines = 0
                for _, match in matched_blocks[task_id].items():
                    if match.get("success", True) and match.get("gt_match_count", 0) > 0:
                        if "gt_blocks" in match:
                            for gb in match["gt_blocks"]:
                                matched_gt_block_ids.add(gb["block_id"])
                                matched_gt_lines += len(gb["diff"])
                        else:
                            # EM match: find the parent GT block by line number
                            # and count the actual number of matched lines (an
                            # EM block can span multiple lines after the
                            # block-level EM fix).
                            em_line = match["block_start"]
                            for gb in gt_blocks:
                                if gb["block_start"] <= em_line <= gb["block_end"]:
                                    matched_gt_block_ids.add(gb["block_id"])
                                    break
                            matched_gt_lines += len(match.get("diff", {})) or 1

                true_pos_blocks = len(matched_gt_block_ids)

                precision = min((matched_gt_lines + tolerance) / predicted_pos_lines, 1.0) if predicted_pos_lines > 0 else 0.0
                recall = min(true_pos_blocks / actual_pos_blocks, 1.0) if actual_pos_blocks > 0 else 0.0

                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

            self.scores[metric_name][task_id] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matched_blocks": matched_blocks[task_id],
                "unmatched_pred": unmatched_pred[task_id],
                "unmatched_gt": unmatched_gt[task_id]
            }

        avg_precision = sum([eval_score["precision"] for _, eval_score in self.scores[metric_name].items()]) / len(
            self.results)
        avg_recall = sum([eval_score["recall"] for _, eval_score in self.scores[metric_name].items()]) / len(
            self.results)
        avg_f1 = sum([eval_score["f1"] for _, eval_score in self.scores[metric_name].items()]) / len(self.results)

        return "Precision", avg_precision, "Recall", avg_recall, "F1", avg_f1

    def save_results(self):
        save_file = self.output_dir + f"/{self.model_name}_on_{self.eval_set_name}_round_{self.round}_scores.json"
        print(f"Saving results to {save_file}")
        if self.results:
            with open(save_file, "w") as f:
                json.dump(self.scores, f, indent=2)

    def print_summary(self):
        """Print a one-line per-dataset summary of the current scores.
        NOTE: [design thought] Centralized here so every caller (bug_correct.py's
        per-round loop, evaluator.py standalone, and the driver shell scripts)
        prints the same line shape. Drivers then just aggregate these into a
        union across datasets.
        """
        unit = self.scores.get("Unit score", {}) or {}
        sym = self.scores.get("Symbolic block scores", {}) or {}
        n = len(unit)
        if n == 0:
            print(f"[summary] {self.model_name} on {self.eval_set_name} round {self.round}: (no results)")
            return
        u = sum(unit.values())
        p = sum(v["precision"] for v in sym.values())
        r = sum(v["recall"]    for v in sym.values())
        f = sum(v["f1"]        for v in sym.values())
        print(f"[summary] {self.model_name} on {self.eval_set_name} round {self.round}: "
              f"unit={u/n:.3f} prec={p/n:.3f} rec={r/n:.3f} f1={f/n:.3f} (n={n})")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Dataset name", required=True)
    parser.add_argument("--input_file", nargs='+', help="Input buggy file path, under output/{dataset_name}",
                        required=True)
    parser.add_argument("--eval_result_dir", type=str, default="results")
    parser.add_argument("--eval_set_name", type=str, default=None)
    parser.add_argument("--eval_model_name", type=str, default=None)
    parser.add_argument("--stride", type=int, default=2, help="Minimum stride between bug diffs")
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                        help="Bug mode. Controls the --tolerance default when not set explicitly.")
    parser.add_argument("--tolerance", type=int, default=None,
                        help=f"Per-block tolerance: extra predicted lines allowed per matched GT "
                             f"block for full precision credit. 0=strict. If unset, defaults to "
                             f"{DEFAULT_TOLERANCE_SINGLELINE} in --mode single and "
                             f"{DEFAULT_TOLERANCE_MULTILINE} in --mode multi.")
    parser.add_argument("--max_iter", type=int, default=1, help="Maximum number of add-bug iterations")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Parallel sandbox workers for handlers that support it (SWE-smith Docker runs)")
    parser.add_argument("--reload_first_round", action="store_true", help="Whether to reload first round results")
    parser.add_argument("--reload_result_file", type=str, default=1, help="The result file to reload")
    parser.add_argument("--reload_score_file", type=str, default=1, help="The score file to reload")

    args = parser.parse_args()
    # Resolve --tolerance default from --mode if not explicitly passed.
    if args.tolerance is None:
        args.tolerance = (DEFAULT_TOLERANCE_MULTILINE if args.mode == "multi"
                          else DEFAULT_TOLERANCE_SINGLELINE)
    if args.eval_model_name is None:
        args.eval_model_name = args.input_file[0].split("/")[-1].split("_on_")[0]
    else:
        args.eval_model_name = args.eval_model_name.split("/")[-1]
    if args.eval_set_name is None:
        args.eval_set_name = os.path.splitext(args.input_file[0])[0].split("_on_")[-1]

    evaluator = Evaluator(args)
    if args.n_workers and hasattr(evaluator.handler, "n_workers"):
        evaluator.handler.n_workers = args.n_workers

    for i, in_file in enumerate(args.input_file):
        rd = i + 1
        if rd > args.max_iter:
            break
        if not os.path.isabs(in_file) and not os.path.exists(in_file):
            in_file = os.path.join("results", args.dataset_name, "debug_results", in_file)
        if rd == 1 and args.reload_first_round:
            print(f"Skip round {rd}")
            results = json.load(open(args.reload_result_file))
            scores = json.load(open(args.reload_score_file))
            evaluator.result_formatting(results)
            evaluator.scores = scores
        else:
            results = json.load(open(in_file))
            print(f"Evaluate round {rd}")
            evaluator.run_evaluation(results=results, round=rd)
