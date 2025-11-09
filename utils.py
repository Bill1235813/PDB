import copy
import os
import json
import subprocess
from copy import deepcopy
from pathlib import Path
import re
import time
import signal
import textwrap

import numpy as np
import pytest
import difflib
from module import SIMPLE_CODE_BLOCK_REGEX


def rstrip_lines(code_str):
    """
    Format code to remove right indentation

    :param code_str: original code string
    :return: formatted_code with right indentation removed
    """
    return "\n".join([l.rstrip() for l in code_str.splitlines()])


# def check_key_words(code_str):


def file_diff(str1, str2, cleaned=False):
    """
    Compare two file contents line by line, and construct a file diff, such that applying the line_diff_dict (using apply_diff function) on str1 in a reversed order, we get str2.

    :param str1: initial file contents as strings
    :param str2: goal file contents as strings
    :param cleaned: whether to clean the new empty line diff or not
    :return: (delete list, add list, line_diff_dict)
    the line_diff_dict is in format: {"line_number": ("type": xxx, "original": xxx, "modified": xxx)}
    """
    lines1 = [d.rstrip() for d in str1.splitlines()]
    lines2 = [d.rstrip() for d in str2.splitlines()]

    diff = list(difflib.ndiff(lines1, lines2))

    delete_list = []
    add_list = []

    line_num1 = 0
    line_num2 = 0

    for d in diff:
        code = d[0]
        text = d[2:]

        if code == " ":  # unchanged
            line_num1 += 1
            line_num2 += 1
        elif code == "-":  # deletion
            line_num1 += 1
            delete_list.append((line_num1, text))
        elif code == "+":  # addition
            line_num2 += 1
            add_list.append((line_num2, text))

    # Post-process for modifications:
    # If a deletion and an addition at the same line, treat it as "Modify".
    # Build a line_diff_dict without mutating while iterating.
    line_diff_dict = {}
    delete_ptr = 0
    add_ptr = 0
    while delete_ptr < len(delete_list) or add_ptr < len(add_list):
        # Add delta to compute the delta of add and delete operations before
        delete_add_delta = add_ptr - delete_ptr
        if delete_ptr >= len(delete_list):
            tp = "Add"
            original_text = ""
            line_no, modified_text = add_list[add_ptr]
            line_no -= delete_add_delta
            add_ptr += 1
        elif add_ptr >= len(add_list):
            tp = "Delete"
            modified_text = ""
            line_no, original_text = delete_list[delete_ptr]
            delete_ptr += 1
        else:
            delete_line_no = delete_list[delete_ptr][0]
            add_line_no = add_list[add_ptr][0]
            if delete_line_no + delete_add_delta < add_line_no:
                tp = "Delete"
                modified_text = ""
                line_no, original_text = delete_list[delete_ptr]
                delete_ptr += 1
            elif delete_line_no + delete_add_delta > add_line_no:
                tp = "Add"
                original_text = ""
                line_no, modified_text = add_list[add_ptr]
                line_no -= delete_add_delta
                add_ptr += 1
            else:
                tp = "Modify"
                line_no, original_text = delete_list[delete_ptr]
                _, modified_text = add_list[add_ptr]
                delete_ptr += 1
                add_ptr += 1
        while line_no in line_diff_dict:
            line_no = f"{line_no} "
        line_diff_dict[str(line_no)] = {
            "type": tp,
            "original": original_text,
            "modified": modified_text
        }

    # Remove new line diff (optional, but should be used in evaluation)
    if cleaned:
        keys_to_delete = []
        for k, v in line_diff_dict.items():
            if v["original"].strip() == v["modified"].strip() == "":
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del line_diff_dict[k]

    line_diff_dict = dict(sorted(line_diff_dict.items(), key=lambda item: (int(item[0]), item[1]["type"])))
    return delete_list, add_list, line_diff_dict


def apply_diff(original_code, diff, with_delta=False):
    """
    Apply diff on original code to get a modified code.

    :param original_code: original code string to apply diff
    :param diff: the diff in format {"line_number": ("type": xxx, "original": xxx, "modified": xxx)}
    :param with_delta: boolean variable
        when it is False, we apply diff_dict reversely as-is to the original code, normally used when building composition from a set of single bugs;
            For example, with the following diff
            {
                "24": {
                    "type": "Add",
                    "original": "",
                    "modified": "a=1"
                },
                "24 ": {
                    "type": "Add",
                    "original": "",
                    "modified": "a=2"
                },
            }
            when applying reversely, we will add to line 24 two times, first a=2 and then a=1.
            Note that we DO assume the input diff is sorted in this format.
        when it is True, we apply diff_dict reversely with some delta, normally used when constructing fixes to the buggy code, especially when there are multiple adds and deletes.
            For example, with the following diff
            {
                "24": {
                    "type": "Add",
                    "original": "",
                    "modified": "a=1"
                },
                "25": {
                    "type": "Add",
                    "original": "",
                    "modified": "a=2"
                },
            }
            when applying reversely, similarly, we have to add to line 24 (pushing the original content to line 25) two times, first a=2 and then a=1.
            Note that we DO NOT assume the input diff is sorted in this format.
    :return: modified code string
    """
    diffs = []
    for line_no, v in diff.items():
        diffs.append((int(line_no), v["type"], v["original"], v["modified"]))
    if with_delta:
        # We do not assume diffs are sorted when with_delta=True
        diffs.sort(key=lambda x: (x[0], x[1]))

    # Apply modifications
    code_lines = original_code.splitlines()
    delete_add_delta = 0
    for _, tp, orig, mod in diffs:
        if tp == "Add":
            delete_add_delta += 1
        elif tp == "Delete":  # Deletion
            delete_add_delta -= 1

    # Process from bottom to top to avoid messing up indices, put Delete first than Add if line number is the same
    for line_no, tp, orig, mod in diffs[::-1]:
        idx = line_no - 1  # 1-based to 0-based

        if tp == "Modify":
            if 0 <= idx < len(code_lines):
                code_lines[idx] = mod

        elif tp == "Add":
            delete_add_delta -= 1
            if with_delta and 0 <= idx - delete_add_delta <= len(code_lines):
                code_lines.insert(idx - delete_add_delta, mod)
            elif 0 <= idx <= len(code_lines):
                code_lines.insert(idx, mod)

        elif tp == "Delete":
            delete_add_delta += 1
            if 0 <= idx < len(code_lines):
                del code_lines[idx]

    mod_code = "\n".join([l.rstrip() for l in code_lines])
    return mod_code


def parse_diff_to_blocks(diffs, ordered=True):
    """
    Parse diffs into edit blocks, merging consecutive edits into one block of edits.

    :param diffs: the diff in format {"line_number": ("type": xxx, "original": xxx, "modified": xxx)}
    :param ordered: the diff is sorted by line number or not
    :return: a list of block diffs in order, each element in format {
        "block_start": start line number,
        "block_end": end line number,
        "diff": the block diff,
        "block_id": block numbering
    }
    """
    if not ordered:
        orig_diffs = list(sorted(diffs.items(), key=lambda x: (int(x[0]), x[1]["type"])))
    else:
        orig_diffs = list(diffs.items())

    set_del_mod = {"Delete", "Modify"}
    set_add = {"Add"}
    current_block = []
    all_blocks = []
    consecutive = True
    prev_tp, prev_line_no = None, None
    for line_no_str, edit in orig_diffs[::-1]:
        line_no = int(line_no_str)
        tp = edit["type"]

        # starting from the second last, check if consecutive edits
        if prev_tp is not None:
            if tp in set_del_mod and line_no == prev_line_no - 1:
                consecutive = True
            elif tp in set_add and line_no == prev_line_no:
                consecutive = True
            else:
                consecutive = False

        if consecutive:
            current_block.insert(0, (line_no_str, edit))
        else:
            all_blocks.insert(0, {
                "block_start": int(current_block[0][0]),
                "block_end": int(current_block[-1][0]),
                "diff": dict(current_block)
            })
            current_block = [(line_no_str, edit)]
        prev_tp = tp
        prev_line_no = line_no

    # add the remaining block if non-empty
    if len(current_block):
        all_blocks.insert(0, {
            "block_start": int(current_block[0][0]),
            "block_end": int(current_block[-1][0]),
            "diff": dict(current_block)
        })

    # numbering blocks
    for i, block in enumerate(all_blocks):
        block["block_id"] = i

    return all_blocks


def expand_blocks_to_diff(blocks, ordered=True):
    """
    Expand code blocks into diffs.

    :param blocks: a list of block diffs in reverse order, each element in format {
        "block_start": start line number,
        "block_end": end line number,
        "diff": the block diff,
        "block_id": block numbering
    }
    :param ordered: the blocks are sorted by block number or not
    :return: a diff in format {"line_number": ("type": xxx, "original": xxx, "modified": xxx)}
    """
    if not ordered:
        ordered_blocks = sorted(blocks, key=lambda x: x["block_id"])
    else:
        ordered_blocks = blocks

    merged_diff = {}
    for block in ordered_blocks:
        merged_diff |= block["diff"]

    return merged_diff


def verify_block_single_diff(diff, block_count=-1, stride=0):
    """
    Verify each block has only a single line diff, used for checking diff after bug composition.
    Optionally, check number of blocks when specified a block_count.
    Optionally, check if the strides of the blocks are greater than specified value.

    :param diff: the diff in format {"line_number": ("type": xxx, "original": xxx, "modified": xxx)}
    :param block_count: expected number of blocks (e.g., bugs) in the code
    :param stride: expected stride of the blocks
    :return: a boolean value of the check, and a str with failed reason.
    """
    blocks = parse_diff_to_blocks(diff)
    if block_count >= 0 and len(blocks) != block_count:
        return False, f"Blocks count {len(blocks)} not equals to expected number {block_count}."
    block_end = -np.inf
    for i, b in enumerate(blocks):
        if b["block_start"] - block_end < stride:
            return False, (f"Block {i} starts at line {b['block_start']}, while previous one ends at line {block_end}, "
                           f"smaller than expected stride {stride}.")
        if len(b["diff"]) > 1:
            return False, f"Block {i} of {len(blocks)} has {len(b['diff'])} line diffs, more than expect 1."
    return True, ""


def load_symbolic_cache(dataset_name):
    """
    Load the per-dataset symbolic acceptance cache.
    Schema:
        {
          "<task_id>": { "<buggy_line>": ["accepted_fixed_line", ...] }
        }
    """
    cache_path = Path("data") / dataset_name / "symbolic_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_symbolic_cache(dataset_name, cache):
    """
    Persist the symbolic acceptance cache back to data/{dataset}/symbolic_cache.json
    """
    cache_dir = Path("data") / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "symbolic_cache.json"
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def symbolic_judge(task_id, sol_diff, gt_diff, cache):
    """
    Determine whether the solution's diff is symbolically acceptable.

    - gt_diff maps GT -> buggy; we expect solutions to reverse those changes.
      So expected pairs are (buggy_line, gt_line) from gt_diff values.
    - sol_diff maps buggy -> fixed; we compare pairs (buggy_line, fixed_line).
    - cache adds accepted alternatives: cache[task_id][buggy_line] contains a list of
      previously verified fixed lines.

    Returns: (symbolically_ok: bool, unmatched_pairs: set[(buggy_line, fixed_line)])
    """
    # Build expected reverse pairs from gt_diff
    expected_pairs = set()
    if gt_diff:
        for _, v in gt_diff.items():
            # GT: original -> buggy; reverse is (buggy, original)
            expected_pairs.add((v.get("modified", "").strip(), v.get("original", "").strip()))

    # Build solution pairs from sol_diff (buggy -> fixed)
    sol_pairs = set()
    if sol_diff:
        for _, v in sol_diff.items():
            sol_pairs.add((v.get("original", "").strip(), v.get("modified", "").strip()))

    accepted = set(expected_pairs)
    task_cache = cache.get(task_id, {}) if isinstance(cache, dict) else {}
    for buggy_line, accepted_list in task_cache.items():
        for fixed_line in accepted_list:
            accepted.add((str(buggy_line).strip(), str(fixed_line).strip()))

    unmatched = {p for p in sol_pairs if p not in accepted}
    return (len(unmatched) == 0), unmatched


def align_test_to_solution(solution_code, test_code):
    actual_name_match = re.search(r'def\s+(\w+)\s*\(', solution_code)
    if not actual_name_match:
        print("Warning: Could not find a function definition in the provided solution code.")
        return test_code  # Return the original test code

    actual_name = actual_name_match.group(1)
    stale_name_match = re.search(r'from\s+solution\s+import\s+(\w+)', test_code)
    if not stale_name_match:
        print("Warning: Could not find 'from solution import ...' pattern in the test code. No changes made.")
        return test_code  # Return the original test code

    stale_name = stale_name_match.group(1)

    if actual_name != stale_name:
        print(f"Function name mismatch found: Updating test code to use '{actual_name}' instead of '{stale_name}'.")
        return test_code.replace(stale_name, actual_name)
    else:
        print("Function names already match. No changes made.")
        return test_code


def verify_unit_test(dataset, verify_file, gt_file=None, timeout_per_task=20, timeout=1800):
    """
    This verify function is to evaluate the solution based on the dataset API

    Currently, it only supports bigcodebench.

    ------------------------- bigcodebench -------------------------
    For bigcodebench, it uses the bigcodebench.evaluate API to evaluate the solution.
    The input file should be in the JSONL format of:
    {
        "task_id": "123xxx",
        "solution": "The solution to the task",
    }
    It returns a list of task_ids that failed and a list of task_ids that passed.
    """
    if dataset == "bigcodebench":
        if gt_file is not None:
            assert Path(verify_file).parent == Path(gt_file).parent
            os.environ["BIGCODEBENCH_OVERRIDE_PATH"] = Path(gt_file).name
            selected_ids = None
        else:
            os.environ.pop("BIGCODEBENCH_OVERRIDE_PATH", None)
            selected_ids = ",".join([json.loads(s)["task_id"] for s in open(verify_file).readlines()])

        os.environ["BIGCODEBENCH_TIMEOUT_PER_TASK"] = str(timeout_per_task)
        workdir = Path(verify_file).parent
        base_name = Path(verify_file).with_suffix("").name
        candidates = [
            workdir / f"{base_name}_eval_results.json",  # normal case (.jsonl)
            workdir / f"{base_name}_pass_at_k.json",
        ]
        try:
            candidates[0].unlink()
            print(f"Removed existing file: {candidates[0]}")
            candidates[1].unlink()
            print(f"Removed existing file: {candidates[1]}")
        except FileNotFoundError:
            print(f"New verifying files: {candidates[0]} and {candidates[1]}")

        try:
            if selected_ids is not None:
                result = subprocess.run(
                    [
                        "bigcodebench.evaluate",
                        "--execution", "local",
                        "--split", "instruct",
                        "--subset", "full",
                        "--samples", Path(verify_file).name,  # basename only
                        "--selective_evaluate", selected_ids,
                        "--no_gt",
                    ],
                    cwd=workdir,  # run here
                    check=True,
                    timeout=timeout,
                )
            else:
                result = subprocess.run(
                    [
                        "bigcodebench.evaluate",
                        "--execution", "local",
                        "--split", "instruct",
                        "--subset", "full",
                        "--samples", Path(verify_file).name,  # basename only
                        "--no_gt",
                    ],
                    cwd=workdir,  # run here
                    check=True,
                    timeout=timeout,
                )
        except subprocess.CalledProcessError as e:
            # This block runs if the command fails (returns non-zero exit code)
            print("Command failed with an error.")
            print(f"Return Code: {e.returncode}")
        except subprocess.TimeoutExpired as e:
            # This block runs if the command takes too long
            print("Command timed out!")
        except TypeError:
            print("Error: A command argument was not a string. Check your variables.")

        if candidates[0].exists():
            with open(candidates[0], "r") as f:
                data = json.load(f)

            eval_dict = data.get("eval", {})

            fail_ids, correct_ids = [], []
            for task_id, perfs in eval_dict.items():
                status = perfs[0].get("status", "fail")
                (fail_ids if status == "fail" else correct_ids).append(task_id)
            return fail_ids, correct_ids
        else:
            raise FileNotFoundError(f"Cannot locate evaluation results for {base_name}")

    # LiveCodeBench
    elif dataset == "livecodebench":
        """Pass/fail with per-variant expansion using sidecar mapping (rich output)."""
        workdir = Path("/home/zhuwangz/miaosenchai/GenerationDataset/LiveCodeBench")

        eval_output_filename = verify_file.replace(".json", "_output_eval.json")

        command = [
            "python",
            "-m",
            "lcb_runner.runner.custom_evaluator",
            "--custom_output_file",
            str(Path.cwd() / verify_file),
        ]

        result = subprocess.run(
            command,
            cwd=workdir,
            check=True,
        )

        if not Path(eval_output_filename).exists():
            raise FileNotFoundError(f"Rich evaluation output file not found at {eval_output_filename}")

        with open(eval_output_filename, "r") as f:
            eval_data = json.load(f)

        # Load the sidecar map to reconstruct full ids per variant
        map_file = verify_file.replace(".json", "_map.json")
        full_id_map = {}
        if Path(map_file).exists():
            with open(map_file, "r") as f:
                full_id_map = json.load(f)

        # Also load verify input to get question order
        with open(verify_file, "r") as f:
            verify_input = json.load(f)
        ordered_qids = [d.get("question_id") for d in verify_input]

        fail_ids, correct_ids = [], []

        # Rich eval format contains per-index per-candidate graded results in eval_data[1]
        if not (isinstance(eval_data, list) and len(eval_data) > 1 and isinstance(eval_data[1], dict)):
            raise ValueError("Unexpected LiveCodeBench rich output format; missing per-index results")
        per_index = eval_data[1]

        for idx, qid in enumerate(ordered_qids):
            key = str(idx)
            if key not in per_index:
                continue
            candidate_results = per_index[key]
            if not isinstance(candidate_results, list) or len(candidate_results) == 0:
                continue
            # Each element in candidate_results corresponds to one candidate's per-test outcomes
            # Determine pass per candidate as (all True over tests), treating numeric negatives as failure
            full_ids = full_id_map.get(qid, [qid] * len(candidate_results))
            # Evaluate candidates present in results
            num_to_map = min(len(candidate_results), len(full_ids))
            for j in range(num_to_map):
                tests = candidate_results[j]
                passed = False
                if isinstance(tests, list) and len(tests) > 0:
                    # If entries are booleans, pass if all True
                    if all(isinstance(e, bool) for e in tests):
                        passed = all(tests)
                    else:
                        # If any numeric error codes present (<0), treat as failure; otherwise coerce truthiness of booleans only
                        has_error = any(isinstance(e, (int, float)) and e < 0 for e in tests)
                        all_true_bools = all((e is True) for e in tests if isinstance(e, bool))
                        passed = (not has_error) and all_true_bools
                elif isinstance(tests, bool):
                    passed = tests
                else:
                    # Unknown shape -> conservative failure
                    passed = False

                (correct_ids if passed else fail_ids).append(full_ids[j])

            # Any remaining submitted candidates without a corresponding result entry
            # are treated as failures to keep counts consistent
            if len(full_ids) > num_to_map:
                for j in range(num_to_map, len(full_ids)):
                    fail_ids.append(full_ids[j])

        return fail_ids, correct_ids

    elif dataset == "kodcodebench":
        fail_ids, correct_ids = [], []
        with open(verify_file, "r") as f:
            data = json.load(f)

        for entry in data:
            task_id = entry.get("task_id")
            solution = entry.get("solution")[0]
            test = entry.get("test")
            test = align_test_to_solution(solution, test)

            with open('solution.py', 'w') as f:
                f.write(solution)
            with open('test_solution.py', 'w') as f:
                f.write(test)

            exit_code = pytest.main([
                '-q',  # Use the quiet flag instead of -v
                '--no-header',  # Suppress the header
                '--no-summary',  # Suppress the final summary
                '--disable-warnings',  # Suppress any warnings
                'test_solution.py'
            ])

            if exit_code == 0:
                print("Correct ID ------------")
                print(task_id)
                print("------------------------")
                correct_ids.append(task_id)
            else:
                print("Incorrect ID ------------")
                print(task_id)
                print("------------------------")
                fail_ids.append(task_id)

            os.remove('solution.py')
            os.remove('test_solution.py')

        return fail_ids, correct_ids

    else:
        raise ValueError(f"Dataset '{dataset}' not supported")


def build_verify_unit_test(dataset_name, log_file_prefix, results, sol_field="solution"):
    if dataset_name == "livecodebench":
        verify_file = log_file_prefix + ".json"
        # Group by normalized question id so multiple variants are evaluated together
        grouped = {}
        # Sidecar map of base qid -> ordered full ids (to reconstruct results)
        qid_to_full_ids = {}
        for entry in results:
            code = entry.get(sol_field)
            if code is None:
                continue
            qid = str(entry["task_id"]).split("_")[0]
            grouped.setdefault(qid, [])
            grouped[qid].append(code)
            qid_to_full_ids.setdefault(qid, [])
            qid_to_full_ids[qid].append(entry["task_id"])  # preserve order
        data_to_write = []
        for qid, codes in grouped.items():
            entry = {
                "question_id": qid,
                "code_list": codes,
                # Provide per-candidate metadata to match lengths expected by the evaluator
                "metadata": [{} for _ in codes],
            }
            data_to_write.append(entry)
        if data_to_write:
            with open(verify_file, "w") as f:
                json.dump(data_to_write, f, indent=4)
            # Write sidecar map alongside verify file
            with open(verify_file.replace(".json", "_map.json"), "w") as f:
                json.dump(qid_to_full_ids, f, indent=2)
            return verify_file
        else:
            print("No submissions to evaluate.")
            return None
    elif dataset_name == "bigcodebench":  # bigcodebench
        verify_file = log_file_prefix + ".jsonl"
        with open(verify_file, "w") as f:
            wrote_any = False
            for entry in results:
                if entry[sol_field] is not None:
                    json.dump({
                        "task_id": entry["task_id"],
                        "solution": entry[sol_field]
                    }, f)
                    f.write("\n")
                    wrote_any = True
        if wrote_any:
            return verify_file
        else:
            print("No submissions to evaluate.")
            return None
    else:
        raise ValueError("Unexpected dataset name.")


def save_formatted_gt(dataset, log_file_prefix, data):
    """
    Saves formatted ground truth data to file
    :param dataset:
    :param log_file_prefix:
    :param data:
    :return:
    """
    if dataset == "bigcodebench":
        original_gt_data = json.load(open("data/bigcodebench/bigcodebench-full-data.json"))
        gt_data = []
        for d in data:
            task_id = d["task_id"]
            while task_id not in original_gt_data:
                task_id = task_id.rsplit("_", 1)[0]
            selected = copy.deepcopy(original_gt_data[task_id])
            selected["task_id"] = d["task_id"]
            gt_data.append(selected)
        out_path = f"{log_file_prefix}.jsonl"
        with open(out_path, "w") as f:
            f.write("\n".join([json.dumps(d) for d in gt_data]))
            # json.dump(gt_data, f, indent=2)
    elif dataset == "livecodebench":
        # LiveCodeBench evaluator uses its own internal GT; no formatted GT needed.
        gt_data = []
        out_path = None
    else:
        raise ValueError(f"Dataset '{dataset}' not supported")
    return gt_data, out_path


def mark_editable_lines(dataset, data):
    if dataset == "bigcodebench":
        if len(data) > 0:
            assert "task_prompt" in data[0]
            for d in data:
                code_matches = SIMPLE_CODE_BLOCK_REGEX.findall(d["task_prompt"])
                if code_matches:
                    d["frozen_lines"] = len(code_matches[-1].strip().splitlines())
                else:
                    d["frozen_lines"] = 0
                code_lines = d["gt_solution"].splitlines()
                code_length = len(code_lines)
                d["gt_length"] = code_length
                d["editable_lines"] = []
                for i, l in enumerate(code_lines):
                    if i >= d["frozen_lines"] and l.strip() != "":
                        d["editable_lines"].append((i + 1, l))
    elif dataset == "livecodebench":
        for d in data:
            code_lines = d["gt_solution"].splitlines()
            code_length = len(code_lines)
            d["frozen_lines"] = 0
            d["gt_length"] = code_length
            d["editable_lines"] = []
            for i, l in enumerate(code_lines):
                if i >= d["frozen_lines"] and l.strip() != "":
                    d["editable_lines"].append((i + 1, l))
    else:
        raise ValueError("Unexpected dataset name.")


def verify_single_solution(dataset_name, item, solution, verify_prefix):
    """
    Write a minimal eval file for a single task and call verify().
    Returns True if the solution passes unit tests, else False.
    """
    verify_dir = Path("log") / dataset_name
    verify_dir.mkdir(parents=True, exist_ok=True)

    try:
        if dataset_name == "bigcodebench":
            vf = str(verify_dir / f"{verify_prefix}_single_correct.jsonl")
            with open(vf, "w") as f:
                json.dump({"task_id": item["task_id"], "solution": solution}, f)
                f.write("\n")
            fail_ids, correct_ids = verify_unit_test(dataset_name, vf)
            return item["task_id"] in correct_ids

        elif dataset_name == "livecodebench":
            vf = str(verify_dir / f"{verify_prefix}_single_correct.json")
            with open(vf, "w") as f:
                json.dump([{"question_id": item["task_id"], "code_list": [solution]}], f, indent=2)
            fail_ids, correct_ids = verify_unit_test(dataset_name, vf)
            return item["task_id"] in correct_ids

        elif dataset_name == "kodcodebench":
            vf = str(verify_dir / f"{verify_prefix}_single_correct.json")
            with open(vf, "w") as f:
                json.dump([{"task_id": item["task_id"], "solution": [solution],
                            "test": item.get("test") or item["original_data"].get("test")}], f, indent=2)
            fail_ids, correct_ids = verify_unit_test(dataset_name, vf)
            return item["task_id"] in correct_ids

        else:
            print(f"[verify] Unsupported dataset: {dataset_name}")
            return False
    except Exception as e:
        print(f"[verify_single_solution] Error for {item.get('task_id')}: {e}")
        return False


if __name__ == "__main__":
    str1 = "import itertools\nfrom random import shuffle\n\ndef task_func(numbers=list(range(1, 3))):\n    cumulative_diff = 0\n\n    for permutation_tuple in permutation_stream:\n        perm_list = list(permutation_tuple)\n        shuffle(perm_list)\n        diff_accumulator = 0\n        index = 0\n\n        while index < len(perm_list) - 1:\n            first_val = perm_list[index]\n            second_val = perm_list[index + 1]\n            if first_val >= second_val:\n                diff_accumulator += first_val - second_val\n            else:\n                diff_accumulator += second_val - first_val\n            index += 1\n\n        cumulative_diff += diff_accumulator\n        processed_count += 1\n\n    return cumulative_diff / processed_count"
    str2 = "import itertools\nfrom random import shuffle\n\ndef task_func(numbers=list(range(1, 3))):\n    permutation_stream = itertools.permutations(numbers)\n    cumulative_diff = 0\n    processed_count = 0\n\n    for permutation_tuple in permutation_stream:\n        perm_list = list(permutation_tuple)\n        shuffle(perm_list)\n        diff_accumulator = 0\n        index = 1\n\n        while index < len(perm_list) - 1:\n            first_val = perm_list[index]\n            second_val = perm_list[index + 1]\n            if first_val >= second_val:\n                diff_accumulator += first_val - second_val\n            else:\n                diff_accumulator += second_val - first_val\n            index += 1\n\n        cumulative_diff += diff_accumulator\n        processed_count += 2\n        a=1\n        a=2\n\n    return cumulative_diff / processed_count"
    delete, add, json_diff = file_diff(str1, str2)
    print(json.dumps(json_diff, indent=2))
    print(apply_diff(str1, json_diff, True) == str2)
    print(str2)
    print(apply_diff(str1, json_diff, True))
