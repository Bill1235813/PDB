import os
import json
import subprocess
from pathlib import Path
import ast
import re
import textwrap
import pytest
import difflib


def file_diff(str1, str2):
    """
    Compare two file contents line by line.
    Input: str1, str2 (file contents as strings)
    Output: (delete, add, line_diff_dict)
    line_diff_dict in format: {line_number: ("type": xxx, "original": xxx, "modified": xxx)}
    types are Add [insert str to line xxx], Delete [delete line and move up the next] and Modify.
    """
    lines1 = [d.strip() for d in str1.splitlines()]
    lines2 = [d.strip() for d in str2.splitlines()]

    diff = list(difflib.ndiff(lines1, lines2))

    delete = {}
    add = {}
    line_diff_dict = {}

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
            delete[f"line {line_num1}"] = text
            line_diff_dict[f"{line_num1} Delete"] = {
                "type": "Delete",
                "original": text,
                "modified": ""
            }
        elif code == "+":  # addition
            line_num2 += 1
            add[f"line {line_num2}"] = text
            line_diff_dict[f"{line_num2} Add"] = {
                "type": "Add",
                "original": "",
                "modified": text
            }

    # Post-process for modifications:
    # If a deletion is immediately followed by an addition at (roughly) same line,
    # treat it as "Modify". Build a merged dict without mutating while iterating.
    items = list(line_diff_dict.items())
    merged = {}
    i = 0
    while i < len(items):
        k1, v1 = items[i]
        if i + 1 < len(items):
            k2, v2 = items[i + 1]
            is_del_add = (v1["type"] == "Delete" and v2["type"] == "Add")
            n1 = int(k1.split()[0])
            n2 = int(k2.split()[0])
            # Only merge if they're the exact same line number
            if is_del_add and n1 == n2:
                merged[str(n1)] = {
                    "type": "Modify",
                    "original": v1["original"],
                    "modified": v2["modified"]
                }
                i += 2
                continue
        merged[k1.split()[0]] = v1
        i += 1

    line_diff_dict = merged

    return delete, add, line_diff_dict


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


def verify(dataset, verify_file):
    if dataset == "bigcodebench":
        workdir = Path(verify_file).parent
        selected_ids = ",".join([json.loads(s)["task_id"] for s in open(verify_file).readlines()])
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
            capture_output=True,
            text=True
        )
        print(result.stderr)

        base_name = Path(verify_file).with_suffix("").name
        candidates = [
            workdir / f"{base_name}_eval_results.json",  # normal case (.jsonl)
            workdir / f"{base_name}.json",
        ]

        for p in candidates:
            if p.exists():
                eval_path = p
                break
            else:
                raise FileNotFoundError(f"Cannot locate evaluation results for {base_name}")

        with open(eval_path, "r") as f:
            data = json.load(f)

        eval_dict = data.get("eval", {})

        fail_ids, correct_ids = [], []
        for task_id, perfs in eval_dict.items():
            status = perfs[0].get("status", "fail")
            (fail_ids if status == "fail" else correct_ids).append(task_id)

        return fail_ids, correct_ids

    # LiveCodeBench

    elif dataset == "livecodebench":
        """Minimal pass/fail checker using LiveCodeBench's execution harness."""
        # This flow uses a JSON array and the custom_evaluator.py script
        workdir = Path("/home/zhuwangz/miaosenchai/GenerationDataset/LiveCodeBench")

        eval_output_filename = verify_file.replace(".json", "_output_eval_all.json")

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
            capture_output=True,
            text=True
        )

        print("Evaluation script stdout:")
        print(result.stdout)
        print("Evaluation script stderr:")
        print(result.stderr)

        if not Path(eval_output_filename).exists():
            raise FileNotFoundError(f"Evaluation output file not found at {eval_output_filename}")

        with open(eval_output_filename, "r") as f:
            eval_data = json.load(f)

        fail_ids, correct_ids = [], []

        for item in eval_data:
            task_id = item.get("question_id")
            graded_list = item.get("graded_list", [])

            # If the list is empty or any element is False the solution failed.
            is_correct = bool(graded_list) and all(graded_list)

            if is_correct:
                correct_ids.append(task_id)
            else:
                fail_ids.append(task_id)

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
            fail_ids, correct_ids = verify(dataset_name, vf)
            return item["task_id"] in correct_ids

        elif dataset_name == "livecodebench":
            vf = str(verify_dir / f"{verify_prefix}_single_correct.json")
            with open(vf, "w") as f:
                json.dump([{ "question_id": item["task_id"], "code_list": [solution] }], f, indent=2)
            fail_ids, correct_ids = verify(dataset_name, vf)
            return item["task_id"] in correct_ids

        elif dataset_name == "kodcodebench":
            vf = str(verify_dir / f"{verify_prefix}_single_correct.json")
            with open(vf, "w") as f:
                json.dump([{ "task_id": item["task_id"], "solution": [solution], "test": item.get("test") or item["original_data"].get("test") }], f, indent=2)
            fail_ids, correct_ids = verify(dataset_name, vf)
            return item["task_id"] in correct_ids

        else:
            print(f"[verify] Unsupported dataset: {dataset_name}")
            return False
    except Exception as e:
        print(f"[verify_single_solution] Error for {item.get('task_id')}: {e}")
        return False
