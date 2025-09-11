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
        workdir = Path("/home/ec2-user/DebugBench/Livecode-m")

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
