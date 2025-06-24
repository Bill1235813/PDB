import os
import json
import subprocess
from pathlib import Path
import ast
import re
import textwrap

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

# Restructure by Miaosen
def verify(dataset, verify_file):
    if dataset == "bigcodebench":
        workdir = Path(verify_file).parent
        result = subprocess.run(
            [
                "bigcodebench.evaluate",
                "--execution", "local",
                "--split", "instruct",
                "--subset", "full",
                "--samples", Path(verify_file).name,   # basename only
                "--no_gt",
            ],
            cwd=workdir,               # run here
            capture_output=True,
            text=True
        )
        print(result.stdout)

        base_name = Path(verify_file).with_suffix("").name
        candidates = [
            workdir / f"{base_name}_eval_results.json",   # normal case (.jsonl)
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

    else:
        raise ValueError(f"Dataset '{dataset}' not supported")
