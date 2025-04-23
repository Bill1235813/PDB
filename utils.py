import os
import json
import subprocess
from pathlib import Path

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
    else:
        raise ValueError("Dataset not supported")
