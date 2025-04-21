import os
import json
import subprocess


def verify(dataset, verify_file):
    if dataset == "bigcodebench":
        result = subprocess.run(
            ["bigcodebench.evaluate",
             "--execution", "local",
             "--split", "instruct",
             "--subset", "full",
             "--samples", verify_file,
             "--no_gt"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        with open(verify_file.replace(".jsonl, _eval_results.json"), "r") as f:
            log = json.load(f)
        log = log["eval"]
        fail_ids = []
        correct_ids = []
        for id, eval_result in log:
            if eval_result["status"] == "fail":
                fail_ids.append(id)
            else:
                correct_ids.append(id)
        return fail_ids, correct_ids
    else:
        raise ValueError("Dataset not supported")