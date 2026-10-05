import copy
import json
import os
import subprocess
from pathlib import Path

from dataset.base import DatasetHandler


class BigCodeBenchHandler(DatasetHandler):
    """
    Handler for the BigCodeBench dataset.

    BigCodeBench provides coding tasks with structured prompts, canonical solutions,
    and test suites. Evaluation runs via the `bigcodebench.evaluate` CLI tool.
    """

    GT_DATA_PATH = "dataset/bigcodebench/data/full_data.json"

    def preprocess(self, raw_data):
        """
        Transform raw BigCodeBench data into standardized format.

        Accepts either:
          - dict format (original BigCodeBench JSON keyed by task_id)
          - list format (already partially processed, each item has task_id field)
        """
        processed_data = []
        if isinstance(raw_data, dict):
            for task_id, example in raw_data.items():
                processed_data.append({
                    "task_id": task_id,
                    "gt_solution": example["code_prompt"] + "\n" + example["canonical_solution"],
                    "task_prompt": example["instruct_prompt"],
                    "test": example["test"] if "test" in example else None,
                })
        elif isinstance(raw_data, list):
            for example in raw_data:
                processed_data.append({
                    "task_id": example["task_id"],
                    "gt_solution": example["gt_solution"],
                    "task_prompt": example["task_prompt"],
                    "test": example["test"] if "test" in example else None,
                })
        else:
            raise NotImplementedError
        return processed_data

    def verify_unit_test(self, verify_file, gt_file=None, timeout_per_task=20, timeout=1800, n_workers=None):
        """
        Run unit tests using the bigcodebench.evaluate CLI.

        The input file should be in JSONL format:
            {"task_id": "123xxx", "solution": "The solution to the task"}

        NOTE: BigCodeBench provides its own evaluation harness as a CLI
        tool. We shell out to it rather than importing its internals, because the harness
        manages sandboxing, timeouts, and result collection internally.
        """
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
            workdir / f"{base_name}_eval_results.json",
            workdir / f"{base_name}_pass_at_k.json",
        ]
        try:
            candidates[0].unlink()
            print(f"Removed existing file: {candidates[0]}")
            candidates[1].unlink()
            print(f"Removed existing file: {candidates[1]}")
        except FileNotFoundError:
            print(f"New verifying files: {candidates[0]} and {candidates[1]}")

        eval_args = [
            "--execution", "local",
            "--split", "instruct",
            "--subset", "full",
            "--samples", Path(verify_file).name,
        ]
        if selected_ids is not None:
            eval_args += ["--selective_evaluate", selected_ids]
        eval_args.append("--no_gt")

        try:
            subprocess.run(
                self.venv_cmd("bigcodebench.evaluate", *eval_args),
                cwd=workdir,
                check=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as e:
            print("Command failed with an error.")
            print(f"Return Code: {e.returncode}")
        except subprocess.TimeoutExpired as e:
            print("Command timed out!")
        except TypeError:
            print("Error: A command argument was not a string. Check your variables.")

        if candidates[0].exists():
            with open(candidates[0], "r") as f:
                data = json.load(f)

            eval_dict = data.get("eval", {})

            fail_ids, correct_ids, fail_feedback = [], [], []
            for task_id, perfs in eval_dict.items():
                status = perfs[0].get("status", "fail")
                # NOTE: [design thought] Only "pass" counts as correct. Everything else
                # (fail, timeout, error) is a test failure — timeout in particular means
                # the code hung or was too slow, which is still a bug.
                if status == "pass":
                    correct_ids.append(task_id)
                else:
                    fail_ids.append(task_id)
                    fail_feedback.append(json.dumps(perfs[0].get("details", ""), indent=2))
            return fail_ids, correct_ids, fail_feedback
        else:
            raise FileNotFoundError(f"Cannot locate evaluation results for {base_name}")

    def build_verify_unit_test(self, log_file_prefix, results, sol_field="solution"):
        """
        Build a JSONL verification file for bigcodebench.evaluate.

        Each line: {"task_id": "...", "solution": "..."}
        """
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

    def save_formatted_gt(self, log_file_prefix, data):
        """
        Save ground truth in BigCodeBench's expected format.

        NOTE: BigCodeBench's evaluator needs the full original task
        metadata (not just the solution). We load the original GT file, find the matching
        entry by task_id, and write it out. For composed bug task_ids (e.g., "BigCodeBench/123_0"),
        we strip suffixes until we find the base task_id in the original data.
        """
        original_gt_data = json.load(open(self.GT_DATA_PATH))
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
        return out_path
