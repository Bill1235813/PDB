import json
import subprocess
from pathlib import Path

from dataset.base import DatasetHandler


class LiveCodeBenchHandler(DatasetHandler):
    """
    Handler for the LiveCodeBench dataset.

    LiveCodeBench contains competitive programming problems (LeetCode-style) with
    starter code, test cases, and reference solutions. Evaluation runs via the
    lcb_runner.runner.custom_evaluator module inside this dataset's uv venv
    (see dataset/livecodebench/README.md for install instructions).
    """

    def preprocess(self, raw_data):
        """
        Transform raw LiveCodeBench data into standardized format.

        Each entry must have question_id, question_content, and starter_code.
        The ground truth solution is taken from gt_solution, output_list, or code_list.
        """
        processed_data = []
        for example in raw_data:
            task_id = str(example["question_id"])
            task_prompt = (example["question_content"].strip() + "\n\nStart the code with\n```\n"
                           + example["starter_code"].strip() + "\n```")
            gt_solution = example.get("gt_solution")
            if gt_solution is None:
                out_list = example.get("output_list") or example.get("code_list")
                if isinstance(out_list, list) and len(out_list) > 0 and isinstance(out_list[0], str):
                    gt_solution = out_list[0]
            assert gt_solution is not None
            processed_item = {
                "task_id": str(task_id),
                "gt_solution": gt_solution,
                "task_prompt": task_prompt,
            }
            processed_data.append(processed_item)
        return processed_data

    def verify_unit_test(self, verify_file, gt_file=None, timeout_per_task=20, timeout=1800, n_workers=None):
        """
        Run unit tests using lcb_runner's custom evaluator.

        NOTE: LiveCodeBench's evaluator sorts problems by question_id
        internally, so the result indices don't match the input order. We use a sidecar
        _map.json file to reconstruct the mapping from sorted indices back to the original
        variant IDs.
        """
        workdir = self.install_dir

        eval_output_filename = verify_file.replace(".json", "_output_eval.json")

        subprocess.run(
            self.venv_cmd(
                "lcb_runner.runner.custom_evaluator",
                "--custom_output_file",
                str(Path.cwd() / verify_file),
            ),
            cwd=workdir,
            check=True,
            timeout=timeout,
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

        # NOTE: [pedagogical] The rich eval format from LCB has two elements: eval_data[0] is
        # summary info, eval_data[1] is a dict keyed by sorted index -> list of per-candidate
        # test outcomes. Each candidate's entry is a list of booleans (one per test case).
        if not (isinstance(eval_data, list) and len(eval_data) > 1 and isinstance(eval_data[1], dict)):
            raise ValueError("Unexpected LiveCodeBench rich output format; missing per-index results")
        per_index = eval_data[1]

        # LCB runner sorts the benchmark by question_id, so results are keyed by sorted index.
        sorted_qids = sorted(ordered_qids)
        qid_to_results = {}
        for idx, qid in enumerate(sorted_qids):
            key = str(idx)
            if key in per_index:
                qid_to_results[qid] = per_index[key]

        for qid in ordered_qids:
            if qid not in qid_to_results:
                continue
            candidate_results = qid_to_results[qid]
            if not isinstance(candidate_results, list) or len(candidate_results) == 0:
                continue
            full_ids = full_id_map.get(qid, [qid] * len(candidate_results))
            num_to_map = min(len(candidate_results), len(full_ids))
            for j in range(num_to_map):
                tests = candidate_results[j]
                passed = False
                if isinstance(tests, list) and len(tests) > 0:
                    if all(isinstance(e, bool) for e in tests):
                        passed = all(tests)
                    else:
                        has_error = any(isinstance(e, (int, float)) and e < 0 for e in tests)
                        all_true_bools = all((e is True) for e in tests if isinstance(e, bool))
                        passed = (not has_error) and all_true_bools
                elif isinstance(tests, bool):
                    passed = tests
                else:
                    passed = False

                (correct_ids if passed else fail_ids).append(full_ids[j])

            if len(full_ids) > num_to_map:
                for j in range(num_to_map, len(full_ids)):
                    fail_ids.append(full_ids[j])

        return fail_ids, correct_ids, ""

    def build_verify_unit_test(self, log_file_prefix, results, sol_field="solution"):
        """
        Build a JSON verification file for lcb_runner's custom evaluator.

        NOTE: LiveCodeBench groups multiple variants of the same problem
        by question_id. We also write a sidecar _map.json that maps each base question_id
        back to the full variant IDs, so verify_unit_test can reconstruct per-variant results.
        """
        verify_file = log_file_prefix + ".json"
        # Group by normalized question id so multiple variants are evaluated together
        grouped = {}
        qid_to_full_ids = {}
        for entry in results:
            code = entry.get(sol_field)
            if code is None:
                continue
            qid = str(entry["task_id"]).split("_", 1)[0]
            grouped.setdefault(qid, [])
            grouped[qid].append(code)
            qid_to_full_ids.setdefault(qid, [])
            qid_to_full_ids[qid].append(entry["task_id"])
        data_to_write = []
        for qid, codes in grouped.items():
            entry = {
                "question_id": qid,
                "code_list": codes,
                "metadata": [{} for _ in codes],
            }
            data_to_write.append(entry)
        if data_to_write:
            with open(verify_file, "w") as f:
                json.dump(data_to_write, f, indent=4)
            with open(verify_file.replace(".json", "_map.json"), "w") as f:
                json.dump(qid_to_full_ids, f, indent=2)
            return verify_file
        else:
            print("No submissions to evaluate.")
            return None

    def save_formatted_gt(self, log_file_prefix, data):
        """
        LiveCodeBench does not require a separate ground truth file for evaluation.
        """
        return None
