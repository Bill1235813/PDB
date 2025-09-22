import json
import numpy as np
from utils import save_formatted_gt, build_verify, verify


# from codebleu import calc_codebleu


class Evaluator:
    def __init__(self, args, results, formatted_gt=None):
        self.dataset = args.dataset
        self.output_dir = args.output_dir
        self.results = results
        self.ids = [s["task_id"] for s in self.results]
        # Support both top-level fields and nested under original_data
        self.gt = [
            (s["gt_solution"] if "gt_solution" in s else s["original_data"].get("gt_solution"))
            for s in self.results
        ]
        self.buggy_code = [
            (s["buggy_code"] if "buggy_code" in s else s["original_data"]["buggy_code"])
            for s in self.results
        ]
        self.pred = [s["debug_results"]["solution"] for s in self.results]
        self.gt_diff = [
            (s["diff"] if "diff" in s else s["original_data"]["diff"]) for s in self.results
        ]
        self.pred_diff = [s["debug_results"]["sol_diff"] for s in self.results]
        self.unit_true = [s["debug_results"]["unit_true"] if "unit_true" in s["debug_results"] else None for s in
                          self.results]

        if formatted_gt is None:
            _, self.formatted_gt = save_formatted_gt(self.dataset, self.output_dir + f"/{self.dataset}/unit_test_gt", results)

        self.count = len(self.ids)
        self.metrics = {
            "Symbolic debugging scores": self.symbolic_line_by_line,
            "Unit score": self.unit_score
            # "CodeBLEU": self.code_bleu
        }

        self.scores = {metrics: [] for metrics in self.metrics}

    def run_evaluation(self):
        if self.count == 0:
            print("No task to evaluate!")
        else:
            for name, metric in self.metrics.items():
                print(f"{name}:", metric())
            self.save_results()

    def unit_score(self):
        verify_file = build_verify(self.dataset, self.output_dir + f"/{self.dataset}/unit_test", self.results)
        fail_ids, correct_ids = verify(self.dataset, verify_file, gt_file=self.formatted_gt)
        fail_ids, correct_ids = set(fail_ids), set(correct_ids)
        self.scores["Unit score"] = [0 if idx in fail_ids else 1 for idx in self.ids]
        return "Unit score", len(fail_ids) / self.count

    def _norm(self, s):
        """Normalize string by stripping whitespace."""
        return (s or "").strip()

    def _is_inverse(self, pred_edit, gt_edit):
        """Check if a predicted edit is the inverse of a ground truth edit."""
        p_type = pred_edit['type']
        g_type = gt_edit['type']
        p_orig = self._norm(pred_edit['original'])
        p_mod = self._norm(pred_edit['modified'])
        g_orig = self._norm(gt_edit['original'])
        g_mod = self._norm(gt_edit['modified'])

        if p_type == 'Add' and g_type == 'Delete' and p_mod == g_orig:
            return True
        if p_type == 'Delete' and g_type == 'Add' and p_orig == g_mod:
            return True
        if p_type == 'Modify' and g_type == 'Modify' and p_orig == g_mod and p_mod == g_orig:
            return True
        return False

    def _apply_single_edit(self, source_code, line_num_str, edit_obj):
        """Applies a single line change from a diff to the source code."""
        # Split source into lines, keeping line endings
        lines = source_code.splitlines(True)
        # Diff line numbers are typically 1-based, adjust to 0-based index
        line_idx = int(line_num_str) - 1
        edit_type = edit_obj['type']

        modified_line_content = edit_obj['modified']
        # Ensure the new line has a newline character for consistency
        if not modified_line_content.endswith('\n'):
            modified_line_content += '\n'

        if edit_type == 'Add':
            # Insert the new line at the specified index
            lines.insert(line_idx, modified_line_content)
        elif edit_type == 'Delete':
            # Delete the line at the specified index
            if 0 <= line_idx < len(lines):
                del lines[line_idx]
        elif edit_type == 'Modify':
            # Replace the line at the specified index
            if 0 <= line_idx < len(lines):
                lines[line_idx] = modified_line_content

        return "".join(lines)

    def symbolic_line_by_line(self):
        """
        Evaluates predicted diffs against ground truth diffs, using symbolic
        verification for non-trivial matches.
        """
        total_precision, total_recall, total_f1 = 0, 0, 0

        # # Load cache for acceptable alternatives
        # with open("data/livecodebench/symbolic_cache.json", "r") as f:
        #     cache = json.load(f)

        deep_test = []
        total_pred_corrects = {}

        # 1. First Pass: Find inverse matches and identify candidates for deep testing
        for idx, (gt_code, gt_diff, pred_diff) in enumerate(zip(self.gt, self.gt_diff, self.pred_diff)):
            task_id = self.ids[idx]
            gt_diff_copy = gt_diff.copy()
            pred_corrects = set()

            for pred_line, pred_edit in pred_diff.items():
                # Case 1: line number in gt_diff and edits are opposite (inverse)
                if pred_line in gt_diff_copy and self._is_inverse(pred_edit, gt_diff_copy[pred_line]):
                    pred_corrects.add(pred_line)
                    del gt_diff_copy[pred_line]

                # Case 2: line number in gt_diff and edits are different
                elif pred_line in gt_diff_copy:
                    alternative_solution = self._apply_single_edit(gt_code, pred_line, pred_edit)
                    deep_test.append({
                        "task_id": f"{task_id}_{pred_line}",
                        "solution": alternative_solution
                    })
                    del gt_diff_copy[pred_line]

                # Case 3: line number not in gt_diff but an opposite edit exists elsewhere
                else:
                    # Find the closest inverse edit in the ground truth
                    closest_inverse_gt_line = None
                    min_distance = float('inf')

                    for gt_line, gt_edit in gt_diff_copy.items():
                        if self._is_inverse(pred_edit, gt_edit):
                            distance = abs(int(pred_line) - int(gt_line))
                            if distance < min_distance:
                                min_distance = distance
                                closest_inverse_gt_line = gt_line

                    if closest_inverse_gt_line:
                        gt_edit_to_apply = gt_diff_copy[closest_inverse_gt_line]

                        # First, apply the ground truth edit that is the inverse
                        intermediate_solution = self._apply_single_edit(gt_code, closest_inverse_gt_line,
                                                                        gt_edit_to_apply)
                        # Then, apply the predicted edit to that result
                        alternative_solution = self._apply_single_edit(intermediate_solution, pred_line, pred_edit)

                        deep_test.append({
                            "task_id": f"{task_id}_{pred_line}",
                            "solution": alternative_solution
                        })
                        del gt_diff_copy[closest_inverse_gt_line]

            total_pred_corrects[task_id] = pred_corrects

        # 2. Build and run verification for semantically complex cases
        verify_file = build_verify(self.dataset, self.output_dir + f"/{self.dataset}/deep_test", deep_test, sol_field="solution")
        _, temp_gt = save_formatted_gt(self.dataset, self.output_dir + f"/{self.dataset}/deep_test_gt", deep_test)
        fail_ids, correct_ids = verify(self.dataset, verify_file, gt_file=temp_gt)

        # 3. Update correctness based on verification results
        for correct_id in correct_ids:
            try:
                task_id, line = correct_id.rsplit('_', 1)
                if task_id in total_pred_corrects:
                    total_pred_corrects[task_id].add(line)
            except ValueError:
                print(f"Warning: Could not parse task_id and line from '{correct_id}'")

        # 4. Calculate Precision, Recall, and F1
        total_precision, total_recall, total_f1 = 0, 0, 0

        for idx, (gt_diff, pred_diff) in enumerate(zip(self.gt_diff, self.pred_diff)):
            task_id = self.ids[idx]

            # Handle case where both prediction and ground truth are empty (perfect match)
            if not pred_diff and not gt_diff:
                precision, recall, f1 = 1.0, 1.0, 1.0
            else:
                true_positives = len(total_pred_corrects.get(task_id, set()))
                predicted_positives = len(pred_diff)
                actual_positives = len(gt_diff)

                precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
                recall = true_positives / actual_positives if actual_positives > 0 else 0.0

                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            self.scores["Symbolic debugging scores"].append((precision, recall, f1))

        return "Precision", total_precision / self.count, "Recall", total_recall / self.count, "F1", total_f1 / self.count

    # def code_bleu(self, lang="python"):
    #     self.scores["CodeBLEU"] = [calc_codebleu([gt], [pred], lang)["codebleu"] for gt, pred in
    #                                zip(self.gt, self.pred)]
    #     return np.mean(self.scores["CodeBLEU"]) if self.scores["CodeBLEU"] else 0.0

    def save_results(self):
        for idx, result in enumerate(self.results):
            result["scores"] = {
                name: scores[idx] for name, scores in self.scores.items()
            }


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bigcodebench")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    results = json.load(open("output/bigcodebench/buggy_code_test.json"))
    evaluator = Evaluator(args, results)
    evaluator.run_evaluation()
    print(evaluator.results[0])
