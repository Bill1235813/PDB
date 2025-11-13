import json
import os
import copy

from utils import save_formatted_gt, build_verify_unit_test, verify_unit_test, file_diff, parse_diff_to_blocks, \
    verify_block_single_diff, expand_blocks_to_diff, rstrip_lines, apply_diff
from collections import defaultdict
from argparse import ArgumentParser


class Evaluator:
    def __init__(self, args, results):
        self.dataset = args.dataset_name
        self.output_dir = args.output_dir
        self.model_name = args.model_name
        self.stride = args.stride
        self.precision_tol = args.precision_tolerance
        assert self.precision_tol >= 1, "precision tolerance should be at least 1."
        self.results = results

        self.gt, self.buggy_code, self.pred, self.gt_diff, self.pred_diff = [], [], [], [], []
        final_results = []
        for s in self.results:
            if "gt_diff" in s:
                gt_diff = s["gt_diff"]
            else:
                _, _, gt_diff = file_diff(s["buggy_code"], s["gt_solution"])
            ver_gt_diff, err_str = verify_block_single_diff(gt_diff, block_count=s["bug_count"], stride=self.stride)
            if ver_gt_diff:
                self.gt_diff.append(gt_diff)
                final_results.append(s)
            else:
                continue
                # raise ValueError(err_str)
            # check the format of input file, evaluation is a separated process from debugging
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
                _, _, pred_diff = file_diff(s["buggy_code"], s["debug_results"]["solution"])
                self.pred_diff.append(pred_diff)

        self.results = final_results
        self.ids = [s["task_id"] for s in self.results]
        self.count = len(self.ids)
        self.metrics = {
            "Unit score": self.unit_score,
            "Symbolic block scores": self.symbolic_block_score,
            # "CodeBLEU": self.code_bleu
        }

        self.scores = {metrics: {} for metrics in self.metrics}

    def run_evaluation(self):
        if self.count == 0:
            print("No task to evaluate!")
        else:
            for name, metric in self.metrics.items():
                print(f"{name}:", metric(name))
            self.save_results()
        return self.scores
    # check the unit-test score by calling utils
    def unit_score(self, metric_name):
        print("Compute unit test score:")
        verify_file = build_verify_unit_test(self.dataset,
                                             self.output_dir + f"/{self.dataset}/{self.model_name}_unit_test_verify",
                                             [{"task_id": idx, "solution": pred} for idx, pred in
                                              zip(self.ids, self.pred)])
        _, formatted_gt = save_formatted_gt(self.dataset,
                                            self.output_dir + f"/{self.dataset}/{self.model_name}_unit_test_gt",
                                            self.results)
        fail_ids, correct_ids = verify_unit_test(self.dataset, verify_file, gt_file=formatted_gt, timeout=1800)
        fail_ids, correct_ids = set(fail_ids), set(correct_ids)
        for idx in self.ids:
            self.scores[metric_name][idx] = 0 if idx in fail_ids else 1
        return metric_name, len(correct_ids) / self.count

    def _is_equal(self, pred_edit, gt_edit):
        """Check if a predicted edit is the equal to a ground truth edit."""
        return (pred_edit['type'] == gt_edit['type'] and
                pred_edit['original'] == gt_edit['original'] and
                pred_edit['modified'] == gt_edit['modified'])

    def _line_set_match(self, pred_lines, gt_lines, top_btm_line):
        if gt_lines & pred_lines:
            return True
        elif not gt_lines and not pred_lines:
            return True
        # e.g., gt edits the top line but pred keeps it unchanged.
        elif not gt_lines and top_btm_line in pred_lines:
            return True
        # e.g., pred edits the top line but gt keeps it unchanged.
        elif not pred_lines and top_btm_line in gt_lines:
            return True
        else:
            return False

    def symbolic_block_score(self, metric_name):
        """
        Evaluates predicted diffs against ground truth diffs, using symbolic
        verification for non-trivial matches.
        """
        print("Compute precision and recall:")
        equ_test = []
        matched_blocks = defaultdict(dict)
        unmatched_gt = defaultdict(dict)
        unmatched_pred = defaultdict(dict)
        all_gt_blocks = {task_id: parse_diff_to_blocks(gt_diff) for task_id, gt_diff in zip(self.ids, self.gt_diff)}
        all_buggy = {task_id: buggy for task_id, buggy in zip(self.ids, self.buggy_code)}

        # Find matches and identify candidates for deep testing
        for task_id, gt_diff, pred_diff in zip(self.ids, self.gt_diff, self.pred_diff):
            remain_gt_diff = copy.deepcopy(gt_diff)
            remain_pred_diff = copy.deepcopy(pred_diff)
            buggy = all_buggy[task_id]

            # First pass, only find exact matches
            count_em = 0
            for line_no_str, v in list(pred_diff.items())[::-1]:
                line_no_gt = str(int(line_no_str))
                if line_no_gt in remain_gt_diff and self._is_equal(v, remain_gt_diff[line_no_gt]):
                    matched_blocks[task_id][f"{task_id}_em_{count_em}"] = {
                        "block_start": int(line_no_str),
                        "block_end": int(line_no_str),
                        "diff": {line_no_str: v},
                        "block_id": -1,
                        "success": True,
                        "gt_match_count": 1,
                        "tolerance": 0
                    }
                    count_em += 1
                    del remain_gt_diff[line_no_gt]
                    del remain_pred_diff[line_no_str]

            # Second pass, parse to blocks and find block matches
            remain_gt_blocks = parse_diff_to_blocks(remain_gt_diff)[::-1][::-1]
            remain_pred_blocks = parse_diff_to_blocks(remain_pred_diff)[::-1]
            buggy_lines = buggy.splitlines()
            for b_no, pred_block in enumerate(remain_pred_blocks):
                # 2.1 Check if a prediction block wraps a gt block of edits
                matched_gt_block = []
                for gt_block in remain_gt_blocks:
                    if pred_block["block_start"] <= gt_block["block_start"] <= pred_block["block_end"]:
                        matched_gt_block.append(gt_block)
                # 2.2 Check if a prediction block is "near" a gt block
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
                        pre_gt_lines = set(
                            [buggy_lines[gt_block["block_start"] - idx - 1] for idx in range(1, self.stride + 1) if
                             gt_block["block_start"] - idx > 0])
                        post_gt_lines = set(
                            [buggy_lines[gt_block["block_start"] + idx - 1] for idx in range(1, self.stride + 1) if
                             gt_block["block_start"] + idx <= len(buggy_lines)])
                        # check if line sets have at least one match
                        if (self._line_set_match(pre_pred_lines, pre_gt_lines, buggy_lines[0]) and
                                self._line_set_match(post_pred_lines, post_gt_lines, buggy_lines[-1])):
                            matched_gt_block.append(gt_block)
                            break
                # 2.3 Check if a prediction block is distant but identical to a gt block
                if not len(matched_gt_block) and len(pred_block["diff"]) == 1:
                    for gt_block in remain_gt_blocks:
                        if self._is_equal(list(pred_block["diff"].values())[0], list(gt_block["diff"].values())[0]):
                            matched_gt_block.append(gt_block)
                            break
                # There may be other possible cases ...

                # Update and construct test examples
                if len(matched_gt_block):
                    matched_gt_block_ids = [b["block_id"] for b in matched_gt_block]
                    start_block, end_block = min(matched_gt_block_ids), max(matched_gt_block_ids)
                    pred_block["block_id"] = start_block
                    test_block = (all_gt_blocks[task_id][:start_block] + [pred_block] +
                                  all_gt_blocks[task_id][end_block + 1:])
                    test_diff = expand_blocks_to_diff(test_block)
                    test_solution = apply_diff(buggy, test_diff)
                    equ_test.append({
                        "task_id": f"{task_id}_{b_no}",
                        "solution": test_solution
                    })
                    remain_gt_blocks = [b for b in remain_gt_blocks if b not in matched_gt_block]
                    # On tolerance: For a single-line bug, we consider a correct multi-line edit within
                    # (tolerance + 1) lines a "precise" edit, with a full precision score.
                    # This should be redefined for multi-line bugs.
                    matched_blocks[task_id][f"{task_id}_{b_no}"] = {
                        "pred_block": pred_block,
                        "gt_blocks": matched_gt_block,
                        "start_block": start_block,
                        "end_block": end_block,
                        "gt_match_count": len(matched_gt_block),
                        "tolerance": max(min(len(matched_gt_block) * (self.precision_tol - 1),
                                             len(pred_block["diff"]) - len(matched_gt_block)),
                                         0)
                    }
                else:
                    unmatched_pred[task_id] |= expand_blocks_to_diff([pred_block])

            unmatched_gt[task_id] = expand_blocks_to_diff(remain_gt_blocks)

        # Build and run verification for semantically complex cases
        if len(equ_test) > 0:
            print("Semantic equivalence check:")
            verify_file = build_verify_unit_test(self.dataset,
                                                 self.output_dir + f"/{self.dataset}/{self.model_name}_equ_test_verify",
                                                 equ_test,
                                                 sol_field="solution")
            _, temp_gt = save_formatted_gt(self.dataset,
                                           self.output_dir + f"/{self.dataset}/{self.model_name}_equ_test_gt",
                                           equ_test)
            fail_ids, correct_ids = verify_unit_test(self.dataset, verify_file, gt_file=temp_gt, timeout=1800)

            # Update correctness based on verification results
            redun_test = []
            for correct_id in correct_ids:
                try:
                    task_id = correct_id.rsplit('_', 1)[0]
                    matched_blocks[task_id][correct_id]["success"] = True
                    if correct_id == "BigCodeBench/88_1_0":
                        pass
                    if matched_blocks[task_id][correct_id]["tolerance"] > 0:
                        start_block = matched_blocks[task_id][correct_id]["start_block"]
                        end_block = matched_blocks[task_id][correct_id]["end_block"]
                        pred_block = copy.deepcopy(matched_blocks[task_id][correct_id]["pred_block"])
                        # For a correct block, check if any smaller (1 to tolerance line) edits in this block work
                        pred_diff = list(pred_block["diff"].items())
                        for tol in range(matched_blocks[task_id][correct_id]["tolerance"]):
                            for test_count in range(len(pred_diff) - tol):
                                pred_block["diff"] = dict(pred_diff[test_count:test_count + tol + 1])
                                pred_block["block_start"] = min([int(k) for k in pred_block["diff"].keys()])
                                pred_block["block_end"] = max([int(k) for k in pred_block["diff"].keys()])
                                test_block = (all_gt_blocks[task_id][:start_block] + [pred_block] +
                                              all_gt_blocks[task_id][end_block + 1:])
                                test_diff = expand_blocks_to_diff(test_block)
                                test_solution = apply_diff(all_buggy[task_id], test_diff)
                                redun_test.append({
                                    "task_id": f"{correct_id}_{tol}_{test_count}",
                                    "solution": test_solution
                                })
                except ValueError:
                    print(f"Warning: Could not parse task_id and line from '{correct_id}'")
            for fail_id in fail_ids:
                try:
                    task_id = fail_id.rsplit('_', 1)[0]
                    matched_blocks[task_id][fail_id]["success"] = False
                    matched_blocks[task_id][fail_id]["tolerance"] = 0
                    matched_blocks[task_id][fail_id]["gt_match_count"] = 0
                except ValueError:
                    print(f"Warning: Could not parse task_id and line from '{fail_id}'")

            if len(redun_test) > 0:
                print("Deep redundancy check:")
                verify_file = build_verify_unit_test(self.dataset,
                                                     self.output_dir + f"/{self.dataset}/{self.model_name}_redun_test_verify",
                                                     redun_test,
                                                     sol_field="solution")
                _, temp_gt = save_formatted_gt(self.dataset,
                                               self.output_dir + f"/{self.dataset}/{self.model_name}_redun_test_gt",
                                               redun_test)
                fail_ids, correct_ids = verify_unit_test(self.dataset, verify_file, gt_file=temp_gt, timeout=1800)
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

        # Calculate Precision, Recall, and F1
        total_precision, total_recall, total_f1 = 0, 0, 0

        for task_id, gt_diff, pred_diff in zip(self.ids, self.gt_diff, self.pred_diff):

            # Handle case where both prediction and ground truth are empty
            if not pred_diff and not gt_diff:
                precision, recall, f1 = 1.0, 1.0, 1.0
            else:
                actual_pos = len(gt_diff)
                tolerance = sum([match["tolerance"] for _, match in matched_blocks[task_id].items()])
                predicted_pos = len(pred_diff) - tolerance
                true_pos = sum([match["gt_match_count"] for _, match in matched_blocks[task_id].items()])

                precision = true_pos / predicted_pos if predicted_pos > 0 else 0.0
                recall = true_pos / actual_pos if actual_pos > 0 else 0.0

                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            self.scores[metric_name][task_id] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matched_blocks": matched_blocks[task_id],
                "unmatched_pred": unmatched_pred[task_id],
                "unmatched_gt": unmatched_gt[task_id]
            }

        return "Precision", total_precision / self.count, "Recall", total_recall / self.count, "F1", total_f1 / self.count

    # def code_bleu(self, lang="python"):
    #     self.scores["CodeBLEU"] = [calc_codebleu([gt], [pred], lang)["codebleu"] for gt, pred in
    #                                zip(self.gt, self.pred)]
    #     return np.mean(self.scores["CodeBLEU"]) if self.scores["CodeBLEU"] else 0.0

    def save_results(self):
        if self.results:
            with open(self.output_dir +
                      f"/{self.dataset}/{self.model_name}_round_{self.results[0]['round']}_scores.json", "w") as f:
                json.dump(self.scores, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="bigcodebench")
    parser.add_argument("--input_file", type=str, default="gpt-4o_on_buggy_code_0923-0106.json_correct.json")
    parser.add_argument("--output_dir", type=str, default="eval_measure_test")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--stride", type=int, default=2, help="Minimum stride between bug diffs")
    parser.add_argument("--precision_tolerance", type=int, default=3,
                        help="Maximum number of lines to fix one bug that considered as precise")
    parser.add_argument("--max_iter", type=int, default=2, help="Maximum number of add-bug iterations")

    args = parser.parse_args()
    args.model_name = args.model_name.split("/")[-1]

    eval_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(eval_dir, exist_ok=True)

    # Support both absolute paths and filenames under eval/{dataset_name}
    input_path = args.input_file
    if not os.path.isabs(input_path):
        input_path = os.path.join("eval", args.dataset_name, input_path)
    results = json.load(open(input_path))
    grouped = defaultdict(list)
    grouped_dict = defaultdict(dict)
    for item in results:
        grouped[item["round"]].append(item)
        task_id = item["task_id"]
        grouped_dict[item["round"]][task_id] = item

    # assert args.max_iter in grouped.keys()
    scores = None
    for i, group in grouped.items():
        if i > args.max_iter:
            break
        print("Evaluate round", i)
        if scores:
            for task_id, value in scores["Unit score"].items():
                if value == 1:
                    grouped_dict[i][task_id]["debug_results"] = grouped_dict[i - 1][task_id]["debug_results"]

        evaluator = Evaluator(args, group)
        scores = evaluator.run_evaluation()
    # print(evaluator.results[0])
