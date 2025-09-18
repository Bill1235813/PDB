import json
import numpy as np
# from codebleu import calc_codebleu


class Evaluator:
    def __init__(self, results):
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
        self.unit_true = [s["debug_results"]["unit_true"] for s in self.results]

        self.count = len(self.ids)
        self.metrics = {
            "Symbolic debugging scores": self.symbolic_line_by_line,
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

    def symbolic_line_by_line(self):
        total_precision, total_recall, total_f1 = 0, 0, 0
        # Load cache for acceptable alternatives
        with open("data/livecodebench/symbolic_cache.json", "r") as f:
            cache = json.load(f)

        def norm(s):
            return (s or "").strip()

        for idx, (gt_diff, pred_diff) in enumerate(zip(self.gt_diff, self.pred_diff)):
            task_id = self.ids[idx]
            task_cache = cache.get(task_id, {})

            # Build GT set
            gt_set = set()
            for _, lc in gt_diff.items():
                gt_line = norm(lc["original"]) 
                buggy_line = norm(lc["modified"]) 
                gt_set.add(f"{gt_line} <-- {buggy_line}")

            # Build predicted fixes set (accept GT or alternatives from cache)
            pred_set = set()
            for _, lc in pred_diff.items():
                buggy_line = norm(lc["original"]) 
                fixed_line = norm(lc["modified"]) 
                pred_set.add(f"{fixed_line} <-- {buggy_line}")
                for alt in task_cache.get(buggy_line, []):
                    pred_set.add(f"{norm(alt)} <-- {buggy_line}")

            true_positives = len(gt_set & pred_set)

            precision = true_positives / len(pred_set) if pred_set else 0
            recall = true_positives / len(gt_set) if gt_set else 0

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

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
    results = json.load(open("output/bigcodebench/buggy_code.json"))
    evaluator = Evaluator(results)
    evaluator.run_evaluation()
    print(evaluator.results[0])
