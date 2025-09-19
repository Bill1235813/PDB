import re
import numpy as np
import random
import tqdm
import json
import copy
from pathlib import Path
from utils import verify, file_diff, load_symbolic_cache, save_symbolic_cache, symbolic_judge, verify_single_solution
from examples import hard_bug_examples, bug_type_examples
from collections import defaultdict

BUG_GEN_TEMPLATE = (
    "Your task is to perform a deep analysis of a code snippet and intentionally introduce bugs with NO comments.\n"
    "You will be given two components:\n"
    "PART 1: A problem description outlining the intended functionality.\n"
    "PART 2: A solution to the problem.\n"
    "First, carefully read and understand both the problem description and the provided solution.\n"
    "Then, modify the solution by injecting realistic programming errors to simulate human mistakes.\n\n"
    "Instructions for modifying the code:\n"
    "- Delete all comments from the original code.\n"
    "- Do NOT add any new comments to the modified code.\n"
    "- Do NOT delete any lines (not including comments) from the original code, but you may modify them as needed to introduce bugs.\n"
    "- If there are existing bugs in the original code, IGNORE them and Do NOT modify those lines. Only add new bugs.\n"
    "- Do NOT change any variable names.\n\n"
    "You must inject exactly:\n"
    "- {bug_per_time} insidious bugs (subtle logical errors that are hard to spot)\n"
    "Important rules:\n"
    "- Do not include any comments inside the code.\n"
    "- Do not add any extra output or formatting.\n\n"
    "Only output:\n"
    "- A single code block with the final, modified version of the code containing all {bug_per_time} bugs and NO comments.\n"
    "- The difference between the original and modified (buggy) code in JSON format"
    "You need to check there is NO COMMENT inside your generation for the final step. Don't forget to include ```json ``` for parsing purpose\n"
    "\n"
    "---\n"
    "PART 1: Problem Description\n"
    "```text\n{task_prompt}\n```\n\n"
    "PART 2: Original Code\n"
    "```python\n{gt_solution}\n```\n\n"
    "---\n"
    "Output format (follow *exactly*):\\n"
    "```python\\n[Buggy code here]\\n```\\n"
    "Diff in JSON format (valid JSON, keys MUST be quoted strings):\n"
    "```json\n{{\n  \"<line_number>\": {{ \"original\": \"<orig line>\", \"modified\": \"<new line>\" }},\n  ...\n}}\n```\n"
    "Buggy Code Output (use the format above):\\n"
)

ONE_BUG_GEN_TEMPLATE = (
    "Your task is to perform a deep analysis of a code snippet and intentionally introduce one bug.\n"
    "You will be given two components:\n"
    "PART 1: A task description outlining the intended functionality \n"
    "PART 2: A solution to the task.\n"
    "First, carefully read and understand both the task description and the provided solution.\n"
    "Then, modify the solution by injecting realistic programming errors to simulate human mistakes.\n\n"
    "Instructions for modifying the code:\n"
    "- Delete all comments from the original code.\n"
    "- Do NOT add any new comments to the modified code.\n"
    "- Please ONLY change one line and make sure changing that line induce a HARD bug to the task.\n"
    "- Do NOT change any other variable names.\n\n"
    "Bug to add: {bug_type}\n\n"
    "Important rules:\n"
    "- Do NOT include any comments inside the code.\n"
    "- Do NOT change more than one line.\n"
    "Output format:\n"
    "- A single code block with the final, modified version of the buggy code with NO comments.\n"
    "- The difference between the original and modified buggy code in JSON format.\n"
    "You need to check there is NO COMMENT inside your generation for the final step. Don't forget to include ```json ``` for parsing purpose\n"
    "---\n"
    "PART 1: Problem Description\n"
    "```text\n{task_prompt}\n```\n\n"
    "PART 2: Solution Code\n"
    "```python\n{gt_solution}\n```\n\n"
    "---\n"
    "Output format (follow *exactly*):\\n"
    "```python\\n[Buggy code here]\\n```\\n"
    "Diff in JSON format (valid JSON, keys MUST be quoted strings):\n"
    "```json\n{{\n  \"<line_number>\": {{ \"original\": \"<orig line>\", \"modified\": \"<new line>\" }},\n  ...\n}}\n```\n"
    "Buggy Code Output (use the format above):\\n"
)

DEBUG_TEMPLATE = (
    "Analyze and debug the given Python implementation that contains errors \n"
    "Identify the bugs and fix only the bugs in the code. Do not generate a new solution. You don't need to provide any explanation.\n\n"
    "You must preserve the original code logic exactly. You are NOT allowed to:\n"
    "- Change variable names\n"
    "- Change the loop structure (e.g., for/while)\n"
    "- Remove or replace existing variables\n\n"
    "The input consists of two parts:\n"
    "PART 1: A problem description outlining the intended functionality.\n"
    "PART 2: A buggy implementation that needs to be fixed.\n"
    "Your response should include:\n"
    "- A self-contained, corrected Python implementation;\n"
    "- The difference between the original and modified code in JSON format\n"
    "- Don't forget to include ```json ``` for the parsing purpose"
    "\n"
    "---\n"
    "PART 1: Problem Description\n"
    "```text\n{task_prompt}\n```\n\n"
    "PART 2: Buggy Code\n"
    "```python\n{buggy_code}\n```\n\n"
    "---\n"
    "Output format (follow *exactly*):\n"
    "```python\n[Corrected code here]\n```\n"
    "Corrected Code Output (use the format above):\n"
)

CODE_BLOCK_REGEX = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
DIFF_REGEX = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


# def extract_json_diff(text):
#     match = DIFF_REGEX.search(text)
#
#     if not match:
#         return None
#
#     json_str = match.group(1).strip()
#     # remove placeholder lines with '...' and trailing commas before closing brace
#     json_str = re.sub(r",?\s*\.\.\.\s*", "", json_str)
#     try:
#         diff_dict = json.loads(json_str)
#         return diff_dict
#     except Exception:
#         return None


def diff_to_str(k, v):
    return f"{k}: {v['original']} --> {v['modified']}"


diff_str_pattern = re.compile(r"^(\d+): (.*) --> (.*)$")


def str_to_diff(s):
    m = diff_str_pattern.match(s)
    if not m:
        raise ValueError(f"String not in expected format: {s}")
    k, original, modified = m.groups()
    if original == "":
        type = "Add"
    elif modified == "":
        type = "Delete"
    else:
        type = "Modify"
    return {int(k): {"type": type, "original": original, "modified": modified}}


def compose_and_apply_diff(gt_solution, k, diff_strs):
    """
    Pick k non-adjacent diffs from diff_strs and apply them to gt_solution.

    Rules:
      - Replace: original != "" and modified != ""  -> replace line
      - Insert : original == "" and modified != ""  -> insert modified at line_no (before current line)
      - Delete : original != "" and modified == ""  -> delete line_no

    :param gt_solution: Original ground-truth code (string).
    :param k: Number of modifications.
    :param diff_strs: List of diff strings like "3: foo --> bar".
    :return: (list of chosen diff strings, buggy_code string) or (None, None) if not possible.
    """
    if len(diff_strs) < k:
        return None, None

    # Parse diff_strs into list[(line_no, original, modified, diff_str)]
    diffs = []
    for d in diff_strs:
        parsed = str_to_diff(d)
        for line_no, v in parsed.items():
            diffs.append((line_no, v["original"], v["modified"], d))

    diffs.sort(key=lambda x: x[0])  # sort by line number

    # Try random selections until one satisfies non-adjacency
    for _ in range(100):
        chosen = random.sample(diffs, k)
        chosen.sort(key=lambda x: x[0])
        line_nums = [c[0] for c in chosen]

        # enforce non-adjacency
        if all(abs(line_nums[i] - line_nums[i - 1]) > 1 for i in range(1, len(line_nums))):
            # Apply modifications
            code_lines = gt_solution.splitlines()

            # Process from bottom to top to avoid messing up indices
            for line_no, orig, mod, _ in sorted(chosen, key=lambda x: x[0], reverse=True):
                idx = line_no - 1  # 1-based to 0-based

                if orig and mod:  # Replace
                    if 0 <= idx < len(code_lines):
                        code_lines[idx] = mod

                elif not orig and mod:  # Insertion
                    if 0 <= idx <= len(code_lines):
                        code_lines.insert(idx, mod)

                elif orig and not mod:  # Deletion
                    if 0 <= idx < len(code_lines):
                        del code_lines[idx]

            buggy_code = "\n".join(code_lines)
            chosen_strs = [c[3] for c in chosen]
            return chosen_strs, buggy_code

    return None, None  # couldn’t find valid set


def bug_generate_correct(data, generator_add, genrator_cor, bug_per_time, log_file_prefix, dataset_name):
    """Run the bug-injection + self-repair loop for a single dataset.

    The *dataset_name* argument is forwarded to the verification helper so that
    the correct evaluation harness is invoked (e.g., BigCodeBench vs
    LiveCodeBench).
    """
    all_logs = []
    all_buggy_data = []
    for _ in range(bug_per_time):
        logs, buggy_data = bug_generate(
            data, generator_add, log_file_prefix, dataset_name
        )
        all_logs += logs
        all_buggy_data += buggy_data

    # Save the buggy code log
    with open(log_file_prefix + "_one_bug.json", "w") as f:
        json.dump(all_logs, f, indent=2)

    with open(log_file_prefix + "_one_bug_valid.json", "w") as f:
        json.dump(all_buggy_data, f, indent=2)

    all_buggy_data = bug_compose(all_buggy_data, 4, 20)

    with open(log_file_prefix + "_comp_bug.json", "w") as f:
        json.dump(all_buggy_data, f, indent=2)

    # results = bug_correct(
    #     all_buggy_data, genrator_cor, log_file_prefix, dataset_name
    # )
    # return results


def bug_compose(buggy_data, max_bugs, compose_per_example):
    """
    :param buggy_data: list of {
        "task_id": str,
        "gt_solution": str,
        "task_prompt": str,
        "bug_count": 1,
        "diff": str,
        "buggy_code": str,
        "test": str or None
    }
    :param max_bugs: Number of bugs to generate.
    :param compose_per_example: Number of composition to generate per example.
    :param log_file_prefix:
    :return: dict[bug_count] -> list of buggy examples
    """
    merged = defaultdict(lambda: {
        "task_id": None,
        "gt_solution": None,
        "task_prompt": None,
        "diff": [],
        "test": None
    })

    for entry in buggy_data:
        tid = entry["task_id"]
        if merged[tid]["task_id"] is None:  # first time seeing this task_id
            merged[tid]["task_id"] = tid
            merged[tid]["gt_solution"] = entry["gt_solution"]
            merged[tid]["task_prompt"] = entry["task_prompt"]
            merged[tid]["test"] = entry["test"]

        if entry.get("diff"):
            for line, diff in entry["diff"].items():
                diff_str = diff_to_str(line, diff)
                if not diff_str in merged[tid]["diff"]:  # deduplicate
                    merged[tid]["diff"].append(diff_str)

    all_bugs = {1: buggy_data}  # bug_count: bug_data
    # Apply k random non-adjacent modifications from diff_dict to gt_solution.
    for k in range(2, max_bugs + 1):
        composed_buggy_data = copy.deepcopy(merged)
        for tid, composed_buggy_item in composed_buggy_data.items():
            new_diff = []  # each item is a list of diff_strs
            new_buggy_code = []  # each item is a str
            gt_solution = composed_buggy_item["gt_solution"]
            diff_dict = composed_buggy_item["diff"]

            generated_codes = set()
            for j in range(compose_per_example):
                diff_comp, buggy_code = compose_and_apply_diff(gt_solution, k, diff_dict)

                # Check if composition succeeded and the result is new
                if buggy_code is not None and buggy_code not in generated_codes:
                    new_diff.append(diff_comp)
                    new_buggy_code.append(buggy_code)
                    generated_codes.add(buggy_code)

            composed_buggy_item["diff"] = new_diff
            composed_buggy_item["buggy_code"] = new_buggy_code

        # Expand the composed data back into a flat list of dictionaries
        k_bug_data = []
        for tid, item in composed_buggy_data.items():
            for diff_list, code_str in zip(item["diff"], item["buggy_code"]):
                final_diff_dict = {}
                for d_str in diff_list:
                    final_diff_dict.update(str_to_diff(d_str))

                entry = {
                    "task_id": item["task_id"],
                    "gt_solution": item["gt_solution"],
                    "task_prompt": item["task_prompt"],
                    "bug_count": k,
                    "diff": final_diff_dict,
                    "buggy_code": code_str,
                    "test": item["test"],
                }
                k_bug_data.append(entry)

        all_bugs[k] = k_bug_data

    return all_bugs


if __name__ == "__main__":
    all_buggy_data = json.load(open("log/bigcodebench/log_0909-1425_bug_iter1_one_bug.json"))
    new_buggy_data = []
    for log_entry in all_buggy_data:
        gt_solution = log_entry["original_data"]["gt_solution"]
        _, _, json_diff = file_diff(gt_solution, log_entry["buggy_code"])
        if json_diff is not None:
            log_entry["diff"] = json_diff
            new_buggy_data.append({
                "task_id": log_entry["task_id"],
                "gt_solution": gt_solution,
                "task_prompt": log_entry["original_data"]["task_prompt"],
                "bug_count": 1,
                "diff": json_diff,
                "buggy_code": log_entry["buggy_code"],
                "test": log_entry["test"] if "test" in log_entry else None
            })

    all_buggy_data = bug_compose(new_buggy_data, 4, 20)

    with open("log/bigcodebench/log_0909-1425_comp_bug_new.json", "w") as f:
        json.dump(all_buggy_data, f, indent=2)


def bug_generate(data, generator, log_file_prefix, dataset_name):
    # ------------------------ Bug generation ------------------------
    results = []
    print("Generating buggy code...")
    for index, item in tqdm.tqdm(enumerate(data)):
        # Filter the failed cases

        task_id = item.get("task_id")
        gt_solution = item.get("gt_solution")
        task_prompt = item.get("task_prompt")

        # Initialize log entry for this item
        log_entry = {
            "task_id": task_id,
            "original_data": item,
            "buggy_code": None,
            "diff": None,
            "is_buggy": None
        }

        bug_type = bug_type_examples[random.randint(0, len(bug_type_examples) - 1)]
        bug_desp_str = ""
        select_ic_ids = np.random.choice(range(len(hard_bug_examples)), 3)
        select_ic_examples = [list(hard_bug_examples.items())[id] for id in select_ic_ids]
        for i, example in enumerate(select_ic_examples):
            bug_desp_str += f"Example {i}: {example[0]}\n{example[1]}\n"
        bug_type.replace("[[bug_description]]", bug_desp_str)
        prompt_text = ONE_BUG_GEN_TEMPLATE.format(
            task_prompt=task_prompt,
            gt_solution=gt_solution if "buggy_code" not in item else item["buggy_code"],
            bug_type=bug_type
        )

        try:
            response = generator(prompt=prompt_text)
            if response and isinstance(response, list) and len(response) > 0:
                raw_output = response[0]
            elif hasattr(response, 'completions') and response.completions:
                raw_output = response.completions[0].content
            else:
                raise ValueError("Unexpected response format from the model.")

            # Parse code block
            match = CODE_BLOCK_REGEX.search(raw_output)
            if match:
                log_entry["buggy_code"] = match.group(1).strip()
            else:
                print("No match found in the response. Full response:", raw_output)

            # Extract JSON diff
            _, _, json_diff = file_diff(gt_solution, log_entry["buggy_code"])
            if json_diff is not None:
                log_entry["diff"] = json_diff
            else:
                print("Error extracting JSON diff from the response. Full response:", raw_output)

        except Exception as e:
            print(f"Error processing task_id {task_id}: {e}")

        results.append(log_entry)

    # Verify buggy
    if dataset_name == "livecodebench":
        verify_file = log_file_prefix + "_bug.json"
        data_to_write = [
            {
                "question_id": entry["task_id"],
                "code_list": [entry["buggy_code"]]
            }
            for entry in results if entry["buggy_code"] is not None
        ]
        if data_to_write:
            with open(verify_file, "w") as f:
                json.dump(data_to_write, f, indent=4)
            fail_ids, correct_ids = verify(dataset_name, verify_file)
        else:
            print("No buggy submissions to evaluate.")
            fail_ids, correct_ids = [], []
    elif dataset_name == "kodcodebench":
        verify_file = log_file_prefix + "_bug.json"
        with open(verify_file, "w") as f:
            data_to_write = [
                {
                    "task_id": entry["task_id"],
                    "solution": [entry["buggy_code"]],
                    "test": entry["original_data"]["test"]
                }
                for entry in results if entry["buggy_code"] is not None
            ]
            json.dump(data_to_write, f, indent=4)
        fail_ids, correct_ids = verify(dataset_name, verify_file)
    elif dataset_name == "bigcodebench":  # bigcodebench
        verify_file = log_file_prefix + "_bug.jsonl"
        with open(verify_file, "w") as f:
            wrote_any = False
            for entry in results:
                if entry["buggy_code"] is not None:
                    json.dump({
                        "task_id": entry["task_id"],
                        "solution": entry["buggy_code"]
                    }, f)
                    f.write("\n")
                    wrote_any = True
        if wrote_any:
            fail_ids, correct_ids = [e["task_id"] for e in results], []
            # fail_ids, correct_ids = verify(dataset_name, verify_file)
        else:
            print("No buggy submissions to evaluate.")
            fail_ids, correct_ids = [], []
    else:
        raise ValueError("Unexpected dataset name.")

    # Update results with success status
    remain_data = []
    buggy_data = []
    for entry in results:
        # A submission that *fails* the evaluation harness is the kind of
        # intentionally broken program we want.  Therefore, `fail_ids` now
        # counts as a **buggy** generation, while `correct_ids` means the
        # code is still correct and needs another mutation attempt.

        if entry["task_id"] in fail_ids:
            entry["is_buggy"] = True
            buggy_data.append({
                "task_id": entry["task_id"],
                "gt_solution": entry["original_data"]["gt_solution"],
                "task_prompt": entry["original_data"]["task_prompt"],
                "bug_count": 1,
                "diff": entry.get("diff"),
                "buggy_code": entry["buggy_code"],
                "test": entry["original_data"].get("test", None)  # For kodcodebench
            })
        elif entry["task_id"] in correct_ids:
            entry["is_buggy"] = False
            remain_data.append(entry["original_data"])

    print("Total buggy code generated: {} out of {}".format(len(buggy_data), len(results)))

    return results, buggy_data


def bug_correct(data, generator, log_file_prefix, dataset_name):
    # ------------------------ Bug correction ------------------------
    if not data:
        print("No buggy data to correct; skipping correction phase.")
        return [], []

    cache = load_symbolic_cache(dataset_name)
    # If cache is empty (no file), seed it from GT diffs so symbolic judge
    # accepts exact reversals immediately and persist for future runs.
    if not cache:
        for item in data:
            tid = item.get("task_id")
            gt_diff = item.get("diff") or {}
            if not gt_diff:
                continue
            task_cache = cache.setdefault(tid, {})
            for _, v in gt_diff.items():
                buggy_line = str(v.get("modified", "")).strip()
                fixed_line = str(v.get("original", "")).strip()
                if buggy_line == "" and fixed_line == "":
                    continue
                task_cache.setdefault(buggy_line, [])
                if fixed_line not in task_cache[buggy_line]:
                    task_cache[buggy_line].append(fixed_line)
        save_symbolic_cache(dataset_name, cache)

    results = []
    print("Buggy code correction...")
    # Collect items that fail symbolic check and need batch unit verification
    to_unit_verify = []  # list of task_ids to verify
    need_cache_update = {}  # task_id -> unmatched_pairs
    tid_to_index = {}  # task_id -> results index

    for index, item in tqdm.tqdm(enumerate(data)):
        task_id = item.get("task_id")
        buggy_code = item.get("buggy_code")
        gt_diff = item.get("diff")
        task_prompt = item.get("task_prompt")

        log_entry = {
            "task_id": task_id,
            "original_data": item,
            "solution": None,
            "sol_diff": None,
            "is_corrected": None
        }

        prompt_text = DEBUG_TEMPLATE.format(
            task_prompt=task_prompt,
            buggy_code=buggy_code
        )

        try:
            response = generator(prompt=prompt_text)
            if response and isinstance(response, list) and len(response) > 0:
                raw_output = response[0]
            elif hasattr(response, 'completions') and response.completions:
                raw_output = response.completions[0].content
            else:
                raise ValueError("Unexpected response format from the model.")

            match = CODE_BLOCK_REGEX.search(raw_output)
            if match:
                log_entry["solution"] = match.group(1).strip()
            else:
                print("No match found in the response. Full response:", raw_output)

            if log_entry["solution"] is not None and buggy_code is not None:
                _, _, json_diff = file_diff(buggy_code, log_entry["solution"])
                log_entry["sol_diff"] = json_diff

        except Exception as e:
            print(f"Error processing task_id {task_id}: {e}")

        # Note from Miaosen: Symbolic first; defer unit tests to a single batch ONLY at the end to improve throughtput
        symbolically_true, unmatched_pairs = symbolic_judge(task_id, log_entry.get("sol_diff"), gt_diff, cache)

        unit_true = False
        if log_entry.get("solution"):
            if symbolically_true:
                unit_true = True
                log_entry["is_corrected"] = True
                # cache any unmatched pairs as newly accepted alternatives
                if unmatched_pairs:
                    task_cache = cache.setdefault(task_id, {})
                    for buggy_line, fixed_line in unmatched_pairs:
                        task_cache.setdefault(buggy_line, [])
                        if fixed_line not in task_cache[buggy_line]:
                            task_cache[buggy_line].append(fixed_line)
            else:
                # Defer unit test: collect for batch verification
                to_unit_verify.append(task_id)
                need_cache_update[task_id] = unmatched_pairs
                log_entry["is_corrected"] = False
        else:
            log_entry["is_corrected"] = False

        eval_dict = {
            "model": getattr(generator, "model", None),
            "solution": log_entry.get("solution"),
            "sol_diff": log_entry.get("sol_diff"),
            "symbolic_true": symbolically_true,
            "unit_true": unit_true,
        }
        log_entry["debug_results"] = eval_dict

        results.append(log_entry)
        tid_to_index[task_id] = len(results) - 1

    # Perform a single batched unit verification for symbolically-failing items
    if to_unit_verify:
        verify_dir = Path("log") / dataset_name
        verify_dir.mkdir(parents=True, exist_ok=True)
        verify_prefix = Path(log_file_prefix).name

        if dataset_name == "bigcodebench":
            vf = str(verify_dir / f"{verify_prefix}_single_correct_batch.jsonl")
            with open(vf, "w") as f:
                for tid in to_unit_verify:
                    idx = tid_to_index[tid]
                    sol = results[idx].get("solution")
                    if sol:
                        json.dump({"task_id": tid, "solution": sol}, f)
                        f.write("\n")
        elif dataset_name == "livecodebench":
            vf = str(verify_dir / f"{verify_prefix}_single_correct_batch.json")
            data_to_write = []
            for tid in to_unit_verify:
                idx = tid_to_index[tid]
                sol = results[idx].get("solution")
                if sol:
                    data_to_write.append({
                        "question_id": results[idx]["task_id"],
                        "code_list": [sol]
                    })
            with open(vf, "w") as f:
                json.dump(data_to_write, f, indent=2)
        elif dataset_name == "kodcodebench":
            vf = str(verify_dir / f"{verify_prefix}_single_correct_batch.json")
            data_to_write = []
            for tid in to_unit_verify:
                idx = tid_to_index[tid]
                sol = results[idx].get("solution")
                original_item = results[idx]["original_data"]
                test_code = original_item.get("test") or original_item.get("original_data", {}).get("test")
                if sol:
                    data_to_write.append({
                        "task_id": results[idx]["task_id"],
                        "solution": [sol],
                        "test": test_code
                    })
            with open(vf, "w") as f:
                json.dump(data_to_write, f, indent=2)
        else:
            vf = None

        if vf:
            try:
                fail_ids, correct_ids = verify(dataset_name, vf)
            except Exception as e:
                print(f"[batch verify] Error during verification: {e}")
                fail_ids, correct_ids = to_unit_verify, []

            # Update results and cache based on batch outcomes
            for tid in to_unit_verify:
                idx = tid_to_index[tid]
                unit_true_final = tid in correct_ids
                results[idx]["debug_results"]["unit_true"] = unit_true_final
                results[idx]["is_corrected"] = unit_true_final
                if unit_true_final:
                    unmatched_pairs = need_cache_update.get(tid) or set()
                    if unmatched_pairs:
                        task_cache = cache.setdefault(tid, {})
                        for buggy_line, fixed_line in unmatched_pairs:
                            task_cache.setdefault(buggy_line, [])
                            if fixed_line not in task_cache[buggy_line]:
                                task_cache[buggy_line].append(fixed_line)

    save_symbolic_cache(dataset_name, cache)

    with open(log_file_prefix + "_correct.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, []
