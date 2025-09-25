import re
import os
import datetime
import numpy as np
import random
import tqdm
import json
import copy
import dspy
import argparse
import logging
import preprocess
from pathlib import Path
from utils import verify, file_diff, load_symbolic_cache, save_symbolic_cache, symbolic_judge, build_verify, \
    save_formatted_gt, mark_editable_lines
from module import Rewriter, BugInjector, CODE_BLOCK_REGEX, DIFF_STR_PATTERN
from examples import hard_bug_examples, bug_type_examples
from collections import defaultdict
from codebleu import calc_codebleu


def rewrite(data, dataset_name, log_file_prefix, max_try=2, lang="python", threshold=0.5):
    results = []
    rewriter = Rewriter()

    print("Rewriting")
    for index, item in tqdm.tqdm(enumerate(data), total=len(data)):
        task_id = item.get("task_id")
        gt_solution = item.get("gt_solution")
        task_prompt = item.get("task_prompt")
        log_entry = copy.deepcopy(item)

        trial = 0
        while trial < max_try:
            try:
                response = rewriter(task_prompt=task_prompt, gt_solution=gt_solution)
                match = CODE_BLOCK_REGEX.search(response.rewritten_code)
                if match:
                    log_entry["rewritten_solution"] = match.group(1).strip()
                else:
                    log_entry["rewritten_solution"] = response.rewritten_code.strip()
                    print("No match found in the response. Use full response:", response.rewritten_code.strip())
            except Exception as e:
                log_entry["rewritten_solution"] = None
                print(f"Error processing task_id {task_id}: {e}")

            if log_entry["rewritten_solution"] and \
                    calc_codebleu([gt_solution], [log_entry["rewritten_solution"]], lang)["codebleu"] <= threshold:
                break
            else:
                trial += 1

        results.append(log_entry)

    verify_file = build_verify(dataset_name, log_file_prefix + "_rewrite_verify", results,
                               sol_field="rewritten_solution")
    try:
        fail_ids, correct_ids = verify(dataset_name, verify_file, timeout=3600)
    except Exception as e:
        print(f"Error verifying. Save first.")
        with open(log_file_prefix + "_rewrite.json", "w") as f:
            json.dump(results, f, indent=2)
        return results, []

    # Update results with success status
    new_data = []
    for entry in results:
        if entry["task_id"] in correct_ids:
            entry["rewritten_success"] = True
            new_data.append(entry)
        else:
            entry["rewritten_success"] = False

    print("Rewriting success:", len(new_data), "in", len(results))
    with open(log_file_prefix + "_rewrite.json", "w") as f:
        json.dump(results, f, indent=2)

    for entry in new_data:
        entry["gt_solution"] = entry["rewritten_solution"]
    return results, new_data


def bug_generate(data, dataset_name, log_file_prefix, bug_per_example, ic_size=4):
    # ------------------------ Bug generation ------------------------
    results = []
    bug_gen = BugInjector()

    for count in range(bug_per_example):
        print(f"Generating buggy code step {count}")
        for index, item in tqdm.tqdm(enumerate(data)):
            task_id = item.get("task_id") + f"_{count}"
            gt_solution = item.get("gt_solution")
            task_prompt = item.get("task_prompt")
            log_entry = copy.deepcopy(item)
            log_entry["task_id"] = task_id

            bug_type = bug_type_examples[random.randint(0, len(bug_type_examples) - 1)]
            bug_desp_str = ""
            select_ic_ids = np.random.choice(range(len(hard_bug_examples)), ic_size)
            select_ic_examples = [list(hard_bug_examples.items())[id] for id in select_ic_ids]
            for i, example in enumerate(select_ic_examples):
                bug_desp_str += f"Example {i}: {example[0]}\n{example[1]}\n"
            bug_type.replace("[[bug_description]]", bug_desp_str)

            line_num, line_text = log_entry["editable_lines"][np.random.choice(len(log_entry["editable_lines"]))]
            line_to_edit = f"{line_num}. {line_text}"

            try:
                response = bug_gen(task_prompt=task_prompt, gt_solution=gt_solution, bug_type=bug_type,
                                   line_to_edit=line_to_edit)
                match = CODE_BLOCK_REGEX.search(response.buggy_code)
                if match:
                    log_entry["buggy_code"] = match.group(1).strip()
                    _, _, json_diff = file_diff(gt_solution, log_entry["buggy_code"].strip())
                    if json_diff is not None and len(json_diff) == 1:
                        log_entry["diff"] = json_diff
                        log_entry["bug_count"] = 1
                        results.append(log_entry)
                    else:
                        log_entry["diff"] = None
                        print("JSON diff wrong format from the response. Full response:",
                              log_entry["buggy_code"].strip())
                else:
                    log_entry["buggy_code"] = response.buggy_code.strip()
                    print("No match found in the response. Use full response:", response.buggy_code.strip())

            except Exception as e:
                print(f"Error processing task_id {task_id}: {e}")

    # Verify buggy
    _, formatted_gt = save_formatted_gt(dataset_name, log_file_prefix + f"bug_gen_gt", results)
    verify_file = build_verify(dataset_name, log_file_prefix + "_bug_verify", results, sol_field="buggy_code")
    try:
        fail_ids, correct_ids = verify(dataset_name, verify_file, gt_file=formatted_gt, timeout=7200)
    except Exception as e:
        print(f"Error verifying. Save first.")
        with open(log_file_prefix + "_bug.json", "w") as f:
            json.dump(results, f, indent=2)
        return results, []

    new_data = []
    # Update results with success status
    for entry in results:
        entry["editable_lines"] = len(entry["editable_lines"])
        if entry["task_id"] in fail_ids:
            entry["is_buggy"] = True
            entry["task_id"] = entry["task_id"].rsplit("_", 1)[0]
            new_data.append(entry)
        elif entry["task_id"] in correct_ids:
            entry["is_buggy"] = False

    print("Total buggy code generated: {} out of {}".format(len(new_data), len(results)))
    with open(log_file_prefix + "_bug.json", "w") as f:
        json.dump(results, f, indent=2)
    return results, new_data


def diff_to_str(k, v):
    return f"{k}: {v['original']} --> {v['modified']}"


def str_to_diff(s):
    m = DIFF_STR_PATTERN.match(s)
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


def bug_compose(buggy_data, max_bugs, compose_per_example, output_file):
    """
    :param buggy_data: list of {
        "task_id": str,
        "gt_solution": str,
        "task_prompt": str,
        "bug_count": 1,
        "diff": str,
        "buggy_code": str,
    }
    :param max_bugs: Number of bugs to generate.
    :param compose_per_example: Number of composition to generate per example.
    :param output_file:
    :return: dict[bug_count] -> list of buggy examples
    """
    merged = defaultdict(lambda: {
        "task_id": None,
        "gt_solution": None,
        "task_prompt": None,
        "diff": [],
        "test": None
    })

    filtered_buggy_data = []
    for entry in buggy_data:
        tid = entry["task_id"]
        if merged[tid]["task_id"] is None:  # first time seeing this task_id
            merged[tid]["task_id"] = tid
            merged[tid]["gt_solution"] = entry["gt_solution"]
            merged[tid]["task_prompt"] = entry["task_prompt"]
            merged[tid]["test"] = entry["test"] if "test" in entry else None

        if entry.get("diff"):
            for line, diff in entry["diff"].items():
                diff_str = diff_to_str(line, diff)
                if not diff_str in merged[tid]["diff"]:  # deduplicate
                    merged[tid]["diff"].append(diff_str)
                    filtered_buggy_data.append(entry)

    all_bugs = {1: filtered_buggy_data}  # bug_count: bug_data
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

    print("Total buggy code generated: ", [len(all_bugs[i]) for i in range(1, max_bugs + 1)])

    id_counter = defaultdict(lambda: 0)
    all_bug_flatten = []
    for all_bug in all_bugs.values():
        for item in all_bug:
            task_id = item["task_id"] + f"_{id_counter[item['task_id']]}"
            id_counter[item["task_id"]] += 1
            item["task_id"] = task_id
            all_bug_flatten.append(item)

    # Save the buggy code
    print("Saving composed buggy code to", output_file)
    with open(output_file, "w") as f:
        json.dump(all_bug_flatten, f, indent=2)

    return all_bug_flatten


def gen_main(args):
    data_dir = os.path.join("data", args.dataset_name)
    log_dir = os.path.join("log", args.dataset_name)
    output_dir = os.path.join("output", args.dataset_name)

    if not os.path.exists("keys"):
        os.makedirs("keys")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Add datetime
    time_to_add = datetime.datetime.now().strftime("%m%d-%H%M")

    model_api_file = os.path.join("keys", args.model_api_file)
    id_filtering_file = os.path.join(data_dir, args.id_filtering_file)
    log_file_prefix = os.path.join(log_dir, args.log_prefix) + "_" + time_to_add + "_"
    output_file = os.path.join(output_dir, args.output_prefix) + "_" + time_to_add + ".json"

    # Load the dataset
    if len(args.input_file) == 1:
        input_file = os.path.join(data_dir, args.input_file[0])
        raw_data = json.load(open(input_file, "r"))
    else:
        input_files = [os.path.join(data_dir, args.input_file[i]) for i in range(len(args.input_file))]
        raw_data_list = [json.load(open(input_file, "r")) for input_file in input_files]
        # concatenate the list of dictionaries
        raw_data = raw_data_list[0]
        for d in raw_data_list[1:]:
            raw_data.extend(d)

    # Handle optional ID filtering file
    if os.path.exists(id_filtering_file):
        id_filtering = set(map(str, json.load(open(id_filtering_file, "r"))))
    else:
        id_filtering = None

    # Agnostic ID filtering
    if isinstance(raw_data, list):
        processed_dict = {}
        for d in raw_data:
            item_id = None
            if 'task_id' in d:  # bigcodebench
                item_id = str(d['task_id'])
            elif 'question_id' in d:  # livecodebench & kodcodebench
                item_id = str(d['question_id'])

            if item_id and (not id_filtering or item_id in id_filtering):
                processed_dict[item_id] = d
        raw_data = processed_dict
    elif isinstance(raw_data, dict):
        if id_filtering:
            raw_data = {
                str(k): v for k, v in raw_data.items() if str(k) in id_filtering
            }

    if args.max_id_count > 0:
        raw_data = dict(list(raw_data.items())[:args.max_id_count])

    print("Preprocessing data...")
    raw_data = eval("preprocess." + args.dataset_name + "_preprocess")(raw_data)

    # Load the model
    api_key = open(model_api_file, "r").read().strip()
    generator = dspy.LM(args.model_name, api_key=api_key, temperature=args.temperature, max_tokens=args.max_tokens)
    dspy.settings.configure(lm=generator)

    if args.reload_from_save:
        print("Reload data...")
        buggy_data = json.load(open(os.path.join(log_dir, args.reload_from_save)))
        parsed_prefix = args.reload_from_save.split("__bug.json")[0]
        formatted_gt = os.path.join(log_dir, parsed_prefix + "_bug_gen_gt.jsonl")
        verify_file = os.path.join(log_dir, parsed_prefix + "__bug_verify.jsonl")
        try:
            fail_ids, correct_ids = verify(args.dataset_name, verify_file, gt_file=formatted_gt, timeout=7200)
            new_data = []
            for entry in buggy_data:
                if entry["task_id"] in fail_ids:
                    entry["is_buggy"] = True
                    entry["task_id"] = entry["task_id"].rsplit("_", 1)[0]
                    new_data.append(entry)
                elif entry["task_id"] in correct_ids:
                    entry["is_buggy"] = False

            print("Total buggy code generated: {} out of {}".format(len(new_data), len(buggy_data)))
            buggy_data = new_data
        except Exception as e:
            print(f"Error verifying. Save first.")
            with open(log_file_prefix + "_bug.json", "w") as f:
                json.dump(buggy_data, f, indent=2)
    else:
        if args.rewrite:
            print("Rewriting code...")
            _, remain_data = rewrite(raw_data, args.dataset_name, log_file_prefix)
        else:
            remain_data = raw_data

        mark_editable_lines(args.dataset_name, remain_data)
        _, buggy_data = bug_generate(remain_data, args.dataset_name, log_file_prefix, args.bug_per_time)

    bug_compose(buggy_data, args.max_bugs, args.bug_per_time, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Dataset name", required=True)
    parser.add_argument("--model_name", type=str, help="Debugging model name", required=True)
    parser.add_argument("--model_api_file", type=str, required=True, help="Model API file is required for generation")
    parser.add_argument("--input_file", nargs='+', help="Input file path, under data/{dataset_name}",
                        default="bigcodebench-full-data.json")
    parser.add_argument("--reload_from_save", type=str, default="", help="Reload from saved dir")
    parser.add_argument("--id_filtering_file", type=str, help="ID filtering file path, under data/{dataset_name}",
                        default="id_filtering.json")
    parser.add_argument("--log_prefix", type=str, help="Log file under log/{dataset_name}",
                        default="log")
    parser.add_argument("--output_prefix", type=str, help="Output file path, under output/{dataset_name}",
                        default="buggy_code")
    parser.add_argument("--rewrite", action="store_true", help="Whether to rewrite the code")
    parser.add_argument("--bug_per_time", type=int, default=20, help="Number of bugs to add per iteration")
    parser.add_argument("--max_bugs", type=int, default=4, help="Max number of bugs to compose")
    parser.add_argument("--max_tokens", type=int, default=16000, help="Maximum number of tokens")
    parser.add_argument("--max_id_count", type=int, default=30, help="max number of ids to be used, -1 for no limit")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the generator")

    logging.getLogger().setLevel(logging.ERROR)

    args = parser.parse_args()
    gen_main(args)
