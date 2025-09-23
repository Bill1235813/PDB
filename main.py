import argparse
import copy
import json
import dspy
import os
import datetime
import tqdm
from evaluator import Evaluator
from utils import file_diff
from module import MINIMAL_DEBUG_TEMPLATE, FREE_DEBUG_TEMPLATE, CODE_BLOCK_REGEX


def bug_correct(data, generator, log_file_prefix, output_file, args):
    # ------------------------ Bug correction ------------------------
    if not data:
        print("No buggy data to correct; skipping correction phase.")
        return [], []

    cache = None
    # dataset_name = args.dataset_name
    # cache = load_symbolic_cache(dataset_name)
    # # If cache is empty (no file), seed it from GT diffs so symbolic judge
    # # accepts exact reversals immediately and persist for future runs.
    # if not cache:
    #     for item in data:
    #         tid = item.get("task_id")
    #         gt_diff = item.get("diff") or {}
    #         if not gt_diff:
    #             continue
    #         task_cache = cache.setdefault(tid, {})
    #         for _, v in gt_diff.items():
    #             buggy_line = str(v.get("modified", "")).strip()
    #             fixed_line = str(v.get("original", "")).strip()
    #             if buggy_line == "" and fixed_line == "":
    #                 continue
    #             task_cache.setdefault(buggy_line, [])
    #             if fixed_line not in task_cache[buggy_line]:
    #                 task_cache[buggy_line].append(fixed_line)
    #     save_symbolic_cache(dataset_name, cache)

    results = []
    print("Buggy code correction...")
    # Collect items that fail symbolic check and need batch unit verification
    to_unit_verify = []  # list of task_ids to verify
    need_cache_update = {}  # task_id -> unmatched_pairs
    tid_to_index = {}  # task_id -> results index

    for index, item in tqdm.tqdm(enumerate(data)):
        task_id = item.get("task_id")
        buggy_code = item.get("buggy_code")
        task_prompt = item.get("task_prompt")

        for round in range(args.max_iter):
            log_entry = copy.deepcopy(item)
            log_entry["debug_results"] = {"model": args.model_name}
            log_entry["round"] = round + 1

            prompt_template = MINIMAL_DEBUG_TEMPLATE if args.debug_mode == "minimal" else FREE_DEBUG_TEMPLATE
            prompt_text = prompt_template.format(
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
                    log_entry["debug_results"]["solution"] = match.group(1).strip()
                else:
                    log_entry["debug_results"]["solution"] = raw_output
                    print("No match found in the response. Use full response:", raw_output)

                if log_entry["debug_results"]["solution"] is not None and buggy_code is not None:
                    _, _, json_diff = file_diff(item.get("buggy_code"), log_entry["debug_results"]["solution"])
                    log_entry["debug_results"]["sol_diff"] = json_diff
                else:
                    log_entry["debug_results"]["sol_diff"] = {}

            except Exception as e:
                log_entry["debug_results"]["solution"] = None
                log_entry["debug_results"]["sol_diff"] = {}
                print(f"Error processing task_id {task_id}: {e}")

            results.append(log_entry)

            if log_entry["debug_results"]["solution"] is not None:
                buggy_code = log_entry["debug_results"]["solution"]
            else:
                break

        # Note from Miaosen: Symbolic first; defer unit tests to a single batch ONLY at the end to improve throughtput
        # symbolically_true, unmatched_pairs = symbolic_judge(task_id, log_entry.get("sol_diff"), gt_diff, cache)

        # unit_true = False
        # if log_entry.get("solution"):
        #     if symbolically_true:
        #         unit_true = True
        #         log_entry["is_corrected"] = True
        #         # cache any unmatched pairs as newly accepted alternatives
        #         if unmatched_pairs:
        #             task_cache = cache.setdefault(task_id, {})
        #             for buggy_line, fixed_line in unmatched_pairs:
        #                 task_cache.setdefault(buggy_line, [])
        #                 if fixed_line not in task_cache[buggy_line]:
        #                     task_cache[buggy_line].append(fixed_line)
        #     else:
        #         # Defer unit test: collect for batch verification
        #         to_unit_verify.append(task_id)
        #         need_cache_update[task_id] = unmatched_pairs
        #         log_entry["is_corrected"] = False
        # else:
        #     log_entry["is_corrected"] = False
        #
        # eval_dict = {
        #     "model": getattr(generator, "model", None),
        #     "solution": log_entry.get("solution"),
        #     "sol_diff": log_entry.get("sol_diff"),
        #     "symbolic_true": symbolically_true,
        #     "unit_true": unit_true,
        # }
        # log_entry["debug_results"] = eval_dict

        # results.append(log_entry)
    #     tid_to_index[task_id] = len(results) - 1
    #
    # # Perform a single batched unit verification for symbolically-failing items
    # if to_unit_verify:
    #     verify_dir = Path("log") / dataset_name
    #     verify_dir.mkdir(parents=True, exist_ok=True)
    #     verify_prefix = Path(log_file_prefix).name
    #
    #     if dataset_name == "bigcodebench":
    #         vf = str(verify_dir / f"{verify_prefix}_single_correct_batch.jsonl")
    #         with open(vf, "w") as f:
    #             for tid in to_unit_verify:
    #                 idx = tid_to_index[tid]
    #                 sol = results[idx].get("solution")
    #                 if sol:
    #                     json.dump({"task_id": tid, "solution": sol}, f)
    #                     f.write("\n")
    #     elif dataset_name == "livecodebench":
    #         vf = str(verify_dir / f"{verify_prefix}_single_correct_batch.json")
    #         data_to_write = []
    #         for tid in to_unit_verify:
    #             idx = tid_to_index[tid]
    #             sol = results[idx].get("solution")
    #             if sol:
    #                 data_to_write.append({
    #                     "question_id": results[idx]["task_id"],
    #                     "code_list": [sol]
    #                 })
    #         with open(vf, "w") as f:
    #             json.dump(data_to_write, f, indent=2)
    #     elif dataset_name == "kodcodebench":
    #         vf = str(verify_dir / f"{verify_prefix}_single_correct_batch.json")
    #         data_to_write = []
    #         for tid in to_unit_verify:
    #             idx = tid_to_index[tid]
    #             sol = results[idx].get("solution")
    #             original_item = results[idx]["original_data"]
    #             test_code = original_item.get("test") or original_item.get("original_data", {}).get("test")
    #             if sol:
    #                 data_to_write.append({
    #                     "task_id": results[idx]["task_id"],
    #                     "solution": [sol],
    #                     "test": test_code
    #                 })
    #         with open(vf, "w") as f:
    #             json.dump(data_to_write, f, indent=2)
    #     else:
    #         vf = None
    #
    #     if vf:
    #         try:
    #             fail_ids, correct_ids = verify(dataset_name, vf)
    #         except Exception as e:
    #             print(f"[batch verify] Error during verification: {e}")
    #             fail_ids, correct_ids = to_unit_verify, []
    #
    #         # Update results and cache based on batch outcomes
    #         for tid in to_unit_verify:
    #             idx = tid_to_index[tid]
    #             unit_true_final = tid in correct_ids
    #             results[idx]["debug_results"]["unit_true"] = unit_true_final
    #             results[idx]["is_corrected"] = unit_true_final
    #             if unit_true_final:
    #                 unmatched_pairs = need_cache_update.get(tid) or set()
    #                 if unmatched_pairs:
    #                     task_cache = cache.setdefault(tid, {})
    #                     for buggy_line, fixed_line in unmatched_pairs:
    #                         task_cache.setdefault(buggy_line, [])
    #                         if fixed_line not in task_cache[buggy_line]:
    #                             task_cache[buggy_line].append(fixed_line)

    # save_symbolic_cache(dataset_name, cache)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


def eval_main(args):
    data_dir = os.path.join("output", args.dataset_name)
    log_dir = os.path.join("log", args.dataset_name)
    output_dir = os.path.join("eval", args.dataset_name)

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

    if args.model_api_file:
        model_api_file = os.path.join("keys", args.model_api_file)
    log_file_prefix = os.path.join(log_dir, args.log_prefix) + "_" + time_to_add + "_"
    eval_file = args.input_file[0].rsplit(".")[0]
    output_file = os.path.join(output_dir, args.output_prefix) + f"_on_{eval_file}.json"

    # Load the dataset
    if len(args.input_file) == 1:
        input_file = os.path.join(data_dir, args.input_file[0])
        buggy_data = json.load(open(input_file, "r"))
    else:
        input_files = [os.path.join(data_dir, args.input_file[i]) for i in range(len(args.input_file))]
        raw_data_list = [json.load(open(input_file, "r")) for input_file in input_files]
        # concatenate the list of dictionaries
        buggy_data = raw_data_list[0]
        for d in raw_data_list[1:]:
            buggy_data.extend(d)

    # Load the model
    if args.model_api_file:
        # Use API-based model when API file is provided
        api_key = open(model_api_file, "r").read().strip()
        generator_cor = dspy.LM(args.model_name, api_key=api_key, temperature=args.temperature,
                                max_tokens=args.max_tokens)
    else:
        # Use local model server when no API file is provided
        local_model_name = "openai/" + args.model_name
        generator_cor = dspy.LM(local_model_name,
                                api_base="http://127.0.0.1:30000/v1",  # Add /v1 prefix for OpenAI-compatible API
                                api_key="local",
                                model_type="chat",
                                max_tokens=args.max_tokens,
                                temperature=args.temperature,
                                cache=False,
                                )
        print(f"Using local model server: {local_model_name}")
    print(f"Enter debugging process")
    results = bug_correct(
        buggy_data,
        generator_cor,
        log_file_prefix,
        output_file,
        args
    )

    # # Run evaluation and save outputs
    # try:
    #     evaluator = Evaluator(results)
    #     evaluator.run_evaluation()
    # except Exception as e:
    #     print(f"Evaluation failed: {e}")
    #
    # try:
    #     with open(output_file, "w") as f:
    #         json.dump(results, f, indent=2)
    #     print(f"Saved results to {output_file}")
    # except Exception as e:
    #     print(f"Failed to save results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Dataset name", required=True)
    parser.add_argument("--model_name", type=str, help="Evaluation model name", required=True)
    parser.add_argument("--model_api_file", type=str,
                        help="Model API file path under keys (optional - if provided, uses API; if omitted, uses local server)")
    parser.add_argument("--debug_mode", choices=["free", "minimal"], default="minimal", type=str)
    parser.add_argument("--input_file", nargs='+', help="Input buggy file path, under output/{dataset_name}",
                        default="buggy_code_0923-0047.json")
    parser.add_argument("--log_prefix", type=str, help="Log file under log/{dataset_name}",
                        default="log_eval")
    parser.add_argument("--output_prefix", type=str, help="Output file path, under eval/{dataset_name}",
                        default="gpt-4o")
    parser.add_argument("--max_iter", type=int, default=2, help="Maximum number of add-bug iterations")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum number of tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the generator")

    args = parser.parse_args()
    eval_main(args)
