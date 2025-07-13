import argparse
import json
import dspy
import os
import datetime
import preprocess
from evaluator import Evaluator
from rewrite import rewrite
from bug_generation import bug_generate_correct


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
            # TODO: I don't know why we handle the filtering here.
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

    # Preprocess the data
    print("Preprocessing data...")
    raw_data = eval("preprocess." + args.dataset_name + "_preprocess")(raw_data)

    # Load the model
    api_key = open(model_api_file, "r").read().strip()
    generator = dspy.LM(args.model_name, api_key=api_key, temperature=1.0, cache=False, max_tokens=21000)

    if args.rewrite:
        print("Rewriting code...")
        remain_data = rewrite(raw_data, generator, args.dataset_name, log_file_prefix + "rewrite")
    else:
        remain_data = raw_data

    valid_buggy_code = []
    generator_add = dspy.LM("gpt-4o-2024-08-06", api_key=api_key, temperature=0.7, cache=False, max_tokens=16000)
    generator_cor = dspy.LM("o4-mini-2025-04-16", api_key=api_key, temperature=1.0, cache=False, max_tokens=21000)
    for i in range(args.max_iter):
        print(f"Generating buggy code, iteration {i + 1}...")
        buggy_code, remain_data = bug_generate_correct(
            remain_data,
            generator_add,
            generator_cor,
            args.bug_per_time,
            log_file_prefix + "bug_iter" + str(i + 1),
            args.dataset_name,
        )
        valid_buggy_code.extend(buggy_code)

    print("Total buggy code generated: ", len(valid_buggy_code))

    # Save the buggy code
    print("Saving buggy code...")
    with open(output_file, "w") as f:
        json.dump(valid_buggy_code, f, indent=2)

    evaluator = Evaluator(valid_buggy_code)
    evaluator.run_evaluation()

    # Save the evaluation
    print("Saving evaluation...")
    with open(output_file, "w") as f:
        json.dump(valid_buggy_code, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Dataset name", required=True)
    parser.add_argument("--model_name", type=str, help="Generator model name", default="openai/o4-mini-2025-04-16")
    parser.add_argument("--model_api_file", type=str, help="Model API file path under keys",
                        default="openai_key.txt")
    parser.add_argument("--input_file", nargs='+', help="Input file path, under data/{dataset_name}",
                        default="bigcodebench-full-data.json")
    parser.add_argument("--id_filtering_file", type=str, help="ID filtering file path, under data/{dataset_name}",
                        default="id_filtering.json")
    parser.add_argument("--log_prefix", type=str, help="Log file under log/{dataset_name}",
                        default="log")
    parser.add_argument("--output_prefix", type=str, help="Output file path, under output/{dataset_name}",
                        default="buggy_code")
    parser.add_argument("--rewrite", action="store_true", help="Whether to rewrite the code")
    parser.add_argument("--max_iter", type=int, default=5, help="Maximum number of add-bug iterations")
    parser.add_argument("--bug_per_time", type=int, default=3, help="Number of bugs to add per iteration")
    parser.add_argument("--max_id_count", type=int, default=30, help="max number of ids to be used, -1 for no limit")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the generator")

    args = parser.parse_args()
    gen_main(args)
