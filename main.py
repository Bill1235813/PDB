import argparse
import json
import dspy
import os
import datetime
import preprocess
from evaluator import Evaluator
from rewrite import rewrite
from bug_generation import bug_correct


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

    if args.model_api_file:
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
    remain_data = raw_data

    # Load the model
    if args.model_api_file:
        # Use API-based model when API file is provided
        api_key = open(model_api_file, "r").read().strip()
        generator_cor = dspy.LM(args.model_name, api_key=api_key, temperature=args.temperature, cache=False, max_tokens=21000)
    else:
        # Use local model server when no API file is provided
        local_model_name = "openai/" + args.model_name
        generator_cor = dspy.LM(local_model_name,
            api_base="http://127.0.0.1:30000/v1",  # Add /v1 prefix for OpenAI-compatible API
            api_key="local",
            model_type="chat",
            max_tokens = 21000,
            temperature=args.temperature,
            cache=False,
            )
        print(f"Using local model server: {local_model_name}")
    print(f"Enter debugging process")
    buggy_code, remain_data = bug_correct(
        remain_data,
        generator_cor,
        log_file_prefix,
        args.dataset_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Dataset name", required=True)
    parser.add_argument("--model_name", type=str, help="Debugging model name", required=True)
    parser.add_argument("--model_api_file", type=str, help="Model API file path under keys (optional - if provided, uses API; if omitted, uses local server)")
    parser.add_argument("--input_file", nargs='+', help="Input file path, under data/{dataset_name}",
                        default="bigcodebench-full-data.json")
    parser.add_argument("--id_filtering_file", type=str, help="ID filtering file path, under data/{dataset_name}",
                        default="id_filtering.json")
    parser.add_argument("--log_prefix", type=str, help="Log file under log/{dataset_name}",
                        default="log")
    parser.add_argument("--output_prefix", type=str, help="Output file path, under output/{dataset_name}",
                        default="buggy_code")
    parser.add_argument("--rewrite", action="store_true", help="Whether to rewrite the code")
    parser.add_argument("--max_iter", type=int, default=1, help="Maximum number of add-bug iterations")
    parser.add_argument("--bug_per_time", type=int, default=20, help="Number of bugs to add per iteration")
    parser.add_argument("--max_id_count", type=int, default=30, help="max number of ids to be used, -1 for no limit")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the generator")

    args = parser.parse_args()
    gen_main(args)
