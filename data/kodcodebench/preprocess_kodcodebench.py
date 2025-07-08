from huggingface_hub import snapshot_download
from datasets import load_dataset
import json
import os
import glob


def convert_parquet_to_jsonl(parquet_file_path, output_dir):
    print(f"\nProcessing Parquet file: {os.path.basename(parquet_file_path)}")

    dataset = load_dataset('parquet', data_files=[parquet_file_path])
    base_name = os.path.basename(parquet_file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_jsonl_path = os.path.join(output_dir, f"{file_name_without_ext}.jsonl")

    if 'train' in dataset:
        data = dataset['train']
        data.to_json(output_jsonl_path, orient='records', lines=True)
        print(f"Successfully converted to {output_jsonl_path}")
        return output_jsonl_path
    else:
        print(f"Warning: No 'train' split found in {parquet_file_path}. Skipping.")
        return None


def convert_jsonl_to_json(input_jsonl_path, output_json_path):
    json_list = []
    print(f"Converting {os.path.basename(input_jsonl_path)} to JSON...")
    with open(input_jsonl_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            stripped_line = line.strip()
            if stripped_line:
                json_list.append(json.loads(stripped_line))

    with open(output_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(json_list, outfile, indent=4)

    print(f"Successfully created {output_json_path}")


if __name__ == "__main__":
    repo_id = "KodCode/KodCode-V1"
    # repo_id = "KodCode/KodCode-V1-SFT-4o"
    # repo_id = "KodCode/KodCode-V1-SFT-R1"
    name = repo_id.split("/")[-1]
    folder_to_download = "data"

    pre_fix = name + "_"
    local_dir = pre_fix + "raw"

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=f"data/*",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"The '{repo_id}' data has been downloaded to '{local_dir}'.")

    local_data_directory = os.path.join(local_dir, "data")
    output_directory = name

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_pattern = os.path.join(local_data_directory, '*.parquet')
    parquet_files = glob.glob(file_pattern)

    for parquet_file in parquet_files:
        created_jsonl_file = convert_parquet_to_jsonl(parquet_file, output_directory)
        if created_jsonl_file:
            json_output_path = os.path.splitext(created_jsonl_file)[0] + '.json'
            convert_jsonl_to_json(created_jsonl_file, json_output_path)
            os.remove(created_jsonl_file)
