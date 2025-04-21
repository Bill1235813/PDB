"""
This is to preprocess the files into an input format that can be used for our task

Current input format in JSON / JSONL:
{
    "task_id": "123xxx",
    "gt_solution": "The ground truth solution to the task",
    "task_prompt": "The task description for the task",
}
"""


def bigcodebench_preprocess(raw_data):
    """
    Preprocess the input file into the format required by bigcodebench
    """
    processed_data = []
    for id, example in raw_data.items():
        processed_data.append({
            "task_id": id,
            "gt_solution": example["code_prompt"] + "\n" + example["canonical_solution"],
            "task_prompt": example["instruct_prompt"]
        }
        )
    return processed_data
