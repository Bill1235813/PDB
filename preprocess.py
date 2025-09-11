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

def livecodebench_preprocess(raw_data):
    """
    Preprocess the LiveCodeBench
    Expected input structure (one element from the list):
    {
        "question_id": "2727",
        "question_content": "... problem statement ...",
        "code_list": ["code1", "code2", ...],
        "graded_list": [true, false, ...]  # optional
        "pass@1": 1.0,
        ...
    }
    """
    processed_data = []

    # raw_data can be either a list or a dict keyed by task_id
    if isinstance(raw_data, dict):
        iterable = raw_data.values()
    else:
        iterable = raw_data

    for example in iterable:
        task_id = str(example.get("question_id"))
        if not task_id:
            print(f"No task_id for example {example}")
            break

        code_list = example.get("code_list")
        if not code_list:
            print(f"No candidate solutions for task {task_id}")
            break

        task_prompt = example.get("question_content")
        processed_data.append(
            {
                "task_id": task_id,
                "buggy_code": code_list,
                "task_prompt": task_prompt,
            }
        )

    return processed_data

def kodcodebench_preprocess(raw_data):
    processed_data = []

    for id, example in raw_data.items():
        processed_data.append({
            "task_id": id,
            "gt_solution": example["solution"],
            "task_prompt": example["question"],
            "test": example["test"]
        })

    return processed_data
