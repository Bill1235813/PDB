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


def livecodebench_comp_bug_preprocess(raw_data):
    """
    Expected input shape:
    - Dict mapping bug_count (e.g., "1", "2", ...) -> list[entry]
    - Or directly a list[entry]

    Each entry typically contains:
    {
        "task_id": str,
        "gt_solution": str,
        "task_prompt": str,
        "bug_count": int,
        "diff": { line_no: { original, modified }, ... },
        "buggy_code": str,
        "test": optional
    }

    Output format per item (what bug_correct expects for livecodebench):
    {
        "task_id": str,
        "buggy_code": str,
        "task_prompt": str,
        "diff": optional
    }
    """
    processed_data = []

    # Flatten to a list of entries
    if isinstance(raw_data, dict):
        # Values should be lists of entries
        entries = []
        for v in raw_data.values():
            if isinstance(v, list):
                entries.extend(v)
        # If not found, fallback to dict values directly
        if not entries:
            entries = list(raw_data.values())
    elif isinstance(raw_data, list):
        entries = raw_data
    else:
        entries = []

    for ex in entries:
        if not isinstance(ex, dict):
            continue

        task_id = str(ex.get("task_id")) if ex.get("task_id") is not None else None
        buggy_code = ex.get("buggy_code")
        task_prompt = ex.get("task_prompt") or ""
        diff = ex.get("diff")
        gt_solution = ex.get("gt_solution")
        bug_count = ex.get("bug_count")
        test = ex.get("test")

        if not task_id or not buggy_code:
            continue

        processed_item = {
            "task_id": task_id,
            "buggy_code": buggy_code,
            "task_prompt": task_prompt,
        }
        if isinstance(diff, dict):
            processed_item["diff"] = diff
        if isinstance(gt_solution, str):
            processed_item["gt_solution"] = gt_solution
        if isinstance(bug_count, int):
            processed_item["bug_count"] = bug_count
        if test is not None:
            processed_item["test"] = test

        processed_data.append(processed_item)

    return processed_data
