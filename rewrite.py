import re
import tqdm
import json
from utils import verify

# Rewriting instructions
REWRITE_TEMPLATE = (
    "Your task is to perform a deep analysis of code and rewrite it with structural and stylistic perturbations. \n"
    "You will be given two parts:\n"
    "PART 1: A problem description/requirement (the task that needs to be solved)\n"
    "PART 2: A correct solution to the problem\n"
    "First, carefully read both the problem description and the provided solution to understand what the code is supposed to do.\n"
    "Then, your task is to perform a deep rewriting (perturbation) of the correct code implementation "
    "without changing its behavior or output.\n"
    "Use the following rewriting rules and you need to rewrite every line if it satisfies the rule:\n"
    "    - Rename some variables to less meaningful or shorter names (e.g., sum_production -> sum_p)\n"
    "    - Remove all the comments and don't include any new comments.\n"
    "    - Modify numeric literals using intermediate arithmetic operations then adjust values before use (e.g., first add 2, then subtract 3, finally add 1).\n"
    "    - Reorder independent statements\n"
    "    - Split compound conditions into multiple sequential or nested if statements.\n"
    "    - Change loop syntax (e.g., for to while or vice versa, if applicable)\n"
    "    - Convert recursion to iteration or vice versa (if functionally equivalent)\n"
    "    - Replace standard operations with weird equivalent alternatives (e.g., use list comprehensions vs. loops, or `a += b` vs. `a = a + b`)\n"
    "    - Invert control flow where logical structure remains the same (e.g., replace `if not ...` with an inverted block)\n"
    "    - Modify literal expressions without changing their value (e.g., replace `4` with `1 + 2 + 1`, or use hex/binary literals)\n"
    "    - Replace boolean expressions with logically equivalent forms (e.g., `x and y` vs. `not (not x or not y)`)\n"
    "    - Transform chained if-elif-else structures into nested if blocks.\n"
    "    - Merge or flatten adjacent or nested if blocks.\n"
    "    - Arbitrarily split or merge lines of code.\n"
    "\n"
    "Your response should ONLY contain the rewritten Python code, which can be directly executed"
    "\n"
    "---\n"
    "PART 1: Problem Description\n"
    "```text\n{task_prompt}\n```\n\n"
    "PART 2: Solution\n"
    "```python\n{gt_solution}\n```\n\n"
    "---\n"
    "Output format:\n"
    "```python\n"
    "[Your Rewritten Python Code Here]\n"
    "```\n"
    "---\n"
    "Rewritten Code Output (using the specified format):\n"
)

CODE_BLOCK_REGEX = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def rewrite(data, rewrite_model, dataset_name, log_file):
    results = []

    for index, item in tqdm.tqdm(enumerate(data), total=len(data)):
        task_id = item.get("task_id")
        gt_solution = item.get("gt_solution")
        task_prompt = item.get("task_prompt")

        # Initialize log entry for this item
        log_entry = {
            "task_id": task_id,
            "original_data": item,
            "rewritten_solution": None,
            "success": None
        }

        prompt_text = REWRITE_TEMPLATE.format(
            task_prompt=task_prompt,
            gt_solution=gt_solution
        )

        try:
            response = rewrite_model(prompt=prompt_text)
            if response and isinstance(response, list) and len(response) > 0:
                raw_output = response[0]
            elif hasattr(response, 'completions') and response.completions:
                raw_output = response.completions[0].content
            else:
                raise ValueError("Unexpected response format from the model.")

            # Parse Regex
            match = CODE_BLOCK_REGEX.search(raw_output)
            if match:
                log_entry["rewritten_solution"] = match.group(1).strip()
            else:
                print("No match found in the response. Full response:", raw_output)

        except Exception as e:
            print(f"Error processing task_id {task_id}: {e}")

        results.append(log_entry)

    # Verify
    verify_file = log_file + "_verify.jsonl"
    if dataset_name == "livecodebench":
        verify_file = log_file + ".json"
        with open(verify_file, "w") as f:
            data_to_write = [
                {
                    "question_id": entry["task_id"],
                    "code_list": [entry["rewritten_solution"]]
                }
                for entry in results if entry["rewritten_solution"] is not None
            ]
            json.dump(data_to_write, f, indent=4)
    elif dataset_name == "kodcodebench":  # kodcodebench
        with open(verify_file, "w") as f:
            data_to_write = [
                {
                    "task_id": entry["task_id"],
                    "solution": [entry["rewritten_solution"]],
                    "test": entry["original_data"]["test"]
                }
                for entry in results if entry["rewritten_solution"] is not None
            ]
            json.dump(data_to_write, f, indent=4)
    else:  # bigcodebench
        with open(verify_file, "w") as f:
            for entry in results:
                if entry["rewritten_solution"] is not None:
                    json.dump({
                        "task_id": entry["task_id"],
                        "solution": entry["rewritten_solution"]
                    }, f)
                    f.write("\n")

    fail_ids, correct_ids = verify(dataset_name, verify_file)

    # Update results with success status
    new_data = []
    for entry in results:
        if entry["task_id"] in fail_ids:
            entry["success"] = False
        elif entry["task_id"] in correct_ids:
            entry["success"] = True
            new_data.append({
                "task_id": entry["task_id"],
                "gt_solution": entry["rewritten_solution"],
                "task_prompt": entry["original_data"]["task_prompt"],
                "test": entry["original_data"].get("test", None)  # For kodcodebench
            })
        else:
            print("There is a parsing problem during the rewrite process.")
            entry["success"] = None

    # Save the results
    with open(log_file + ".json", "w") as f:
        json.dump(results, f, indent=2)

    return new_data
