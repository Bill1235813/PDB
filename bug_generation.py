import re
import tqdm
import json
from utils import verify

BUG_GEN_TEMPLATE = (
    "Your task is to perform a deep analysis of a code snippet and intentionally introduce bugs with NO comments.\n"
    "You will be given two components:\n"
    "PART 1: A problem description outlining the intended functionality \n"
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

DEBUG_TEMPLATE = (
    "Analyze and debug the given Python implementation that contains errors \n"
    "Identify the bugs, explain the issues, and fix only the bugs in the code. Do not generate a new solution.\n\n"
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
    "Diff in JSON format (valid JSON):\n"
    "```json\n{{\n  \"<line_number>\": {{ \"original\": \"<orig line>\", \"modified\": \"<new line>\" }},\n  ...\n}}\n```\n"
    "Corrected Code Output (use the format above):\n"
)

CODE_BLOCK_REGEX = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
DIFF_REGEX = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_diff(text):
    match = DIFF_REGEX.search(text)

    if not match:
        return None

    json_str = match.group(1).strip()
    # remove placeholder lines with '...' and trailing commas before closing brace
    json_str = re.sub(r",?\s*\.\.\.\s*", "", json_str)
    try:
        diff_dict = json.loads(json_str)
        return diff_dict
    except Exception:
        return None


def bug_generate_correct(data, generator_add, genrator_cor, bug_per_time, log_file_prefix, dataset_name):
    """Run the bug-injection + self-repair loop for a single dataset.

    The *dataset_name* argument is forwarded to the verification helper so that
    the correct evaluation harness is invoked (e.g BigCodeBench vs
    LiveCodeBench).
    """
    remain_data, buggy_data = bug_generate(
        data, generator_add, bug_per_time, log_file_prefix, dataset_name
    )
    hard_buggy_data, easy_buggy_data = bug_correct(
        buggy_data, genrator_cor, log_file_prefix, dataset_name
    )
    return hard_buggy_data, remain_data + easy_buggy_data


def bug_generate(data, generator, bug_per_time, log_file_prefix, dataset_name):
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

        prompt_text = BUG_GEN_TEMPLATE.format(
            task_prompt=task_prompt,
            gt_solution=gt_solution if "buggy_code" not in item else item["buggy_code"],
            bug_per_time=bug_per_time
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
            json_diff = extract_json_diff(raw_output)
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
    else:  # bigcodebench
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
            fail_ids, correct_ids = verify(dataset_name, verify_file)
        else:
            print("No buggy submissions to evaluate.")
            fail_ids, correct_ids = [], []

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
                "diff": entry.get("diff"),
                "buggy_code": entry["buggy_code"],
                "test": entry["original_data"].get("test", None)  # For kodcodebench
            })
        elif entry["task_id"] in correct_ids:
            entry["is_buggy"] = False
            remain_data.append(entry["original_data"])

    print("Total buggy code generated: {} out of {}".format(len(buggy_data), len(results)))

    # Save the buggy code log
    with open(log_file_prefix + "_bug.json", "w") as f:
        json.dump(results, f, indent=2)

    return remain_data, buggy_data


def bug_correct(data, generator, log_file_prefix, dataset_name):
    # ------------------------ Bug correction ------------------------
    if not data:
        print("No buggy data to correct; skipping correction phase.")
        return [], []

    results = []
    print("Buggy code correction...")
    for index, item in tqdm.tqdm(enumerate(data)):
        task_id = item.get("task_id")
        buggy_code = item.get("buggy_code")
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

            # Parse code block
            match = CODE_BLOCK_REGEX.search(raw_output)
            if match:
                log_entry["solution"] = match.group(1).strip()
            else:
                print("No match found in the response. Full response:", raw_output)

            # Extract JSON diff
            json_diff = extract_json_diff(raw_output)
            if json_diff is not None:
                log_entry["sol_diff"] = json_diff
            else:
                print("Error extracting JSON diff from the response. Full response:", raw_output)

        except Exception as e:
            print(f"Error processing task_id {task_id}: {e}")

        results.append(log_entry)

    # Verify buggy
    if dataset_name == "livecodebench":
        verify_file = log_file_prefix + "correct.json"
        with open(verify_file, "w") as f:
            data_to_write = [
                {
                    "question_id": entry["task_id"],
                    "code_list": [entry["solution"]]
                }
                for entry in results if entry["solution"] is not None
            ]
            json.dump(data_to_write, f, indent=4)
    elif dataset_name == "kodcodebench":
        verify_file = log_file_prefix + "_correct.json"
        with open(verify_file, "w") as f:
            data_to_write = [
                {
                    "task_id": entry["task_id"],
                    "solution": [entry["solution"]],
                    "test": entry["original_data"]["test"]  # For kodcodebench
                }
                for entry in results if entry["solution"] is not None
            ]
            json.dump(data_to_write, f, indent=4)
    else:
        verify_file = log_file_prefix + "_correct.jsonl"
        with open(verify_file, "w") as f:
            for entry in results:
                if entry["solution"] is not None:
                    json.dump({
                        "task_id": entry["task_id"],
                        "solution": entry["solution"]
                    }, f)
                    f.write("\n")
    fail_ids, correct_ids = verify(dataset_name, verify_file)

    # Update results with success status
    hard_buggy_data = []
    easy_buggy_data = []
    for entry in results:
        if entry["task_id"] in fail_ids:
            entry["is_corrected"] = False
            eval_dict = {
                "model": generator.model,
                "solution": entry["solution"],
                "sol_diff": entry["sol_diff"],
            }
            hard_buggy_data.append(entry["original_data"])
            hard_buggy_data[-1]["debug_results"] = eval_dict
        elif entry["task_id"] in correct_ids:
            entry["is_corrected"] = True
            easy_buggy_data.append(entry["original_data"])

    print("Total number of problems debugged by powerful model: {} out of {}".format(len(easy_buggy_data), len(results)))

    # Save the buggy code log
    with open(log_file_prefix + "_correct.json", "w") as f:
        json.dump(results, f, indent=2)

    return hard_buggy_data, easy_buggy_data
