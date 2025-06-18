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
    "- Do NOT change any variable names.\n\n"
    "You must inject exactly:\n"
    "- {bug_per_time} insidious bugs (subtle logical errors that are hard to spot)\n"
    "Important rules:\n"
    "- Do not include any comments inside the code.\n"
    "- Do not add any extra output or formatting.\n\n"
    "Only output a single code block with the final, modified version of the code containing all 7 bugs and NO comments."
    "You need to check there is NO COMMENT inside your generation for the final step"
    "\n"
    "---\n"
    "PART 1: Problem Description\n"
    "```text\n{task_prompt}\n```\n\n"
    "PART 2: Solution\n"
    "```python\n{gt_solution}\n```\n\n"
    "---\n"
    "Output format:\n"
    "```python\n"
    "[Buggy code here]\n"
    "```\n"
    "Diff in JSON format:{{\n"
    "  [line_number]: {{\n"
    "    \"original\": \"[original code]\",\n"
    "    \"modified\": \"[modified code]\",\n"
    "  }},\n"
    "  ...\n"
    "}}\n"
    "---\n"
    "Buggy Code Output (using the specified format):\n"
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
    "\n"
    "---\n"
    "PART 1: Problem Description\n"
    "```text\n{task_prompt}\n```\n\n"
    "PART 2: Buggy Code\n"
    "```python\n{buggy_code}\n```\n\n"
    "---\n"
    "Output format:\n"
    "```python\n"
    "[Corrected code here]\n"
    "```\n"
    "Diff in JSON format:{{\n"
    "  [line_number]: {{\n"
    "    \"original\": \"[original code]\",\n"
    "    \"modified\": \"[modified code]\",\n"
    "  }},\n"
    "  ...\n"
    "}}\n"
    "---\n"
    "Corrected Code Output (using the specified format):\n"
)

CODE_BLOCK_REGEX = re.compile(r"^\s*```python\s*\n?(.*?)\n?^\s*```\s*$", re.DOTALL | re.MULTILINE)
DIFF_REGEX = re.compile(r"Diff in JSON format:(.*?)---", re.DOTALL)


def extract_json_diff(text):
    match = DIFF_REGEX.search(text)

    if not match:
        return None

    json_str = match.group(1).strip()
    json_str = re.sub(r',\s*\.\.\.', '', json_str)
    try:
        # Parse the JSON string
        diff_dict = json.loads(json_str)
        converted_dict = {}
        for key, value in diff_dict.items():
            if key.isdigit():
                converted_dict[int(key)] = value
            else:
                converted_dict[key] = value
        return converted_dict
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None


def bug_generate_correct(data, generator, bug_per_time, log_file_prefix, dataset_name):
    """Run the bug-injection + self-repair loop for a single dataset.

    The *dataset_name* argument is forwarded to the verification helper so that
    the correct evaluation harness is invoked (e.g BigCodeBench vs
    LiveCodeBench).
    """
    remain_data, buggy_data = bug_generate(
        data, generator, bug_per_time, log_file_prefix, dataset_name
    )
    hard_buggy_data, easy_buggy_data = bug_correct(
        buggy_data, generator, log_file_prefix, dataset_name
    )
    return hard_buggy_data, remain_data + easy_buggy_data


def bug_generate(data, generator, bug_per_time, log_file_prefix, dataset_name):
    # ------------------------ Bug generation ------------------------
    results = []
    print("Generating buggy code...")
    for index, item in tqdm.tqdm(enumerate(data)):
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
    verify_file = log_file_prefix + "_verifybug.jsonl"
    if dataset_name == "livecodebench":
        verify_file = log_file_prefix + "_verifybug.json"
        with open(verify_file, "w") as f:
            data_to_write = [
                {
                    "question_id": entry["task_id"],
                    "code_list": [entry["buggy_code"]]
                }
                for entry in results if entry["buggy_code"] is not None
            ]
            json.dump(data_to_write, f, indent=4)
    else: # bigcodebench
        with open(verify_file, "w") as f:
            for entry in results:
                if entry["buggy_code"] is not None:
                    json.dump({
                        "task_id": entry["task_id"],
                        "solution": entry["buggy_code"]
                    }, f)
                    f.write("\n")
    fail_ids, correct_ids = verify(dataset_name, verify_file)

    # Update results with success status
    remain_data = []
    buggy_data = []
    for entry in results:
        if entry["task_id"] in fail_ids:
            entry["is_buggy"] = False
            remain_data.append(entry["original_data"])
        elif entry["task_id"] in correct_ids:
            entry["is_buggy"] = True
            buggy_data.append({
                "task_id": entry["task_id"],
                "gt_solution": entry["original_data"]["gt_solution"],
                "task_prompt": entry["original_data"]["task_prompt"],
                "diff": entry["diff"] if "diff" not in entry["original_data"] else entry["diff"].update(
                    entry["original_data"]["diff"]),
                "buggy_code": entry["buggy_code"],
            })
        else:
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
        diff = item.get("diff")
        task_prompt = item.get("task_prompt")

        # Initialize log entry for this item
        log_entry = {
            "task_id": task_id,
            "original_data": item,
            "solution": None,
            "sol_diff": None,
            "is_corrected": None
        }

        prompt_text = BUG_GEN_TEMPLATE.format(
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
    verify_file = log_file_prefix + "_verifycorrect.json"
    if dataset_name == "livecodebench":
        with open(verify_file, "w") as f:
            data_to_write = [
                {
                    "question_id": entry["task_id"],
                    "code_list": [entry["solution"]]
                }
                for entry in results if entry["solution"] is not None
            ]
            json.dump(data_to_write, f, indent=4)
    else:
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
            hard_buggy_data.append(entry["original_data"])
        elif entry["task_id"] in correct_ids:
            entry["is_corrected"] = True
            easy_buggy_data.append(entry["original_data"])
        else:
            easy_buggy_data.append(entry["original_data"])

    print("Total hard buggy code generated: {} out of {}".format(len(hard_buggy_data), len(results)))

    # Save the buggy code log
    with open(log_file_prefix + "_correct.json", "w") as f:
        json.dump(results, f, indent=2)

    return hard_buggy_data, easy_buggy_data
