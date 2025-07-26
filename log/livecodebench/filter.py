import json
import os

def filter_json_objects(input_path, output_path):
    """
    Reads a JSON file containing a list of objects, counts them,
    filters for objects with "is_buggy": true, and writes the
    filtered objects to a new JSON file.

    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{input_path}' is not a valid JSON file.")
        return

    if not isinstance(data, list):
        print("Error: The root of the JSON file is not a list of objects.")
        return

    total_count = len(data)
    print(f"Total number of objects in '{os.path.basename(input_path)}': {total_count}")

    buggy_objects = [item for item in data if item.get("is_buggy") is True]
    buggy_count = len(buggy_objects)
    print(f"Found {buggy_count} objects with 'is_buggy: true'")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(buggy_objects, f, indent=4)
        print(f"Successfully wrote {buggy_count} buggy objects to '{os.path.basename(output_path)}'")
    except IOError as e:
        print(f"Error writing to file '{output_path}': {e}")


def create_dummy_input_file(file_path):
    """Creates a dummy JSON file for testing purposes."""
    dummy_data = [
        {
            "id": "001",
            "description": "This is a normal object.",
            "is_buggy": False
        },
        {
            "id": "002",
            "description": "This object has a bug.",
            "is_buggy": True
        },
        {
            "id": "003",
            "description": "Another normal object.",
            "is_buggy": False
        },
        {
            "id": "004",
            "description": "This one is also buggy.",
            "is_buggy": True
        },
        {
            "id": "005",
            "description": "An object that is not a bug.",
            "is_buggy": False
        }
    ]
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, indent=4)
        print(f"Created a dummy input file: '{os.path.basename(file_path)}'")
    except IOError as e:
        print(f"Could not create dummy input file '{file_path}': {e}")


if __name__ == '__main__':
    # Define file paths.
    input_path = '/home/zhuwangz/miaosenchai/rescue_code_bench/log/livecodebench/true_livecode_gpt4o_added.json'
    
    # Create an output file name based on the input file name.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = 'true_livecode_gpt4o_added_filtered.json'
    output_path = os.path.join(script_dir, output_file)

    # Process the file
    filter_json_objects(input_path, output_path)
