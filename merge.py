import json
import os

file_dir = "output/bigcodebench"
in_files = ["o4_buggy_code_0923-1233.json", "ar_buggy_code_0923-1235.json"]

data = []
for in_file in in_files:
    data += json.load(open(os.path.join(file_dir, in_file)))

with open(os.path.join(file_dir, "bigcodebench_buggy_v0.json"), "w") as f:
    json.dump(data, f, indent=2)