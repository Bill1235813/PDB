from json import load, dump

with open("/home/zhuwangz/miaosenchai/rescue_code_bench/data/livecodebench/filtered_eval.json", "r") as f:
    data = load(f)

filtered_data = []
i = 0
for item in data:
    if item["contest_date"].startswith("2025"):
        i+=1 
        filtered_data.append(item)

with open("2025_filtered_eval.json", "w") as f:
    dump(filtered_data, f, indent=4)

print(i)