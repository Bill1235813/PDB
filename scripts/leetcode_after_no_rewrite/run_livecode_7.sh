python main.py \
  --dataset_name livecodebench \
  --model_name openai/o4-mini-2025-04-16 \
  --model_api_file /home/ec2-user/DebugBench/rescue_code_bench/keys/openai_key.txt\
  --input_file /home/ec2-user/DebugBench/Livecode-m/data/buggy_code/leetcode_after_no_rewrite_bug.json\
  --log_prefix  leetcode_after_no_rewrite_7\
  --output_prefix correct \
  --max_iter 1 \
  --bug_per_time 3 \
  --max_id_count 100 \
  --temperature 1

# --input_file /home/ec2-user/DebugBench/Livecode-m/data/buggy_code/gpt4o_before_rewrite_bug.json\*
# --input_file /home/ec2-user/DebugBench/Livecode-m/data/buggy_code/gpt4o_before_no_rewrite_bug.json\*
# --input_file /home/ec2-user/DebugBench/Livecode-m/data/buggy_code/gpt4o_after_rewrite_bug.json\*
# --input_file /home/ec2-user/DebugBench/Livecode-m/data/buggy_code/gpt4o_after_no_rewrite_bug.json\*
# --input_file /home/ec2-user/DebugBench/Livecode-m/data/buggy_code/leetcode_after_no_rewrite_bug.json\*
# --input_file /home/ec2-user/DebugBench/Livecode-m/data/buggy_code/leetcode_after_rewrite_bug.json\*
# --input_file /home/ec2-user/DebugBench/Livecode-m/data/buggy_code/leetcode_before_no_rewrite_bug.json\*
# --input_file /home/ec2-user/DebugBench/Livecode-m/data/buggy_code/leetcode_before_rewrite_bug.json\