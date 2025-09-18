python main.py \
  --dataset_name livecodebench \
  --model_name openai/o4-mini-2025-04-16 \
  --input_file /home/ec2-user/DebugBench/Livecode-m/data/buggy_code/new_after_no_rewrite.json\
  --log_prefix  o4_after_no_rewrite_1\
  --output_prefix correct \
  --bug_per_time 3 \
  --max_id_count 100 \
  --temperature 1 \
  --model_api_file /home/ec2-user/DebugBench/rescue_code_bench/keys/openai_key.txt \