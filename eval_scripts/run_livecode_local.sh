python main.py \
  --dataset_name livecodebench \
  --model_name  Qwen/Qwen2.5-Coder-7B-Instruct\
  --input_file /home/ec2-user/DebugBench/Livecode-m/data/buggy_code/new_after_no_rewrite.json\
  --log_prefix  o4_after_no_rewrite_1\
  --output_prefix correct \
  --bug_per_time 3 \
  --max_id_count 100 \
  --temperature 1 \