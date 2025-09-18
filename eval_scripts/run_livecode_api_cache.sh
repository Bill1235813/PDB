python main.py \
  --dataset_name livecodebench \
  --model_name openai/o4-mini-2025-04-16 \
  --input_file /home/zhuwangz/miaosenchai/rescue_code_bench/data/log_0909-1710_comp_bug_new.json \
  --log_prefix  livecode-test\
  --output_prefix correct \
  --max_id_count 10 \
  --temperature 1 \
  --model_api_file /home/zhuwangz/miaosenchai/rescue_code_bench/keys/openai_key.txt \