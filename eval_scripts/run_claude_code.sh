python main.py \
  --dataset_name bigcodebench \
  --model_name claude-code-agent \
  --use_claude_code \
  --debug_mode minimal \
  --input_file /home/zhuwangz/miaosenchai/rescue_code_bench/eval_scripts/sampled_data_t.json \
  --max_iter 3 \
  --claude_timeout 300 \
  --log_prefix unit-claude_code_bigcodebench \
  --output_prefix unit-claude_code_bigcodebench