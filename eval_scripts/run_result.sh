python3 evaluator.py \
    --dataset_name bigcodebench \
    --input_file /home/zhuwangz/miaosenchai/rescue_code_bench/eval/bigcodebench/claude_code_bigcodebench_on_sampled_data.json \
    --output_dir eval \
    --model_name claude-code-agent \
    --stride 2 \
    --precision_tolerance 3 \
    --max_iter 1

