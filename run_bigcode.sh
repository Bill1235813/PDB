#python -m pdb bug_generation.py \
#  --dataset_name bigcodebench \
#  --model_name openai/o4-mini \
#  --model_api_file openai_key.txt \
#  --input_file bigcodebench-full-data.json \
#  --log_prefix test \
#  --output_prefix test \
#  --bug_per_time 2 \
#  --max_bugs 2 \
#  --max_id_count 2 \
#  --temperature 1.0 \
#  --rewrite

python -m pdb bug_generation.py \
  --dataset_name bigcodebench \
  --model_name openai/o4-mini \
  --model_api_file openai_key.txt \
  --input_file bigcodebench-full-data.json \
  --log_prefix o4_log \
  --output_prefix o4_buggy_code \
  --bug_per_time 20 \
  --max_bugs 4 \
  --max_id_count 100 \
  --temperature 1.0 \
  --rewrite

#python -m pdb main.py \
#  --dataset_name bigcodebench \
#  --model_name openai/gpt-4o \
#  --model_api_file openai_key.txt \
#  --input_file test_0923-1348.json \
#  --max_iter 2 \
#  --temperature 0.7
#
#python -m pdb evaluator.py \
#  --dataset_name bigcodebench \
#  --model_name openai/gpt-4o \
#  --input_file gpt-4o_on_test_0923-1348.json  \
#  --max_iter 2

