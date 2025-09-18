CUDA_VISIBLE_DEVICES=0 \
python3 -m sglang.launch_server \
  --model-path /home/ec2-user/DebugBench/Model/Qwen/Qwen2.5-Coder-7B-Instruct \
  --tp-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.9 \