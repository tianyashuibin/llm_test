source ~/.venvs/vllm-cu130/bin/activate

nohup vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --host 127.0.0.1 \
  --port 8000 > vllm.log 2>&1 &
