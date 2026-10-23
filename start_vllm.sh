source ~/.venvs/vllm-cu130/bin/activate

nohup vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.7 \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 8000 > vllm.log 2>&1 &
