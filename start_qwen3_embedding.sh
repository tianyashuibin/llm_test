#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_VENV="${VLLM_VENV:-$HOME/.venvs/vllm-cu130}"
source "$VLLM_VENV/bin/activate"

nohup vllm serve Qwen/Qwen3-Embedding-0.6B \
  --runner pooling \
  --dtype half \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.7 \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 8001 > "$SCRIPT_DIR/qwen3_embedding.log" 2>&1 < /dev/null &

echo "$!" > "$SCRIPT_DIR/qwen3_embedding.pid"
echo "Qwen3-Embedding 已在后台启动，PID: $(cat "$SCRIPT_DIR/qwen3_embedding.pid")"
echo "日志: $SCRIPT_DIR/qwen3_embedding.log（请检查日志确认服务就绪）"
