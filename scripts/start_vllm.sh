#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_VENV="${VLLM_VENV:-$HOME/.venvs/vllm-cu130}"
source "$VLLM_VENV/bin/activate"

nohup vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --host 127.0.0.1 \
  --port 8000 > "$SCRIPT_DIR/vllm.log" 2>&1 < /dev/null &

echo "$!" > "$SCRIPT_DIR/vllm.pid"
echo "vLLM 已在后台启动，PID: $(cat "$SCRIPT_DIR/vllm.pid")"
echo "日志: $SCRIPT_DIR/vllm.log（请检查日志确认服务就绪）"
