#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_VENV="${VLLM_VENV:-$HOME/.venvs/vllm-cu130}"
source "$VLLM_VENV/bin/activate"

python -c 'import bge_m3_sparse_processor' || {
  echo "请先在 $VLLM_VENV 中安装 bge-m3-sparse-plugin。" >&2
  exit 1
}

nohup vllm serve BAAI/bge-m3 \
  --runner pooling \
  --hf-overrides '{"architectures":["BgeM3EmbeddingModel"]}' \
  --io-processor-plugin bge_m3_sparse_plugin \
  --dtype half \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.7 \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 8000 > "$SCRIPT_DIR/bge_m3.log" 2>&1 < /dev/null &

echo "$!" > "$SCRIPT_DIR/bge_m3.pid"
echo "BGE-M3 已在后台启动，PID: $(cat "$SCRIPT_DIR/bge_m3.pid")"
echo "日志: $SCRIPT_DIR/bge_m3.log（请检查日志确认服务就绪）"
