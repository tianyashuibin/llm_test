#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-baseline}"
if (( $# )); then shift; fi
mkdir -p results
python_bin="${PYTHON_BIN:-python3}"
case "$mode" in
  env)
    nvidia-smi
    "$python_bin" -c 'import torch, vllm; print("torch", torch.__version__, "CUDA", torch.version.cuda, "vLLM", vllm.__version__); print("GPU", torch.cuda.get_device_name())'
    nsys --version
    ncu --version
    ;;
  baseline)
    "$python_bin" profile_single_request.py --output results/baseline.json "$@"
    ;;
  systems)
    nsys profile --trace=cuda,nvtx,osrt --sample=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --output=results/single_request \
      "$python_bin" profile_single_request.py --output results/systems.json "$@"
    ;;
  stats)
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
      results/single_request.nsys-rep
    ;;
  compute)
    : "${KERNEL_REGEX:?Set KERNEL_REGEX from the observed Nsight Systems kernel name}"
    ncu --profile-from-start off --target-processes all \
      --kernel-name-base demangled --kernel-name "regex:${KERNEL_REGEX}" \
      --launch-skip "${LAUNCH_SKIP:-0}" --launch-count "${LAUNCH_COUNT:-1}" \
      --set "${NCU_SET:-basic}" --export results/kernel \
      "$python_bin" profile_single_request.py --output results/compute.json "$@"
    ;;
  *) echo "Usage: bash run.sh {env|baseline|systems|stats|compute} [model arguments]" >&2; exit 2 ;;
esac
