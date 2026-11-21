#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-baseline}"
if (( $# )); then shift; fi
mkdir -p results
python_bin="${PYTHON_BIN:-python3}"

# Model presets (same three variants as tmp.txt). The quantized ones are
# profiled together by the quant-* modes below.
model_fp16="Qwen/Qwen3-1.7B"
model_w8a16="Qwen/Qwen3-1.7B-GPTQ-Int8"
model_w8a8="nytopop/Qwen3-1.7B.w8a8"
quant_tags=(w8a16 w8a8)

model_for_tag() {
  case "$1" in
    fp16) echo "$model_fp16" ;;
    w8a16) echo "$model_w8a16" ;;
    w8a8) echo "$model_w8a8" ;;
    *) return 1 ;;
  esac
}

# Tag names every artifact so runs for different models do not overwrite each
# other: results/<tag>.nsys-rep, results/systems_<tag>.json, results/kernel_<tag>.ncu-rep.
tag="${TAG:-single_request}"

# Default NCU targets: the top kernels by cumulative GPU time in each variant's
# cuda_gpu_kern_sum. One regex per line; NCU profiles each separately.
# Override per tag with KERNEL_REGEX_<TAG>, globally with KERNEL_REGEX; both
# take a newline-separated list.
#
# w8a16 has two: marlin handles the quantized linear layers, while lm_head
# is left unquantized by GPTQ and runs through cuBLAS gemvx instead. That
# GEMV is 32 instances of ~2.41 ms (one per decoded token) at 21.2%.
default_kernel_regexes() {
  case "$1" in
    fp16) printf '%s\n' 'gemvx::kernel.*__half' ;;
    w8a16) printf '%s\n' 'marlin::Marlin<' 'gemvx::kernel.*__half' ;;
    w8a8) printf '%s\n' 'cutlass::Kernel2<.*OpMultiplyAddSaturate' ;;
    *) return 1 ;;
  esac
}

# Prints the regex list for a tag, one per line. Exits non-zero if empty.
kernel_regexes_for_tag() {
  local var="KERNEL_REGEX_$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  local list=""
  if declare -p "$var" >/dev/null 2>&1; then list="${!var}"; fi
  if [[ -z "$list" ]]; then list="${KERNEL_REGEX:-}"; fi
  if [[ -z "$list" ]]; then list="$(default_kernel_regexes "$1" 2>/dev/null || true)"; fi
  [[ -n "$list" ]] || {
    echo "Set $var (or KERNEL_REGEX) from the observed Nsight Systems kernel name" >&2
    return 1
  }
  printf '%s\n' "$list"
}

do_systems() {  # tag, then extra model arguments
  local tag="$1"; shift
  nsys profile --trace=cuda,nvtx,osrt --sample=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true \
    --output="results/${tag}" \
    "$python_bin" profile_single_request.py --output "results/systems_${tag}.json" "$@"
}

do_stats() {  # tag
  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
    "results/$1.nsys-rep"
}

# Turns a kernel regex into a short filesystem-safe label, e.g.
# 'marlin::Marlin<' -> 'marlin', 'cutlass::Kernel2<...' -> 'cutlass'.
regex_slug() {
  printf '%s' "$1" | sed -e 's/::.*//' -e 's/[^A-Za-z0-9_-]/_/g' | cut -c1-32
}

do_compute() {  # tag, then extra model arguments
  local tag="$1"; shift
  local regexes n=0 regex slug suffix
  regexes="$(kernel_regexes_for_tag "$tag")" || return 1
  local total
  total="$(printf '%s\n' "$regexes" | grep -c . || true)"
  while IFS= read -r regex; do
    [[ -n "$regex" ]] || continue
    n=$((n + 1))
    # Keep the historical filename when there is a single target; otherwise
    # suffix each report so targets do not overwrite each other.
    if (( total == 1 )); then suffix=""; else suffix="_$(regex_slug "$regex")"; fi
    slug="kernel_${tag}${suffix}"
    echo "=== ncu: ${tag}${suffix} <- regex:${regex} ==="
    ncu --profile-from-start off --target-processes all \
      --kernel-name-base demangled --kernel-name "regex:${regex}" \
      --launch-skip "${LAUNCH_SKIP:-0}" --launch-count "${LAUNCH_COUNT:-1}" \
      --set "${NCU_SET:-basic}" --force-overwrite \
      --export "results/${slug}" \
      "$python_bin" profile_single_request.py --output "results/compute_${tag}${suffix}.json" "$@"
  done <<< "$regexes"
}

# Single-model runs: TAG selects the artifact prefix, and when TAG names a
# preset the matching --model is supplied unless the caller passed one.
# Bash 3.2 errors on expanding an empty array under `set -u`, so expand via
# the `${arr[@]+...}` guard form.
model_args=("$@")
if (( ${#model_args[@]} == 0 )) && preset_model="$(model_for_tag "$tag" 2>/dev/null)"; then
  model_args=(--model "$preset_model")
fi
if (( ${#model_args[@]} )); then
  model_expand=("${model_args[@]}")
else
  model_expand=()
fi

case "$mode" in
  env)
    nvidia-smi
    "$python_bin" -c 'import torch, vllm; print("torch", torch.__version__, "CUDA", torch.version.cuda, "vLLM", vllm.__version__); print("GPU", torch.cuda.get_device_name())'
    nsys --version
    ncu --version
    ;;
  baseline)
    "$python_bin" profile_single_request.py --output "results/baseline_${tag}.json" "${model_expand[@]+"${model_expand[@]}"}"
    ;;
  systems)
    do_systems "$tag" "${model_expand[@]+"${model_expand[@]}"}"
    ;;
  stats)
    do_stats "$tag"
    ;;
  compute)
    do_compute "$tag" "${model_expand[@]+"${model_expand[@]}"}"
    ;;
  quant-systems)
    for t in "${quant_tags[@]}"; do
      echo "=== nsys: $t ($(model_for_tag "$t")) ==="
      do_systems "$t" --model "$(model_for_tag "$t")" "$@"
    done
    ;;
  quant-stats)
    for t in "${quant_tags[@]}"; do
      echo "=== nsys stats: $t ==="
      do_stats "$t"
    done
    ;;
  quant-compute)
    for t in "${quant_tags[@]}"; do
      echo "=== ncu: $t ($(model_for_tag "$t")) ==="
      do_compute "$t" --model "$(model_for_tag "$t")" "$@"
    done
    ;;
  *)
    cat >&2 <<'EOF'
Usage: bash run.sh {env|baseline|systems|stats|compute|quant-systems|quant-stats|quant-compute} [model arguments]

Single-model modes use TAG (default: single_request) to name artifacts, e.g.
  TAG=w8a8 bash run.sh systems                # model preset resolved from TAG
  TAG=w8a8 KERNEL_REGEX='...' bash run.sh compute

Quantized sweep modes run both W8A16 and W8A8 back to back:
  bash run.sh quant-systems    # results/{w8a16,w8a8}.nsys-rep
  bash run.sh quant-stats
  bash run.sh quant-compute    # KERNEL_REGEX_W8A16 / KERNEL_REGEX_W8A8, else KERNEL_REGEX
EOF
    exit 2
    ;;
esac
