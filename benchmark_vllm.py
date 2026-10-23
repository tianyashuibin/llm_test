import statistics
import time

import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INPUT_LEN = 256
OUTPUT_LEN = 128
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5

print("Initializing vLLM...")
llm = LLM(
    model=MODEL,
    dtype="half",
    max_model_len=1024,
    gpu_memory_utilization=0.7,
    enforce_eager=True,
)

# 构造精确长度的输入 Token，避免文本分词长度不一致。
tokenizer = llm.get_tokenizer()
base_ids = tokenizer.encode(
    "Explain how language model inference works.",
    add_special_tokens=False,
)
prompt_ids = (
    base_ids * ((INPUT_LEN + len(base_ids) - 1) // len(base_ids))
)[:INPUT_LEN]

prompt = {"prompt_token_ids": prompt_ids}

warmup_params = SamplingParams(
    temperature=0.0,
    max_tokens=32,
    ignore_eos=True,
)
bench_params = SamplingParams(
    temperature=0.0,
    max_tokens=OUTPUT_LEN,
    ignore_eos=True,
)

def run_once(params):
    start = time.perf_counter()
    outputs = llm.generate([prompt], params)
    wall_time = time.perf_counter() - start
    result = outputs[0]
    return result, wall_time

print("\nWarming up...")
for i in range(WARMUP_RUNS):
    _, elapsed = run_once(warmup_params)
    print(f"Warmup {i + 1}: wall={elapsed:.3f}s")

# 记录正式测量阶段的 PyTorch CUDA 峰值分配量。
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

results = []
wall_times = []

print("\nBenchmarking...")
for i in range(BENCHMARK_RUNS):
    result, wall = run_once(bench_params)
    wall_times.append(wall)

    n_input = len(result.prompt_token_ids or [])
    n_output = len(result.outputs[0].token_ids)
    metrics = getattr(result, "metrics", None)

    ttft = None
    tpot = None

    if metrics is not None:
        arrival = getattr(metrics, "arrival_time", None)
        first = getattr(metrics, "first_token_time", None)
        last = getattr(metrics, "last_token_time", None)

        if arrival is not None and first is not None:
            ttft = first - arrival

        if first is not None and last is not None and n_output > 1:
            tpot = (last - first) / (n_output - 1)

    # 端到端 wall time / 输出 Token 数。
    # 这不是严格的 Decode TPOT，包含调度和其他开销。
    wall_tpot = wall / n_output if n_output else float("nan")

    results.append({
        "ttft": ttft,
        "tpot": tpot,
        "wall": wall,
        "wall_tpot": wall_tpot,
        "input_tokens": n_input,
        "output_tokens": n_output,
    })

    print(
        f"Run {i + 1}: "
        f"input={n_input}, output={n_output}, "
        f"wall={wall:.3f}s, "
        f"TTFT={ttft if ttft is not None else 'N/A'}, "
        f"TPOT={tpot if tpot is not None else 'N/A'}"
    )

def summarize(key, label, scale=1.0):
    values = [r[key] for r in results if r[key] is not None]
    if values:
        print(
            f"{label}: mean={statistics.mean(values) * scale:.3f}, "
            f"median={statistics.median(values) * scale:.3f}"
        )
    else:
        print(f"{label}: unavailable")

print("\n========== Summary ==========")
summarize("ttft", "TTFT (seconds)")
summarize("tpot", "TPOT (seconds/token)")
summarize("wall", "End-to-end latency (seconds)")
summarize("wall_tpot", "Wall time/output token (seconds/token)")

total_tokens = sum(r["output_tokens"] for r in results)
total_wall = sum(wall_times)
print(f"Output throughput (tokens/s): {total_tokens / total_wall:.3f}")
print(
    "PyTorch CUDA peak allocated memory (GiB): "
    f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.3f}"
)

print("\nBenchmark complete.")
