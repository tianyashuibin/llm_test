import statistics
import time

from openai import OpenAI
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain how language model inference works."
MAX_TOKENS = 128
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def run_once(max_tokens):
    start = time.perf_counter()
    first_chunk_time = None
    last_chunk_time = None
    chunks = []
    chunk_times = []

    stream = client.completions.create(
        model=MODEL,
        prompt=PROMPT,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    )

    for event in stream:
        text = event.choices[0].text
        if text:
            now = time.perf_counter()
            if first_chunk_time is None:
                first_chunk_time = now
            last_chunk_time = now
            chunks.append(text)
            chunk_times.append(now)

    end = time.perf_counter()
    output_text = "".join(chunks)
    output_tokens = len(
        tokenizer.encode(output_text, add_special_tokens=False)
    )

    ttft = (
        first_chunk_time - start
        if first_chunk_time is not None else None
    )

    # 这是客户端观察到的近似 TPOT，包含服务端和传输开销。
    # 以 Tokenizer 统计的输出 Token 数估算，不是 GPU 纯 Decode 时间。
    tpot = (
        (last_chunk_time - first_chunk_time) / (output_tokens - 1)
        if first_chunk_time is not None
        and last_chunk_time is not None
        and output_tokens > 1
        else None
    )

    return {
        "ttft": ttft,
        "tpot": tpot,
        "wall": end - start,
        "tokens": output_tokens,
        "chunks": len(chunks),
    }

print("Warmup...")
for i in range(WARMUP_RUNS):
    print(f"Warmup {i + 1}:", run_once(32))

print("\nBenchmark...")
results = []

for i in range(BENCHMARK_RUNS):
    r = run_once(MAX_TOKENS)
    results.append(r)
    print(
        f"Run {i + 1}: "
        f"TTFT={r['ttft']:.4f}s, "
        f"TPOT~={r['tpot']:.4f}s/token, "
        f"wall={r['wall']:.4f}s, "
        f"tokens={r['tokens']}, chunks={r['chunks']}"
    )

def show(key, label, scale=1.0):
    values = [r[key] for r in results if r[key] is not None]
    if values:
        print(
            f"{label}: mean={statistics.mean(values) * scale:.4f}, "
            f"median={statistics.median(values) * scale:.4f}"
        )

print("\n========== Summary ==========")
show("ttft", "TTFT (seconds)")
show("tpot", "Approx TPOT (seconds/token)")
show("wall", "End-to-end latency (seconds)")

total_tokens = sum(r["tokens"] for r in results)
total_wall = sum(r["wall"] for r in results)
print(f"Output throughput: {total_tokens / total_wall:.2f} tokens/s")
print("Note: client-side streaming measurements; TPOT is approximate.")
