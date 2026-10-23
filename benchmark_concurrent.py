
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
#PROMPT = "Explain how language model inference works."
MAX_TOKENS = 128

CONCURRENCIES = [1, 2, 4, 8]
WARMUP_ROUNDS = 2
BENCHMARK_ROUNDS = 5

tokenizer = AutoTokenizer.from_pretrained(MODEL)

BASE_PROMPT = (
    "Explain language model inference, including tokenization, "
    "prefill, transformer attention, KV cache, autoregressive decoding, "
    "continuous batching, GPU memory bandwidth, and inference latency. "
)

token_ids = tokenizer.encode(
    BASE_PROMPT * 100,
    add_special_tokens=False,
)

PROMPT = tokenizer.decode(
    token_ids[:256],
    skip_special_tokens=True,
)

print(
    "Prompt tokens:",
    len(tokenizer.encode(PROMPT, add_special_tokens=False)),
)

def make_client():
    return OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="not-needed",
        timeout=300.0,
        max_retries=0,
    )

def run_request(client, max_tokens=MAX_TOKENS):
    start = time.perf_counter()
    first_chunk = None
    last_chunk = None
    chunks = []

    stream = client.completions.create(
        model=MODEL,
        prompt=PROMPT,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    )

    try:
        for event in stream:
            text = event.choices[0].text
            if text:
                now = time.perf_counter()
                if first_chunk is None:
                    first_chunk = now
                last_chunk = now
                chunks.append(text)
    finally:
        stream.close()

    end = time.perf_counter()
    output_text = "".join(chunks)
    output_tokens = len(
        tokenizer.encode(output_text, add_special_tokens=False)
    )

    ttft = first_chunk - start if first_chunk is not None else None

    # 客户端近似 TPOT：首个到最后一个文本块的间隔 / (输出 token 数 - 1)
    tpot = None
    if (
        first_chunk is not None
        and last_chunk is not None
        and output_tokens > 1
    ):
        tpot = (last_chunk - first_chunk) / (output_tokens - 1)

    return {
        "ttft": ttft,
        "tpot": tpot,
        "wall": end - start,
        "tokens": output_tokens,
    }

def percentile(values, p):
    values = sorted(values)
    if not values:
        return float("nan")
    index = (len(values) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)

def run_round(concurrency, clients, max_tokens=MAX_TOKENS):
    # 统一记录整轮起止时间；所有请求在独立线程中并发发起
    round_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_request, clients[i], max_tokens)
            for i in range(concurrency)
        ]
        results = [f.result() for f in futures]

    round_end = time.perf_counter()
    round_wall = round_end - round_start
    total_tokens = sum(r["tokens"] for r in results)

    return {
        "results": results,
        "wall": round_wall,
        "tokens": total_tokens,
        "throughput": total_tokens / round_wall if round_wall else 0,
    }

def show_request_metric(results, key, label):
    values = [r[key] for r in results if r[key] is not None]
    if not values:
        print(f"  {label}: N/A")
        return

    print(
        f"  {label}: mean={statistics.mean(values) * 1000:.2f} ms, "
        f"median={statistics.median(values) * 1000:.2f} ms, "
        f"p95={percentile(values, 0.95) * 1000:.2f} ms"
    )

clients = [make_client() for _ in range(max(CONCURRENCIES))]

for concurrency in CONCURRENCIES:
    print(f"\n{'=' * 20} Concurrency={concurrency} {'=' * 20}")

    # 每个并发档位先预热
    for i in range(WARMUP_ROUNDS):
        warm = run_round(
            concurrency, clients, max_tokens=32
        )
        print(
            f"Warmup {i + 1}: wall={warm['wall']:.3f}s, "
            f"throughput={warm['throughput']:.2f} tokens/s"
        )

    rounds = []

    for i in range(BENCHMARK_ROUNDS):
        result = run_round(concurrency, clients)
        rounds.append(result)
        print(
            f"Run {i + 1}: wall={result['wall']:.3f}s, "
            f"total_tokens={result['tokens']}, "
            f"throughput={result['throughput']:.2f} tokens/s"
        )

    # 汇总请求级延迟：将所有轮次的请求合并
    all_requests = [
        request
        for round_result in rounds
        for request in round_result["results"]
    ]

    print("\nSummary:")
    show_request_metric(all_requests, "ttft", "TTFT")
    show_request_metric(all_requests, "tpot", "Approx TPOT")

    round_throughputs = [r["throughput"] for r in rounds]
    print(
        f"  Throughput: mean={statistics.mean(round_throughputs):.2f} "
        f"tokens/s, median={statistics.median(round_throughputs):.2f} "
        f"tokens/s"
    )
    print(
        f"  Round wall: mean="
        f"{statistics.mean(r['wall'] for r in rounds):.3f}s"
    )
    print(
        f"  Request wall: mean="
        f"{statistics.mean(r['wall'] for r in all_requests):.3f}s, "
        f"p95={percentile([r['wall'] for r in all_requests], 0.95):.3f}s"
    )

    # 避免各档位之间完全没有间隔
    time.sleep(2)

