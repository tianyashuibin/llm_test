"""BGE-M3 /pooling closed-loop load test; Python standard library only."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
import os
from pathlib import Path
import queue
import random
import statistics
import time
from urllib.parse import urlsplit


WORDS = """the a is fast small large search model text query document vector sparse
dense find explain how why what data cloud server train learn store index
system book world hello good new old science music travel food city water
green blue light night day run test build read write useful simple english""".split()


def random_text(rng, min_chars, max_chars):
    target = rng.randint(min_chars, max_chars)
    words = []
    while True:
        remaining = target - len(" ".join(words)) - bool(words)
        # Avoid leaving one character, which cannot fit a space plus a word.
        candidates = [word for word in WORDS
                      if len(word) <= remaining and remaining - len(word) != 1]
        if not candidates:
            break
        words.append(rng.choice(candidates))
    text = " ".join(words)
    assert len(text) == target
    return text


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * p
    lo = int(index)
    hi = math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def validate_response(body):
    # IOProcessorResponse wraps the plugin's own response in its data field.
    data = body["data"]
    entries = data["data"] if isinstance(data, dict) else data
    if len(entries) != 1:
        raise ValueError("expected one embedding result")
    entry = entries[0]
    if len(entry["dense_embedding"]) != 1024:
        raise ValueError("expected 1024-dimensional dense embedding")
    sparse = entry["sparse_embedding"]
    if not isinstance(sparse, list) or not sparse:
        raise ValueError("missing sparse embedding")
    for item in sparse:
        if not isinstance(item["token_id"], int) or not isinstance(item["weight"], (int, float)):
            raise ValueError("invalid sparse token ID or weight")


def run_load(args, payloads, concurrency):
    jobs = queue.Queue()
    for payload in payloads:
        jobs.put(payload)
    url = urlsplit(args.base_url)
    path = url.path.rstrip("/") + "/pooling"
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]

    def worker():
        connection = None
        results = []
        try:
            while True:
                try:
                    payload = jobs.get_nowait()
                except queue.Empty:
                    break
                start = time.perf_counter()
                error = None
                try:
                    if connection is None:
                        cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
                        connection = cls(url.hostname, url.port, timeout=args.timeout)
                    connection.request("POST", path, body=payload, headers=headers)
                    response = connection.getresponse()
                    raw = response.read()
                    if response.status != 200:
                        raise ValueError(f"HTTP {response.status}: {raw[:200].decode(errors='replace')}")
                    validate_response(json.loads(raw))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if connection is not None:
                        connection.close()
                        connection = None
                results.append((time.perf_counter() - start, error))
        finally:
            if connection is not None:
                connection.close()
        return results

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        results = [item for future in futures for item in future.result()]
    elapsed = time.perf_counter() - start
    latencies = [latency * 1000 for latency, error in results if error is None]
    errors = Counter(error for _, error in results if error is not None)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "success": len(latencies),
        "failed": len(results) - len(latencies),
        "error_rate": (len(results) - len(latencies)) / len(results),
        "elapsed_s": elapsed,
        "success_rps": len(latencies) / elapsed,
        "attempt_rps": len(results) / elapsed,
        "mean_ms": statistics.mean(latencies) if latencies else None,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "errors": dict(errors),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--requests", type=int, default=1000, help="requests per concurrency level")
    parser.add_argument("--warmup", type=int, default=20, help="warmup requests per level")
    parser.add_argument("--min-chars", type=int, default=400, help="minimum input characters (default: 400)")
    parser.add_argument("--max-chars", type=int, default=500, help="maximum input characters, not tokens (default: 500)")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="bge_m3_benchmark.json")
    args = parser.parse_args()
    if (args.requests < 1 or args.warmup < 0 or args.min_chars < 1
            or args.max_chars < args.min_chars
            or args.timeout <= 0 or any(c < 1 for c in args.concurrency)):
        parser.error("require requests/timeout/concurrency > 0, warmup >= 0, and 1 <= min-chars <= max-chars")
    if urlsplit(args.base_url).scheme not in ("http", "https") or not urlsplit(args.base_url).hostname:
        parser.error("base-url must be an HTTP(S) server URL without /v1")
    rng = random.Random(args.seed)
    texts = [random_text(rng, args.min_chars, args.max_chars) for _ in range(args.requests + args.warmup)]
    lengths = [len(text) for text in texts[args.warmup:]]
    input_stats = {"min_chars": min(lengths), "max_chars": max(lengths),
                   "mean_chars": statistics.mean(lengths)}
    payloads = [json.dumps({"model": args.model, "task": "plugin", "data": {
        "input": text, "return_tokens": False,
    }}).encode() for text in texts]
    print(f"Input: random English words, {args.min_chars}–{args.max_chars} characters (not tokens); one text/request.")
    print(f"Measured input lengths: min={min(lengths)}, max={max(lengths)}, mean={statistics.mean(lengths):.2f}")
    print(f"Sample ({len(texts[0])} chars): {texts[0]}")
    print("Closed-loop concurrency; no retries. Latency includes HTTP, JSON parsing and output validation.")
    print("Latency percentiles cover successful requests only; warmup is excluded.")
    summaries = []
    for concurrency in args.concurrency:
        print(f"\nConcurrency={concurrency}, warming up...", flush=True)
        if args.warmup:
            warmup = run_load(args, payloads[:args.warmup], concurrency)
            if warmup["failed"]:
                raise SystemExit(f"Warmup failed; check endpoint/plugin: {warmup['errors']}")
        result = run_load(args, payloads[args.warmup:], concurrency)
        summaries.append(result)
        print(f"Success={result['success']}/{result['requests']}, error={result['error_rate']:.2%}, "
              f"RPS={result['success_rps']:.2f}, elapsed={result['elapsed_s']:.2f}s")
        for name in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
            value = result[name]
            print(f"  {name}: {value:.2f}" if value is not None else f"  {name}: N/A")
        for error, count in list(result["errors"].items())[:5]:
            print(f"  Error x{count}: {error}")
        Path(args.output).write_text(json.dumps({"config": vars(args), "input_stats": input_stats,
                                                "results": summaries}, indent=2) + "\n")
    print(f"\nResults saved to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
