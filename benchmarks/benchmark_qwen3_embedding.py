"""Qwen3-Embedding-0.6B /v1/embeddings keyword-file closed-loop load test; Python standard library only."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
import http.client
import json
import math
import os
from pathlib import Path
import queue
import random
import statistics
import threading
import time
from urllib.parse import urlsplit


def load_texts(path):
    with path.open(encoding="utf-8-sig") as source:
        texts = [line.strip() for line in source if line.strip()]
    if texts and texts[0].casefold() == "keyword":
        texts.pop(0)
    if not texts:
        raise ValueError("input file contains no keywords")
    return texts


def make_payload(args, text):
    return json.dumps({"model": args.model, "input": text,
                       "encoding_format": "float"}, ensure_ascii=False).encode("utf-8")


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * p
    lo = int(index)
    hi = math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def validate_response(body):
    entries = body["data"]
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("expected one embedding result")
    entry = entries[0]
    if entry["index"] != 0:
        raise ValueError("expected embedding index 0")
    vector = entry["embedding"]
    if not isinstance(vector, list) or len(vector) != 1024:
        raise ValueError("expected 1024-dimensional embedding")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in vector):
        raise ValueError("embedding contains invalid or non-finite values")


def run_load(args, payloads, concurrency):
    jobs = queue.Queue()
    for payload in payloads:
        jobs.put(payload)
    url = urlsplit(args.base_url)
    prefix = url.path.rstrip("/")
    path = prefix + ("/embeddings" if prefix.endswith("/v1") else "/v1/embeddings")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
    progress_lock = threading.Lock()
    progress = {"completed": 0, "failed": 0, "last_error": None}

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
                with progress_lock:
                    progress["completed"] += 1
                    if error is not None:
                        progress["failed"] += 1
                        progress["last_error"] = error
        finally:
            if connection is not None:
                connection.close()
        return results

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        pending = set(futures)
        while pending:
            _, pending = wait(pending, timeout=5)
            if pending:
                with progress_lock:
                    snapshot = dict(progress)
                print(f"  Progress: completed={snapshot['completed']}/{len(payloads)}, "
                      f"failed={snapshot['failed']}, "
                      f"remaining={len(payloads) - snapshot['completed']}, "
                      f"elapsed={time.perf_counter() - start:.1f}s", flush=True)
                if snapshot["last_error"]:
                    print(f"  Last error: {snapshot['last_error']}", flush=True)
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
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--requests", type=int, default=1000, help="requests per concurrency level")
    parser.add_argument("--warmup", type=int, default=20, help="warmup requests per level")
    parser.add_argument("--input-file", type=Path, required=True,
                        help="UTF-8 file, one keyword per line; keyword header is skipped")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="qwen3_embedding_benchmark.json")
    args = parser.parse_args()
    if (args.requests < 1 or args.warmup < 0 or args.timeout <= 0
            or not math.isfinite(args.timeout)
            or any(c < 1 for c in args.concurrency)):
        parser.error("require requests/timeout/concurrency > 0 and warmup >= 0")
    url = urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.hostname or url.query or url.fragment:
        parser.error("base-url must be an HTTP(S) server URL without query or fragment")
    args.input_file = args.input_file.expanduser().resolve()
    try:
        corpus = load_texts(args.input_file)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    random.Random(args.seed).shuffle(corpus)
    texts = [corpus[i % len(corpus)] for i in range(args.requests + args.warmup)]
    lengths = [len(text) for text in texts[args.warmup:]]
    input_stats = {"file": str(args.input_file), "available_keywords": len(corpus),
                   "min_chars": min(lengths), "max_chars": max(lengths),
                   "mean_chars": statistics.mean(lengths)}
    payloads = [make_payload(args, text) for text in texts]
    print(f"Input: {args.input_file}; {len(corpus)} keywords; one keyword/request.")
    print(f"Endpoint: /v1/embeddings; model: {args.model}; seed: {args.seed}")
    print(f"Measured input lengths: min={min(lengths)}, max={max(lengths)}, mean={statistics.mean(lengths):.2f} chars")
    print(f"Sample: {texts[0]}")
    print("Each concurrency level uses the same inputs; inputs repeat after the corpus is exhausted.")
    print("Closed-loop concurrency; no retries. Latency includes HTTP, JSON parsing and output validation.")
    print("Latency percentiles cover successful requests only; warmup is excluded.")
    summaries = []
    for concurrency in args.concurrency:
        print(f"\nConcurrency={concurrency}, warming up...", flush=True)
        if args.warmup:
            warmup = run_load(args, payloads[:args.warmup], concurrency)
            if warmup["failed"]:
                raise SystemExit(f"Warmup failed; check endpoint/model: {warmup['errors']}")
            print(f"Warmup complete: {warmup['elapsed_s']:.2f}s", flush=True)
        print(f"Measuring {args.requests} requests...", flush=True)
        result = run_load(args, payloads[args.warmup:], concurrency)
        summaries.append(result)
        print(f"Success={result['success']}/{result['requests']}, error={result['error_rate']:.2%}, "
              f"RPS={result['success_rps']:.2f}, elapsed={result['elapsed_s']:.2f}s")
        for name in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
            value = result[name]
            print(f"  {name}: {value:.2f}" if value is not None else f"  {name}: N/A")
        for error, count in list(result["errors"].items())[:5]:
            print(f"  Error x{count}: {error}")
        Path(args.output).write_text(json.dumps({"config": {**vars(args), "input_file": str(args.input_file)}, "input_stats": input_stats,
                                                "results": summaries}, indent=2) + "\n")
    print(f"\nResults saved to {Path(args.output).resolve()}")
    if any(result["failed"] for result in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
