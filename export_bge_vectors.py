"""Export keywords to BGE-M3 dense + sparse vectors via vLLM /pooling (stdlib only)."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit


def iter_keywords(source):
    first = True
    for line_number, line in enumerate(source, 1):
        word = line.strip()
        if not word:
            continue
        if first:
            first = False
            if word.casefold() == "keyword":
                continue
        yield line_number, word


def finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def extract_vectors(body):
    data = body["data"]
    entries = data["data"] if isinstance(data, dict) else data
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("expected one embedding result")
    dense = entries[0]["dense_embedding"]
    sparse = entries[0]["sparse_embedding"]
    if not isinstance(dense, list) or len(dense) != 1024 or not all(map(finite_number, dense)):
        raise ValueError("expected 1024 finite dense values")
    if not isinstance(sparse, list):
        raise ValueError("expected sparse token/weight list")
    for item in sparse:
        token_id = item["token_id"]
        if (type(token_id) is not int or token_id < 0 or not finite_number(item["weight"])):
            raise ValueError("invalid sparse token ID or weight")
    return dense, sparse


class EmbeddingClient:
    def __init__(self, args):
        self.args = args
        self.url = urlsplit(args.base_url)
        self.path = self.url.path.rstrip("/") + "/pooling"
        self.local = threading.local()
        self.connections = []
        self.lock = threading.Lock()
        self.headers = {"Content-Type": "application/json"}
        if os.environ.get("VLLM_API_KEY"):
            self.headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]

    def convert(self, item):
        line_number, word = item
        payload = json.dumps({"model": self.args.model, "task": "plugin", "data": {
            "input": word, "return_tokens": False,
        }}, ensure_ascii=False).encode("utf-8")
        for attempt in range(self.args.retries + 1):
            connection = getattr(self.local, "connection", None)
            try:
                if connection is None:
                    cls = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
                    connection = cls(self.url.hostname, self.url.port, timeout=self.args.timeout)
                    self.local.connection = connection
                    with self.lock:
                        self.connections.append(connection)
                connection.request("POST", self.path, body=payload, headers=self.headers)
                response = connection.getresponse()
                raw = response.read()
                if response.status != 200:
                    raise ValueError(f"HTTP {response.status}: {raw[:300].decode(errors='replace')}")
                dense, sparse = extract_vectors(json.loads(raw))
                return {"line_number": line_number, "word": word,
                        "dense_embedding": dense, "sparse_embedding": sparse}, None
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if connection is not None:
                    connection.close()
                    with self.lock:
                        self.connections.remove(connection)
                    self.local.connection = None
                if attempt < self.args.retries:
                    time.sleep(min(0.5 * 2 ** attempt, 5))
        return None, {"line_number": line_number, "word": word,
                      "attempts": self.args.retries + 1, "error": error}

    def close(self):
        for connection in self.connections:
            connection.close()


def export(args, source, output, errors):
    client = EmbeddingClient(args)
    start = last_progress = time.monotonic()
    completed = failed = 0
    items = iter(iter_keywords(source))
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            pending = deque()
            # Bound in-flight work and preserve input order without retaining all vectors.
            for _ in range(args.concurrency * 2):
                item = next(items, None)
                if item is None:
                    break
                pending.append(executor.submit(client.convert, item))
            while pending:
                record, error = pending.popleft().result()
                completed += 1
                if error is None:
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                else:
                    failed += 1
                    errors.write(json.dumps(error, ensure_ascii=False) + "\n")
                    errors.flush()
                    print(f"Failed line {error['line_number']}: {error['error']}", flush=True)
                item = next(items, None)
                if item is not None:
                    pending.append(executor.submit(client.convert, item))
                now = time.monotonic()
                if now - last_progress >= 5:
                    output.flush()
                    print(f"Processed={completed}, success={completed - failed}, failed={failed}, "
                          f"elapsed={now - start:.1f}s, success_rps={(completed - failed)/(now - start):.2f}", flush=True)
                    last_progress = now
    finally:
        client.close()
        output.flush()
        errors.flush()
    print(f"Finished: processed={completed}, success={completed - failed}, failed={failed}, "
          f"elapsed={time.monotonic() - start:.2f}s", flush=True)
    return 1 if failed or not completed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSONL file; existing files are never overwritten")
    parser.add_argument("--errors-output", type=Path, help="default: OUTPUT.errors.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=2, help="additional attempts after a failure")
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("require concurrency/timeout > 0 and retries >= 0")
    url = urlsplit(args.base_url)
    if (url.scheme not in ("http", "https") or not url.hostname or url.query or url.fragment
            or url.path.rstrip("/").endswith("/v1")):
        parser.error("base-url must be an HTTP(S) server URL without /v1, query or fragment")
    args.input_file = args.input_file.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.errors_output = (args.errors_output or Path(str(args.output) + ".errors.jsonl")).expanduser().resolve()
    if len({args.input_file, args.output, args.errors_output}) != 3:
        parser.error("input, output and error file paths must differ")
    for path in (args.output, args.errors_output):
        if path.exists():
            parser.error(f"file already exists: {path}; choose a new output path")
    print(f"Input: {args.input_file}\nOutput: {args.output}\nErrors: {args.errors_output}", flush=True)
    print("One keyword/request; header and blank lines skipped; duplicates and input order preserved.", flush=True)
    try:
        with args.input_file.open(encoding="utf-8-sig") as source, \
                args.output.open("x", encoding="utf-8") as output, \
                args.errors_output.open("x", encoding="utf-8") as errors:
            status = export(args, source, output, errors)
    except (OSError, UnicodeError) as exc:
        raise SystemExit(str(exc))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
