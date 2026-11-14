"""Offline vLLM: warm up, then profile exactly one generate call."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--input-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/request.json"))
    args = parser.parse_args()
    if min(args.input_tokens, args.output_tokens, args.warmup) < 1:
        parser.error("token counts and warmup must be positive")
    if args.input_tokens + args.output_tokens > args.max_model_len:
        parser.error("input + output exceeds max-model-len")

    # Keep the GPU executor in this process so CUDA profiling APIs delimit
    # the actual worker's kernels, rather than only a frontend process.
    import os
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="half",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.cuda_graphs,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    tokenizer = llm.get_tokenizer()
    marker = "__QWEN_PROFILE_USER_CONTENT_7b49__"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or rendered.count(marker) != 1:
        raise ValueError("chat template must render user content exactly once")
    before, after = rendered.split(marker)
    prefix = tokenizer.encode(before, add_special_tokens=False)
    suffix = tokenizer.encode(after, add_special_tokens=False)
    content = tokenizer.encode(
        "请解释大语言模型推理中 prefill 和 decode 的区别。",
        add_special_tokens=False,
    )
    needed = args.input_tokens - len(prefix) - len(content) - len(suffix)
    if needed < 0:
        raise ValueError("input-tokens is too small for the chat template and question")
    # Explicit token segments preserve template delimiters and exact length.
    # This controlled workload is not a quality evaluation.
    filler = tokenizer.encode(" 推理", add_special_tokens=False)
    padding = (filler * ((needed + len(filler) - 1) // len(filler)))[:needed]
    prompt_ids = prefix + content + padding + suffix
    assert len(prompt_ids) == args.input_tokens
    params = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)
    prompt = {"prompt_token_ids": prompt_ids}
    for _ in range(args.warmup):
        llm.generate([prompt], params, use_tqdm=False)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("measured_single_request")
    torch.cuda.profiler.start()
    start = time.perf_counter()
    try:
        result = llm.generate([prompt], params, use_tqdm=False)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        torch.cuda.profiler.stop()
        torch.cuda.nvtx.range_pop()

    output = result.outputs[0]
    report = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "cuda_version": torch.version.cuda,
        "input_tokens": len(result.prompt_token_ids),
        "output_tokens": len(output.token_ids),
        "warmup_requests": args.warmup,
        "enforce_eager": not args.cuda_graphs,
        "prefix_caching": False,
        "chunked_prefill": False,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second_including_prefill": len(output.token_ids) / elapsed,
        "text": output.text,
        "note": "Profiler replay/tracing can distort timing; use baseline for latency. Forced output length is not a quality evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
