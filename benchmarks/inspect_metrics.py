from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    dtype="half",
    max_model_len=1024,
    gpu_memory_utilization=0.7,
    enforce_eager=True,
)

params = SamplingParams(
    temperature=0.0,
    max_tokens=16,
    ignore_eos=True,
)

outputs = llm.generate(["Explain GPU inference."], params)
result = outputs[0]

print("\nRequestOutput type:", type(result))
print("RequestOutput fields:", getattr(result, "__dict__", "no __dict__"))

metrics = getattr(result, "metrics", None)
print("\nMetrics type:", type(metrics))
print("Metrics value:", metrics)
print("Metrics fields:", getattr(metrics, "__dict__", "no __dict__"))

if metrics is not None:
    for name in dir(metrics):
        if not name.startswith("_"):
            try:
                value = getattr(metrics, name)
                if not callable(value):
                    print(f"{name} = {value!r}")
            except Exception as exc:
                print(f"{name} = <error: {exc}>")
