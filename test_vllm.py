from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

llm = LLM(
    model=model_name,
    dtype="half",
    max_model_len=1024,
    gpu_memory_utilization=0.7,
    enforce_eager=True,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
)

prompts = [
    "Explain GPU inference in one sentence.",
    "What is KV cache in a language model?",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print("=" * 60)
    print("Prompt:", output.prompt)
    print("Generated:", output.outputs[0].text)
