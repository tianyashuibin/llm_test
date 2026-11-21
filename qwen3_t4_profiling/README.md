# Qwen3-1.7B / T4 单请求推理剖析

目标是学习执行流程和定位热点：baseline → Nsight Systems → 选定 kernel → Nsight Compute → 优化前后对照。这里没有实测数据；报告由 T4 运行后生成。

## 环境与实验边界

- 单张 T4，FP16，Qwen/Qwen3-1.7B，vLLM 离线推理。
- 在 T4 上激活已能运行 vLLM 的虚拟环境；需要安装支持当前驱动、CUDA 和 Turing GPU 的 Nsight Systems、Nsight Compute。记录具体版本，不盲目升级现有环境。
- 首次运行需要模型已缓存或可访问 Hugging Face；`--model /absolute/model/path` 支持本地模型。
- 测量期间 GPU 应没有其他推理负载。vLLM 会按显存利用率预分配 KV cache，这不是请求实际用掉的全部显存。
- 默认输入 256 token、输出强制 32 token、预热 3 个请求，测量区间只包含一个请求。关闭思考，通过合法 chat template 构造固定长度 token 输入；填充文本和 `ignore_eos=True` 用于可控性能实验，不用于回答质量评价。
- 默认 eager，关闭 prefix cache、chunked prefill。设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 让 V1 GPU executor 在当前进程执行；具体版本应在 Systems 中确认采到了真实 GPU kernel。如果版本不支持这些选项，保留报错和版本信息再调整。

## 1. 检查环境与 baseline

在此目录运行：

```bash
bash run.sh env
bash run.sh baseline
```

可用 `PYTHON_BIN=/absolute/venv/bin/python bash run.sh baseline` 指定解释器。其他模式也支持该变量。

`results/baseline_<tag>.json`（默认 `results/baseline_single_request.json`）保存 GPU、软件版本、配置、实际 token 数、端到端耗时。单次测量只用于跑通；评价优化收益时需要重复 baseline、记录分布，并保持模型、输入/输出长度及 GPU 状态一致。这里的 tokens/s 包含 prefill，不是纯 decode 速度。

## 2. Nsight Systems：先看整个请求

```bash
bash run.sh systems
bash run.sh stats > results/systems_stats.txt
```

用 Nsight Systems GUI 打开 `results/single_request.nsys-rep`。采集由 CUDA profiler start/stop 限定，NVTX 的 `measured_single_request` 标识请求；NVTX range 在 capture 启动前开始，不同工具版本可能只显示其部分信息。以捕获区间内 CUDA 时间线为准。

量化版本的采集用 `TAG` 区分产物前缀，不需要改脚本：

```bash
TAG=w8a16 bash run.sh systems      # -> results/w8a16.nsys-rep, results/systems_w8a16.json
TAG=w8a8  bash run.sh systems      # -> results/w8a8.nsys-rep,  results/systems_w8a8.json
bash run.sh quant-systems          # 依次跑上面两个
bash run.sh quant-stats            # 依次出两份 stats
```

`TAG` 命中预设（`fp16` / `w8a16` / `w8a8`）时会自动补上对应的 `--model`；命令行显式传 `--model` 可覆盖。默认 `TAG=single_request`，即原来的文件名不变。

建议依次观察：

1. 请求开头的 prefill：处理整段输入，矩阵形状通常比 decode 大。
2. 随后的重复 decode：一次推进一个输出 token，观察 attention、矩阵乘、归一化、RoPE、采样等 kernel 的重复序列。不能仅凭 kernel 名认定所属阶段，需要看顺序、形状和调用上下文。
3. CPU launch 与 GPU 空隙：是否 GPU 在等 CPU，是否存在同步或拷贝。`cuda_api_sum` 中的同步耗时包含等待 GPU 的时间，不等同于 CPU 计算热点。
4. `cuda_gpu_kern_sum`：按累计 GPU 时间找总成本，结合次数与平均/最大耗时区分高频小 kernel 和少量长 kernel。汇总包含 prefill 与 decode，应回到时间线分别选实例。

这轮没有 CPU 采样，若要分析 Python/C++ CPU 热点，需要后续单独启用合适的 CPU 采样方案。

记录候选 kernel 的完整名称、阶段、同名实例序号、持续时间、grid/block 和累计占比，先选 1–3 个有代表性的热点，不直接采集全部 kernel 的全部指标。

## 3. Nsight Compute：检查选定 kernel

把 Systems 中观察到的名称转成尽量精确的正则表达式：

```bash
KERNEL_REGEX='你实际观察到的kernel名称正则' bash run.sh compute
```

默认只采一个匹配实例和 basic 指标。若同名 kernel 在 prefill/decode 都出现，用匹配实例跳过数选目标：

```bash
KERNEL_REGEX='实际名称正则' LAUNCH_SKIP=10 LAUNCH_COUNT=1 bash run.sh compute
```

`LAUNCH_SKIP` 应根据实际匹配序列确定，10 只是语法示例。检查 NCU 输出是否命中，确认 grid/block 与目标实例对应。新采集覆盖旧报告前，请先另存旧结果；工具默认可能拒绝覆盖。

确认目标后再扩大指标集：

```bash
KERNEL_REGEX='实际名称正则' NCU_SET=full bash run.sh compute
```

### 已测出的热点与默认目标

三个版本的 `cuda_gpu_kern_sum` 第一名（以及 W8A16 的第二名）已作为默认正则写进脚本，可直接运行：

```bash
TAG=fp16  bash run.sh compute    # gemvx::kernel.*__half
TAG=w8a16 bash run.sh compute    # marlin::Marlin<  +  gemvx::kernel.*__half
TAG=w8a8  bash run.sh compute    # cutlass::Kernel2<.*OpMultiplyAddSaturate
```

| 版本 | Kernel | 占比 | 实例 | Avg / Med |
| --- | --- | --- | --- | --- |
| FP16 | `gemvx::kernel<...__half...>` | 66.4% | 2636 | 128.8 μs / 68.0 μs |
| W8A16 | `marlin::Marlin<...>` | 51.6% | 3472 | 54.1 μs / 49.9 μs |
| W8A16 | `gemvx::kernel<...__half...>` | 21.2% | 32 | 2.410 ms / 2.409 ms |
| W8A8 | `cutlass::Kernel2<...OpMultiplyAddSaturate...>` | 62.0% | 3472 | 79.8 μs / 73.1 μs |

一个 tag 配多个目标时，NCU 逐个采集，报告写成 `results/kernel_<tag>_<slug>.ncu-rep`（如 `kernel_w8a16_marlin` / `kernel_w8a16_gemvx`）；只有一个目标时保持旧名 `results/kernel_<tag>.ncu-rep`。自定义多目标用换行分隔：

```bash
KERNEL_REGEX_W8A16=$'marlin::Marlin<\ngemvx::kernel.*__half' bash run.sh compute
```

W8A16 的第二名值得单独看：它是 32 次、每次约 2.41 ms 的 GEMV，正好一个输出 token 一次，而实例数只有 marlin 的零头却吃掉 21.2% 的时间。这与 `lm_head` 未被量化、回落到 cuBLAS 的行为一致（GPTQ 通常跳过 lm_head），即 decode 每步都要把 151936×2048 的半精度权重过一遍显存。若属实，优化方向不在 kernel 内部，而是让这批输出层权重也走 W8A16，或用 `--num-lookahead-slots`/采样侧手段减少其调用；先靠 NCU 的 launch 形状（M≈1、N≈vocab、K=hidden）确认，再看 DRAM 吞吐是否已接近 T4 的带宽峰值。

用 Nsight Compute GUI 打开 `results/kernel_<tag>.ncu-rep`，观察：

- Speed of Light / Roofline：计算吞吐和内存吞吐，区分计算、带宽或延迟限制。
- Memory Workload Analysis：DRAM/L2 流量、命中率和访存效率。
- Occupancy / Launch Statistics：寄存器、共享内存、block 大小和 SM 利用情况；高 occupancy 本身不是优化目标。
- Scheduler / Warp State：eligible warps 和 stall 原因，结合吞吐指标解释，不能仅凭一个 stall 排名下结论。
- Source：只有存在源码映射/line info 时才能关联 CUDA/Triton 源码；否则先从反汇编和 launch 指标分析。

各指标/section 的可用性依 T4 和 Nsight 版本而异。NCU replay 会显著改变执行时间、缓存状态和调度，不用 `compute.json` 判断端到端性能，也不与 baseline 的耗时直接比较。

若出现 `ERR_NVGPUCTRPERM`，GPU 性能计数器权限需要机器管理员按 NVIDIA 指引配置；这是环境权限问题，不是模型问题。不要自动修改驱动配置。

## 4. 第一项优化对照：CUDA Graph

先另存 eager 的报告，再运行：

```bash
bash run.sh baseline --cuda-graphs
bash run.sh systems --cuda-graphs
```

检查 CPU launch 开销与 decode 间隙是否减少、baseline 延迟是否改善。Graph 捕获发生在初始化/预热阶段，不在请求采集区间内。Graph 模式的 kernel 精细采集可能需要版本相关的 NCU graph 选项；先用 eager 模式学习 kernel，再用 Systems 验证 Graph 的系统收益。

后续每次只改变一个变量：输入长度、输出长度、attention 后端、量化方案等。T4 的后端和量化支持需按当前 vLLM 版本验证，不能直接套用其他 GPU 的配置。

## 分析记录模板

| 项目 | 记录 |
|---|---|
| GPU / 驱动 / CUDA / vLLM / Nsight 版本 | |
| 模型 / dtype / 输入输出 token / 执行配置 | |
| 无 profiler 的请求延迟分布 | |
| prefill / decode 时间与 GPU 空隙 | |
| 热点 kernel 名称 / 阶段 / 实例 / 占比 | |
| NCU 关键指标与瓶颈证据 | |
| 优化假设与单一配置变更 | |
| 优化后的 baseline 与时间线变化 | |
