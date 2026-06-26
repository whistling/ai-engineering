# TensorRT-LLM on Blackwell with FP8 and NVFP4

> TensorRT-LLM 仅限 NVIDIA，但在 Blackwell 上具有压倒性的优势。在 GB200 NVL72 上与 Dynamo 编排结合使用时，SemiAnalysis InferenceX 在 2026 年 Q1–Q2 对一个 120B 模型测得 $0.012/百万标记，而在 H100 + vLLM 上约为 $0.09/M — 经济上相差 7 倍。这个堆栈由三种浮点表示叠加构成：FP8 对 KV cache 和 attention 内核仍然至关重要，因为它具有所需的动态范围；NVFP4（4 位微缩放）处理权重和激活；多标记预测（MTP）和预填充/解码的分离再额外带来 2–3x 的提升。Day-0 模型支持可直接加载 FP4 权重，无需训练后转换。对 2026 年的工程团队来说的关键权衡是：TRT-LLM 是一个封闭的 NVIDIA 堆栈，采用它意味着将可移植性换成吞吐量。在投入之前请对你的模型与硬件组合做精算。

**Type:** 学习  
**Languages:** Python（stdlib，玩具级 FP8/NVFP4 内存与成本计算器）  
**Prerequisites:** Phase 17 · 04（vLLM 服务内部），Phase 10 · 13（量化）  
**Time:** ~75 分钟

## 学习目标

- 解释为什么即便权重采用 NVFP4，KV cache 和 attention 仍然需要 FP8。  
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并推断节省来自哪里。  
- 列出 TRT-LLM 利用的 Blackwell 专有特性（day-0 FP4、MTP、分离式服务、all-to-all 原语）。  
- 判断在何种情况下 TRT-LLM 的 NVIDIA 锁定是值得的（相对于 Hopper 上的 vLLM 的 7 倍成本差异）。

## 问题背景

到 2026 年，推理经济学的前沿问题是“每美元能处理多少标记”。答案取决于四个叠加的选择：硬件世代（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、服务引擎（vLLM vs SGLang vs TRT-LLM）和编排方式（普通 vs 分离式 vs Dynamo）。

在 Hopper + vLLM 上，一个 120B MoE 的运行成本约为 $0.09/百万标记。在 Blackwell + TRT-LLM + Dynamo 上，同一模型约为 $0.012/M — 便宜 7 倍。部分差距来自硬件（Blackwell 在每卡 LLM 吞吐上比 Hopper 高 11–15x），部分来自软件栈：FP4 权重、MTP 草案、预填充/解码分离以及 MoE 专家通信的 NVLink 5 全互联支持。

在 NVIDIA 之外无法复制这样的堆栈。这就是权衡点——用可移植性换取经济性。本课程的意义在于理解哪些堆栈选择贡献了差距的哪一部分。

## 概念讲解

### 为什么 KV cache 仍以 FP8 为底线

2026 年常见的一个错误是假设 NVFP4 能无处可用。事实并非如此。KV cache 需要 FP8（8 位浮点）因为它存储的 attention key/value 覆盖很宽的动态范围。将 KV 量化到 FP4 会造成灾难性的精度损失——分布尾部被截断，attention 分数崩溃。FP8 的指数位为 KV cache 提供了必要的范围。

NVFP4（2025–2026）适用于权重和激活。微缩放（microscaling）：每个权重块有自己的缩放因子，使得小块可以覆盖不同的动态范围而不丢失每张量的尺度信息。对于激活，FP4 可以胜任，因为激活在层内通常处于较小范围。

典型的 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。  
- 激活：NVFP4。  
- KV cache：FP8。  
- Attention 累加器：FP32（softmax 稳定性）。

### TRT-LLM 利用的 Blackwell 专有原语

- **Day-0 FP4 权重**：模型提供方直接发布 FP4 权重；TRT-LLM 可以直接加载，无需训练后转换。无需为 FP4 运行 AWQ / GPTQ。  
- **多标记预测（MTP）**：与 EAGLE（Phase 17 · 05）相同的思想，但集成在 TRT-LLM 构建中。  
- **分离式服务**：在不同的 GPU 池上分别处理 prefill 和 decode，KV cache 通过 NVLink 或 InfiniBand 传输。与 Dynamo（Phase 17 · 20）的思想一致。  
- **All-to-all 通信原语**：NVLink 5 将 MoE 专家通信的延迟比 Hopper 缩短 3 倍。TRT-LLM 的 MoE 内核为此进行了优化。  
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 上针对缩放因子的硬件加速支持。

### 你应该记住的数据

- 在 HGX B200 上，通过 TRT-LLM，GPT-OSS-120B 的成本约为 $0.02/M 标记。  
- 在 GB200 NVL72 上，通过 Dynamo（编排 TRT-LLM）约为 $0.012/M 标记。  
- H100 + vLLM 在可比工作负载上约为 $0.09/M 标记。  
- TRT-LLM 在 2026 年三个月更新中带来了 2.8x 的吞吐提升。  
- Blackwell 相比 Hopper 的每卡 LLM 吞吐提升 11–15x。  
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在提交的所有任务中占优。

### FP4 对质量的真实代价

NVFP4 的量化非常激进。在需要推理链（思维链）、数学、长上下文的代码生成等重推理工作负载上，FP4 权重会显著降低质量。每块的校准可以缓解但不能完全消除这一问题。提供推理模型的团队常常采用权衡方案：FP8 权重 + FP4 激活，或者直接在 H200 上全程使用 FP8。

规则：在将模型转换为 NVFP4 权重之前，务必在你的评估集上验证任务质量。

### 为什么这是一个 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 封闭内核的组合。模型需要为特定 GPU SKU 编译。没有 AMD、没有 Intel、没有 ARM。如果你的基础设施策略是多厂商混合，TRT-LLM 就不适合用于服务层——你仍然可以在混合硬件上使用 vLLM 进行部署。如果你的环境只使用 NVIDIA，那么这 7 倍的差距足以弥补锁定成本。

### 2026 年实操建议

对于年推理费用超千万美元（$100M+）的企业，在 Hopper + vLLM 上会损失 7–10 倍的节省机会。将成本占比高的工作负载迁移到 Blackwell + TRT-LLM + Dynamo。将实验/迭代层保留在 H100 + vLLM 以保持模型迭代速度。对每个 NVFP4 转换后的模型在生产前进行质量验证。

### 分离式服务的额外收益

TRT-LLM 的分离式服务（预填充与解码分开放置）在 Phase 17 · 20 中有深入讨论。在 Blackwell 上，这些加乘效应是：FP4 权重 × MTP 提速 × 分离式放置 × 缓存感知路由。7 倍的数字是假设采用了完整的栈。

```figure
pipeline-parallel
```

## 使用方法

`code/main.py` 会计算 HBM 占用、（在内存带宽受限时的）解码吞吐，以及在三种堆栈之间的 $/M-标记：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它以观察复合效应以及每项变化对差距的贡献份额。

## 交付物

本课件会生成 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型规模和年标记量，它会判断 Blackwell + TRT-LLM 堆栈是否值得为 NVIDIA 锁定付费。

## 练习

1. 运行 `code/main.py`。对于一个 120B MoE、30% 活跃参数，计算 H100 BF16、H100 FP8 以及 B200 NVFP4/FP8 在内存带宽限制下的解码吞吐。哪一步带来的跃升最大？  
2. 一个客户每年在 H100 + vLLM 上花费 $2M。若要在 12 个月内通过迁移到 TRT-LLM 收回成本，他们需要购买多少台 Blackwell GPU（按 7 倍经济差距估算）？  
3. 在将权重转换为 NVFP4 后，你在 MATH 测试上看到精度下降 3 个点。列出两条恢复路径：一条以质量为先（保留 FP8 权重），一条以成本为先（使用领域内数据进行校准）。  
4. 阅读 MLPerf v6.0 的推理结果。哪个任务的 Blackwell 对 Hopper 的优势最小，为什么？  
5. 计算在 128k 上下文下，一个 405B 模型采用 NVFP4 权重 + FP8 KV cache 所需的 HBM。它能否放入单个 GB200 NVL72 节点？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| FP8 | "eight-bit float" | 8 位浮点；用于 KV cache 和 attention，因其具有必要的动态范围 |
| NVFP4 | "four-bit micro" | NVIDIA 的 4 位微缩放浮点格式；在 Blackwell 上用于权重和激活 |
| MXFP8 | "MX eight" | 微缩放的 FP8 变体；在 Blackwell 的 Tensor Core 上有硬件加速 |
| Day-0 FP4 | "ship FP4 weights" | 模型提供方直接发布 FP4 权重；无需训练后转换步骤 |
| MTP | "multi-token prediction" | TRT-LLM 集成的投机性解码草案（Phase 17 · 05） |
| Disaggregated serving | "split prefill/decode" | 预填充和解码在不同的 GPU 池上进行；KV 通过 NVLink/IB 传输 |
| All-to-all | "MoE expert comm" | 将 token 路由到专家 GPU 的通信模式；NVLink 5 将延迟降低约 3 倍 |
| InferenceX | "SemiAnalysis inference bench" | 2026 年被业界接受的每标记成本基准测试 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果。  
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 的 all-to-all 与 MoE 内核优化。  
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档。  
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — 在 TRT-LLM 之上的分离式编排。  
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数字的基准套件。