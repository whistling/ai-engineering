# Production Quantization — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> 量化格式不是通用选择 —— 它取决于硬件、推理引擎和工作负载。GGUF Q4_K_M 或 Q5_K_M 在 CPU 和边缘设备上占优，通过 llama.cpp 和 Ollama 交付。GPTQ 在需要在同一基模型上应用多 LoRA 时，在 vLLM 内部胜出。AWQ 配合 Marlin-AWQ 内核在 7B 级模型上可提供约 741 tok/s 的吞吐量，并在 INT4 中获得最佳 Pass@1 —— 2026 年数据中心生产默认。FP8 在 Hopper、Ada 和 Blackwell 上保持中间立场 —— 接近无损且被广泛支持。NVFP4 与 MXFP4（Blackwell 微缩）属于进取型，需要逐块验证。有两个陷阱会伤到团队：校准数据集必须与部署域匹配，以及 KV 缓存与权重量化是分开的 —— AWQ 那句“我的模型现在是 4 GB”往往忘记了在生产批次规模下还有 10–30 GB 的 KV 缓存。

**Type:** 学习
**Languages:** Python (stdlib，跨格式的内存与吞吐量小型对比)
**Prerequisites:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 分钟

## 学习目标

- 说出截至 2026 年的六种生产级量化格式及其适用场景。
- 根据硬件（CPU vs GPU，Hopper vs Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（常规聊天、推理、多-LoRA）选择合适格式。
- 计算在所选格式下权重节省的内存以及不受影响的 KV 缓存占用。
- 指出会使量化模型在真实域流量上退化的校准数据集陷阱。

## 问题

量化降低了权重内存与 HBM 带宽消耗，这正是解码所需的。一份 FP16 的 70B 模型权重约为 140 GB。将权重量化为 INT4（AWQ 或 GPTQ）后模型变为约 35 GB —— 可以装进一块 H100 并为 KV 缓存留出空间；这很重要，因为在 128 并发序列、2k 上下文下，KV 缓存本身就需要 20–30 GB。

但量化并非没有代价。激进量化会退化质量，尤其是在以推理为主的任务上。不同格式适配不同引擎，硬件对不同精度的原生支持也不同。2026 年的格式花样是真实存在的，你不能盲搬别人的选择 —— 必须根据你的技术栈做决定。

## 概念

### 六种格式

| Format | Bits | 适用场景 | 引擎 |
|--------|------|---------|------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM 上的多-LoRA 场景 | vLLM、TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM (Marlin-AWQ)、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### GGUF — CPU/边缘默认

GGUF 是一种文件格式，而不是严格意义上的量化算法 —— 它在一个容器内打包了多种 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）。Q4_K_M 与 Q5_K_M 是生产默认 —— 在 4–5 位下接近 BF16 的质量。对于 CPU 或边缘推理是最佳选择，因为 llama.cpp 是迄今最快的 CPU 推理引擎。

在 vLLM 中的吞吐率惩罚：7B 约 ~93 tok/s —— 该格式并未为 GPU 内核优化。当部署目标是 CPU/边缘时使用 GGUF；否则不推荐。

### GPTQ — vLLM 中的多-LoRA

GPTQ 是一种带校准步骤的训练后量化算法。Marlin 内核让它在 GPU 上变得更快（相比非 Marlin GPTQ 有 2.6 倍加速）。7B 约 ~712 tok/s。

独特优势：GPTQ-Int4 支持在 vLLM 中使用 LoRA 适配器。如果你要同时服务基模型加 10–50 个微调变体（每个以 LoRA 形式），GPTQ 是你的路线。截止 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ — 数据中心 GPU 默认

Activation-aware Weight Quantization（激活感知权重量化）。在量化过程中保护约 1% 最关键的权重。Marlin-AWQ 内核：相比朴素实现有 10.9 倍速提升。7B 约 ~741 tok/s，在 INT4 格式中 Pass@1 最佳。

对于新的 GPU 服务，如果不需要多-LoRA（GPTQ）或 Blackwell 的激进 FP4（NVFP4），优先选择 AWQ。

### FP8 — 稳妥的中间选项

8 位浮点。接近无损。被广泛支持。Hopper 的 Tensor Core 对 FP8 提供原生加速，Blackwell 也继承该支持。FP8 在质量不可妥协的场景（推理、医疗、代码生成）是 2026 年的安全默认。相较于 INT4 的内存节省减半，但质量风险也远小得多。

### MXFP4 / NVFP4 — Blackwell 的进取选项

Microscaling FP4。每个权重块都有自己的缩放系数。激进但在 Blackwell 的 Tensor Core 上有硬件加速。相比 FP8，每 token 字节数减半 —— 在 Phase 17 · 07 中体现出经济优势。

注意事项：
- 截至 2026 年初还不支持 LoRA。
- 在以推理为主的工作负载上质量下降可见。
- 必须对每个模型在你的评估集上逐块验证。

### 校准陷阱

AWQ 与 GPTQ 需要校准数据集 —— 通常使用 C4 或 WikiText。对于领域模型（代码、医疗、法律），用通用网页文本进行校准会让算法在决定哪些权重需要保护时做出错误判断。HumanEval 的 Pass@1 可能下降几分。

修复办法：使用域内数据进行校准。数百条域内样本通常足够。上线前在评估集上测试。

### KV 缓存陷阱

AWQ 将权重压缩到 4 位，但 KV 缓存是独立的，通常仍为 FP16/FP8。对于 70B 模型使用 AWQ 时：

- 权重：~35 GB（从 140 GB 的 INT4）
- KV 缓存在 128 并发 × 2k 上下文下：~20 GB
- 激活：~5 GB
- 总计：~60 GB —— 可容纳于 H100 80GB 上

天真地认为“我把模型量化到 4 GB 了”会忘记另外 30–50 GB 的开销。HBM 预算需要整体考虑。

另外，KV 缓存的量化（FP8 KV 或 INT8 KV）是另一个独立选择，有其自身权衡 —— 这将直接影响注意力计算的精确性，并不是免费收益。

### AWQ INT4 对推理很危险

思维链、数学、长上下文的代码生成 —— 在激进量化下会明显受损。AWQ INT4 在 MATH 上会损失约 3–5 分。对于以推理为主的工作负载，交付 FP8 或 BF16；接受内存成本。

### 2026 年的选择指南

- CPU/边缘部署：GGUF Q4_K_M。就是它。
- GPU 部署、常规聊天、无需 LoRA：AWQ。
- GPU 部署、需要多-LoRA：使用带 Marlin 的 GPTQ。
- 推理型工作负载：FP8。
- Blackwell 数据中心、经验证的质量：NVFP4 + FP8 KV。
- 模糊情况：对每个候选格式运行 1,000 条样本的评估。

```figure
gpu-memory-breakdown
```

## 使用方法

`code/main.py` 会计算一系列模型规模下的内存占用（权重 + KV + 激活）以及六种格式的相对吞吐率。展示了何处 KV 缓存占主导、何处权重压缩有价值、以及何处 FP8 是稳妥选择。

## 交付成果

本课程会生成 `outputs/skill-quantization-picker.md`。给定硬件、模型规模、工作负载类型和质量容忍度，选择格式并生成校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于 70B 模型、128 并发、2k 上下文，计算每种格式的总 HBM。哪种格式让你能装上一块 H100 80GB？
2. 你有一个 7B 的代码模型。选一个格式并给出理由。如果你对质量容忍度估错，恢复路径是什么？
3. 计算为医疗领域模型校准 AWQ 所需的校准数据集规模。为什么更多数据并不总是更好？
4. 阅读 Marlin-AWQ 的论文或发布说明。用三句话解释为什么 AWQ 在 7B 上能达到 741 tok/s，而原生 GPTQ 约为 712。
5. 在什么情况下将 AWQ 权重与 FP8 KV 缓存结合比保持 KV 为 BF16 更有意义？

## 术语表

| Term | 大家怎么说 | 实际含义 |
|------|------------|---------|
| GGUF | “llama.cpp 格式” | 将多种 K-quant 变体打包的文件格式；CPU/边缘默认 |
| Q4_K_M | “Q4 K M” | 4-bit 的 K-quant 中等档；生产级 GGUF 默认 |
| GPTQ | “gee pee tee q” | 带校准的训练后 INT4 量化；在 vLLM 中支持 LoRA |
| AWQ | “a w q” | 激活感知的 INT4；Marlin 内核；在 INT4 中 Pass@1 最佳 |
| Marlin kernels | “快速 INT4 内核” | 为 Hopper 上的 INT4 定制的 CUDA 内核；有 ~10 倍速提升 |
| FP8 | “8 位浮点” | Hopper/Ada/Blackwell 上的安全默认精度 |
| MXFP4 / NVFP4 | “微缩四位” | Blackwell 上带按块缩放因子的 4 位 FP |
| Calibration dataset | “校准数据” | 用于选择量化参数的输入文本；必须与域匹配 |
| KV cache quantization | “KV INT8” | 与权重分离的选择；会影响注意力精度 |

## 深入阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 对比基准。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 各格式的吞吐率数据。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 格式逐项比较与选型建议。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式与参数。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — AWQ 原始论文。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — GPTQ 原始论文。