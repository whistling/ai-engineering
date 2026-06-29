# 原生稀疏注意力（DeepSeek NSA）

> 在 64k tokens 时，注意力占了解码延迟的 70–80%。每个公开模型团队都有解决方案。DeepSeek 的 NSA（ACL 2025 最佳论文）是留存下来的方法：三条并行注意力分支——压缩的粗粒度 token、有选择保留的细粒度 token，以及用于局部上下文的滑动窗口——通过一个学习到的门组合在一起。它与硬件对齐（对 kernel 友好）、原生可训练（可用于预训练，而不是仅在推理时拼接），并且在 64k 解码上运行速度快于 FlashAttention，同时匹配或优于全注意力的质量。本课构建这三条分支的端到端实现并展示为什么稀疏性是端到端可微的。

**Type:** 构建  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 7 · 12 (KV cache, FlashAttention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)  
**Time:** ~60 分钟

## 学习目标

- 说明 NSA 的三条注意力分支以及每条分支捕获的内容。  
- 解释为什么 NSA 是“原生可训练”的，而之前的稀疏注意力方法通常仅用于推理。  
- 在 64k 上按压缩块大小与选择 top-k，计算 NSA 相对于全注意力的计算节省。  
- 在 stdlib Python 上实现三分支组合，针对短合成序列验证门权重的行为。

## 问题背景

序列长度为 N 时，全注意力的时间复杂度为 `O(N^2)`，每层的 KV 缓存为 `O(N)`。在 64k tokens 时，计算与内存带宽开销是灾难性的。NSA 论文的理论估计：在 64k 时注意力占总解码延迟的 70–80%。下游的一切指标——TTFT、tokens/sec、每百万 token 成本——都被注意力成本主导。

稀疏注意力是显而易见的答案。之前的尝试分为两类。固定模式稀疏（滑动窗口、步幅、块本地）丢弃信息，在长程回忆任务上失败。推理时稀疏（KV cache 剪枝、H2O、StreamingLLM）被应用到一个预训练时使用密集注意力的模型上，由于模型从未被要求通过稀疏模式路由信息，因此只能恢复一小部分潜在加速。

Native Sparse Attention（Yuan 等，DeepSeek + PKU + UW，ACL 2025 最佳论文，arXiv:2502.11089）两者兼顾：在预训练期间学习的稀疏模式，并以与硬件对齐的算法实现，在推理时实际带来计算节省。再过两年，NSA 或其直接后继很可能成为所有前沿长上下文模型的默认注意力机制。

## 概念

### 三条并行分支

对每个 query，NSA 对 KV 缓存的三种不同视图分别运行注意力：

1. **Compressed branch（压缩分支）**。将 token 分组为大小为 `l` 的块（通常为 32 或 64）。每个块通过一个小的学习型 MLP 压缩为一个 summary token。query 对这些压缩 token 做注意力，获得整个序列的粗粒度视图。

2. **Selected branch（选择分支）**。使用来自压缩分支的注意力分数，识别对当前 query 最相关的 top-k 块。读取这些块中的细粒度（未压缩）tokens，query 对它们做注意力。把压缩分支的注意力看作选择的路由信号。

3. **Sliding-window branch（滑动窗口分支）**。query 对最近的 `W` 个 tokens（通常为 512）进行注意力以获取局部上下文。该分支捕获结构性强的短程模式（语法、本地指代）——这类信息两条其他分支可能会遗漏。

三个分支的输出通过一个每位置学习的门组合：

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` 来源于对 query 的一个小 MLP。它们不必相加为 1 —— 可以独立地加权各分支。

### 为什么是“原生可训练”的

选择步骤（top-k 块）是离散的。离散操作会中断梯度流。此前的稀疏注意力工作要么跳过对选择的反向传播（限制了训练），要么使用连续松弛但在推理时不能带来真实稀疏性。

NSA 绕开了这个问题：压缩分支的注意力本身就是对整个序列的可微粗粒度注意力。top-k 操作只是重用压缩分支的注意力分数来决定要加载哪些细粒度块。梯度通过压缩分支的分数流动（这些分数既影响压缩输出也影响选择逻辑），被选择块对最终输出的贡献同样是可微的。非可微的 `top_k` 操作在前向计算图上只是控制哪些块从内存加载，不阻断参数的学习信号。

这就是 NSA 可以在预训练中端到端使用的原因。模型学会在三条分支间联合路由信息，产生一个在推理时真正带来速度提升的稀疏模式。

### 与硬件对齐的 kernel

NSA 的 kernel 针对现代 GPU 的内存层次结构设计。kernel 按 GQA 组加载 queries（外循环），为每组获取相应的稀疏 KV 块（内循环），并在 SRAM 上运行注意力。由于每个 query 组看到的被选块相同（选择是按 query 组，而不是按单个 query-head），KV 加载可以在组内摊销。算术强度保持较高。

论文报告的 Triton kernel 在 64k 解码上比 FlashAttention 快 9 倍，且随着序列长度增长，速度比进一步增大。论文同时给出了前向和反向 kernel。

### 计算预算

设序列长度为 `N`，压缩块大小为 `l`，top-k 选择数为 `k`，滑动窗口为 `w`，被选择的块大小为 `b`（通常等于 `l`）。

- 压缩分支：每个 query 对 `O(N/l)` 个键，所以总成本为 `O(N * N / l)`。  
- 选择分支：每个 query 对 `O(k * b)` 个键，所以总成本为 `O(N * k * b)`。  
- 滑动分支：每个 query 对 `O(w)` 个键，所以总成本为 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

当 `N = 64k, l = 64, k = 16, b = 64, w = 512` 时：每 query 成本为 `1000 + 1024 + 512 = 2536 keys`。全注意力是 `64000 keys`。约 25x 计算减少。

当 `N = 128k, l = 64, k = 16, b = 64, w = 512` 时：每 query 成本为 `2000 + 1024 + 512 = 3536 keys`。全注意力为 `128000 keys`。约 36x 减少。随着序列长度增长，收益越明显——这正是设计目标。

### 与其他方法比较

| Method | Differentiable | Real inference speedup | Long-range recall |
|--------|---------------|------------------------|-------------------|
| Sliding window only | 是 | 是 | 失败 |
| Strided / block-sparse | 是 | 是 | 部分 |
| KV pruning (H2O, StreamingLLM) | 不适用（推理时） | 是 | 部分 |
| MoBA (Moonshot) | 部分 | 是 | 好 |
| NSA | 是（原生） | 是（64k 时 9x） | 匹配全注意力 |

MoBA（Moonshot，arXiv:2502.13189）是同期发布的方法，采取了类似的“三比一更好”思路，将 MoE 原则应用于注意力块。NSA 和 MoBA 是 2026 年长上下文预训练需要了解的两种架构。

```figure
sliding-window-attention
```

## 构建实现

`code/main.py` 在一个短的合成序列上实现三条分支并演示：

- 压缩 MLP（为教学清晰，这里使用一个简单的均值池化基线；真实的 NSA 使用学习型 MLP）。  
- 由压缩分支分数驱动的 top-k 块选择。  
- 对最后 `w` 个 token 的滑动窗口注意力。  
- 门控组合。  
- 与全注意力对比的计算计数输出。

### 第 1 步：将 tokens 压缩为块

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### 第 2 步：压缩分支注意力

对查询与压缩后的 keys 运行 softmax 注意力。压缩分支的分数同时作为 top-k 选择的信号。

### 第 3 步：top-k 块选择

选择 `k` 个分数最高的压缩块的索引。加载这些块中原始的未压缩 tokens，并在它们上运行注意力。

### 第 4 步：滑动窗口注意力

取最近的 `w` 个 tokens，并对它们运行标准注意力。

### 第 5 步：门与组合

一个对 query 的小 MLP 产生三个门标量。最终输出是三条分支输出的加权和。

### 第 6 步：计算计数

打印每个 query 对各分支以及总共参照的键数。与 `N`（全注意力）比较。在一个 1024-token 的合成例子中，取 `l = 32, k = 4, w = 128`，NSA 每 query 看到 `32 + 128 + 128 = 288` 键，而全注意力为 1024 —— 约 3.5x 的减少。

## 使用场景

NSA 已在 DeepSeek 自身的长上下文预训练流水线中使用。截止 2026 年 4 月，公开推理栈的集成状态：

- **DeepSeek 内部**：原生支持，公开权重使用 NSA 或其后继 DSA（Deepseek Sparse Attention）。  
- **vLLM**：为 DeepSeek-V3.x 权重开发中的实验性 NSA 支持。  
- **SGLang**：发布了 NSA 基准；生产路径遵循 vLLM。  
- **llama.cpp / CPU**：不支持；kernel 分解的开销对 CPU 吞吐不划算。

何时使用 NSA：

- 目标是 64k 以上上下文且有较大计算预算的预训练或继续训练。  
- 对 DeepSeek 的长上下文 checkpoint 做推理，且权重为 NSA 原生。

何时不使用：

- 服务一个已有的密集注意力预训练模型。不能在不继续训练的情况下把 NSA 做为补丁接入。  
- 上下文长度低于 16k。三分支开销会超过节省。  
- Batch-1 的交互式聊天。虽然解码延迟敏感，但仅在长上下文下才有明显收益。

## 部署产出

本课会生成 `outputs/skill-nsa-integrator.md`。给定一个长上下文预训练运行规范，它会输出 NSA 集成计划：压缩块大小、top-k、滑动窗口、门 MLP 宽度、kernel 选择，以及能证明架构变更合理的具体长上下文评估。

## 练习

1. 在 1024-token 的合成上运行 `code/main.py`。对三个预设扫 `(l, k, w)` 并打印计算计数。找出在对针在草堆（needle-in-haystack）测试中保持 95% 回忆率的同时，实现最低每 query 键计数的预设。  

2. 用一个小的学习型 MLP（2 层，隐藏维度 32）替换均值池压缩器。在一个信号是块平均值的合成任务上训练它。对比在保留集上与均值池基线的困惑度差距。  

3. 实现门 MLP。它以 query 为输入并输出三个标量。证明门行为合理：随机 query 时接近均匀加权，当 query 命中远端块时对选择分支赋予很高权重。  

4. 计算在 128k 上启用 NSA 的一个 70B 模型的 KV 缓存内存预算。KV head 数为 8，head dim 为 128，BF16。与全注意力及 MLA（Phase 10 · 14 给出 MLA 的数据）比较。确定 NSA 的细粒度分支 KV 缓存与全注意力相等时的序列长度。  

5. 阅读 NSA 论文（arXiv:2502.11089）第 4 节，并用三句话解释为什么重用压缩分支的注意力分数作为 top-k 选择，而不是计算一个单独的路由分数。将答案与梯度流联系起来。

## 关键术语

| Term | 人们常说 | 实际含义 |
|------|---------|----------|
| Compressed branch | “粗粒度视图” | 对块平均的 keys 做注意力，提供全局上下文，每个 query 仅需 `O(N/l)` 个键 |
| Selected branch | “Top-k 块” | 对压缩分支分数最高的 `k` 个块内的未压缩 tokens 做细粒度注意力 |
| Sliding window | “局部上下文” | 对最后 `W` 个 tokens 做注意力以捕获短程模式 |
| Native trainability | “在预训练时开启稀疏” | 稀疏模式在预训练期间被学习，而不是在推理时拼接上去 |
| Compression block size l | “粗视图的分组大小” | 一个 summary 合并了多少 token；典型为 32–64 |
| Top-k | “保留的块数” | 要读取未压缩 token 的压缩块数量；典型为 16 |
| Sliding window W | “局部注意力半径” | 通常为 512；太短损害局部连贯性，太长浪费计算 |
| Branch gate | “如何混合三条分支” | 每位置的 MLP 输出，用于加权三条分支的贡献 |
| Hardware alignment | “对 kernel 友好的稀疏模式” | 选择的稀疏模式使得 GPU kernel 能够实现理论上的加速 |
| DSA | “NSA 的后继” | Deepseek Sparse Attention，NSA 在 DeepSeek 系列中的后续架构 |

## 延伸阅读

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — 论文原文  
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — NSA 所针对的架构家族  
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — 同期工作，基于块的 MoE 风格注意力  
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — 滑动窗口的起源  
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — 推理时稀疏的基线，NSA 在此基础上改进  
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — NSA 在 64k 上击败的全注意力基线