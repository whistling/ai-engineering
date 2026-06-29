# Jamba — Hybrid SSM-Transformer

> 状态空间模型（SSM）与 Transformer 的诉求不同。Transformer 通过注意力在质量上获益，但代价是二次方复杂度。SSM 通过递归实现线性时间推理和常数内存，但质量滞后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）把它们放到同一模型中：每 7 层 Mamba 配 1 层 Transformer，每隔一层使用 MoE，并实现单卡 80GB 上可运行的 256k 上下文窗口。Mamba-3（ICLR 2026）在 SSM 方面通过复数值状态与 MIMO 投影收紧了设计。本文从头到尾阅读三篇论文，并解释为何这种混合配方在三年的扩展中持续成功，而纯 SSM 或纯 Transformer 的长上下文尝试未能普遍奏效。

**Type:** 学习  
**Languages:** Python（标准库，layer-mix 计算器）  
**Prerequisites:** Phase 10 · 14（开放模型架构）、Phase 10 · 17（原生稀疏注意）  
**Time:** ~60 分钟

## 学习目标

- 解释 Jamba 块中的三种原语——Transformer 层、Mamba 层、MoE——以及 1:7:even 的交错配方。
- 概述 SSM 的递归在高层次上是什么样子以及为何它能实现常数内存的推理。
- 计算在 256k 上下文下 Jamba 模型的 KV 缓存占用，并与纯 Transformer 模型进行比较。
- 列出 Mamba-3 的三项创新（指数-梯形离散化、复数值状态更新、MIMO）以及每项针对的问题。

## 问题描述

注意力的复杂度随序列长度呈二次增长。状态空间模型是线性的。这个差异会放大：在 256k token 下，Transformer 的注意力图对每个 head 有 650 亿个条目；而 SSM 的递归状态是固定大小，与序列长度无关。

纯 SSM 模型（Mamba、Mamba-2）在小规模时可以匹配 Transformer 的困惑度，但在状态追踪任务上落后，并在某些类型的上下文内检索任务上失败。直观上：SSM 将历史压缩进固定状态，当历史很长时信息会泄漏。注意力能精确记住所有内容，但要付出二次代价。

显而易见的修复办法：同时使用两者。在需要精确回忆的地方放 Transformer 层，在其他地方使用 SSM 层。调整比例。Jamba 是第一个在生产级别将这种混合配方规模化部署的模型（52B 总参数，12B 活跃参数，256k 上下文，单卡 80GB）。Jamba 1.5 将家族扩展到 398B 总 / 94B 活跃。Mamba-3（ICLR 2026）收紧了纯 SSM 端，是当前可供混合重建的最优纯 SSM 基线。

本课将阅读这三篇论文并构建“选择正确比例”的心智模型。

## 概念

### 一页看懂 SSM

状态空间模型通过固定大小的状态 h 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

在每一步，状态通过线性动力学 A 演化，接受输入 B x_t，并输出 C h_t。A、B、C 可学习。注意关键属性：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要更早的 `x`。内存是常数的。推理每个 token 的复杂度是 O(1)。

提高建模质量的技巧在于 A 的结构。S4（Gu 2021）采用高度结构化的矩阵，可以在训练时高效地作为长卷积来评估。Mamba（Gu、Dao 2023）将固定的 A、B、C 替换为数据相关的参数（即“选择性”部分）。Mamba-2（2024）进一步简化了结构。Mamba-3（2026）在特定位置再次增加了复杂性。

关键属性：对于解码器 LLM 来说，SSM 层可以作为注意力层的替代，使用固定大小的每层状态来替代随序列增长的 KV 缓存。

### Jamba 块

Jamba 块根据两个数交错层次：

- `l`：注意力与 Mamba 的比例。Jamba 使用 `l = 8`，意味着每 7 层 Mamba 配 1 层 Transformer（7 Mamba + 1 Attention = 8 层为一组）。
- `e`：MoE 的频率。Jamba 使用 `e = 2`，意味着每隔一层应用 MoE。

块内的层序列：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个 Jamba 块有 8 层。若堆叠 4 个块（总共 32 层），则得到 28 层 Mamba 和 4 层 Attention，其中 16 层使用 MoE。

### 为何选择 1:7 比例

AI21 做了消融实验：哪种注意力与 Mamba 比例能在困惑度-参数比与长上下文评估的上下文检索上取得最佳平衡？

- 注意力太多（1:1）：质量上升但内存与速度下降。
- 注意力太少（1:15）：内存优异但上下文内检索失败。
- 最佳点：1:7 或 1:8。

直观上：Transformer 层负责精确回忆和状态追踪，Mamba 层负责廉价的大量处理。

### 位置编码

Mamba 层自身通过递归对位置敏感。基于 Mamba 的混合模型中最初的 Attention 层没有使用 RoPE——SSM 层提供了位置信息。Jamba 1.5 为 Attention 层添加了 RoPE，以改善更长上下文的泛化，这是基于经验的长上下文评估后的事后微调。

### 内存预算

对于一个 Jamba-1 结构（32 层：28 Mamba + 4 Attention，hidden=4096，32 个 attention heads）：

- KV 缓存（仅 Attention 层）：`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB` 在 256k BF16 下。只有 4 层 Attention 会贡献。
- SSM 状态：`28 * hidden * state_size` 为前缀的固定大小，但不随序列长度增长。典型 Mamba 状态为每特征 16：`28 * 4096 * 16 * 2 = 3.7 MB` 总计。

与纯 Transformer（32 层、相同 hidden、全 MHA、32 heads）比较：`2 * 32 * 32 * 128 * 256k * 2 = 128 GB`（在 256k BF16 下）。KV 缓存减少约 8 倍。即便与 2024 年多数模型采用的 GQA(8) 基线比较（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 混合在 16 GB 时仍小 2 倍。

这就是 AI21 所说的“在单个 80GB GPU 上实现 256k 上下文”的原因。全 MHA 的纯 Transformer 的 KV 缓存无法装下；即使是 GQA 基线也没有为权重和激活留出空间；Jamba 的则有。

### Mamba-3：2026 年的纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 端引入三项创新：

1. **指数-梯形离散化（Exponential-trapezoidal discretization）。** 用更具表现力的递归取代了 Mamba-2 中的 Euler 方法离散化。卷积类操作在核心递归内施加于状态-输入，而不是作为对 `x_t` 的外部卷积。

2. **复数值状态更新（Complex-valued state update）。** 之前的 Mamba 将状态矩阵从复数（S4）降为实对角（Mamba），再到缩放的恒等（Mamba-2）。Mamba-3 恢复了复数值——等价于对状态施加数据相关的旋转嵌入（rotary embedding）。这恢复了前几代实值简化所损失的状态追踪能力。

3. **多输入多输出投影（MIMO）。** 用矩阵值投影替代按特征的标量投影。提高了建模能力和推理时的硬件利用率，同时不增加解码延迟。

在 1.5B 参数规模下，Mamba-3 在下游平均准确率上比 Gated DeltaNet 提高 0.6 个点；MIMO 变体额外再提高 1.2 个点，总计 1.8 个点。在相同状态大小下，Mamba-3 以一半的状态匹配 Mamba-2 的表现。

Mamba-3 尚未在生产混合模型中大规模部署——但它是下一代 Jamba 级模型中 SSM 端的显然候选者。

### 何时选择混合架构

混合架构适合当：

- 上下文足够长，以至于纯 Transformer 的 KV 缓存变得难以承受（64k+）。
- 任务混合了短程结构（适合 SSM）和长程回忆（需要 Transformer）。
- 你希望在单卡内存预算下部署，而 Transformer 的 KV 缓存本身就放不下。

混合架构不适合当：

- 上下文较短（低于 16k）。SSM 的开销被浪费；纯 Transformer 足够好。
- 任务需要 everywhere-to-everywhere 的注意力（深度推理、多文档交叉引用）。混合中稀疏的注意力层会受限。
- 你在向万亿参数前沿扩展。纯 Transformer + MLA + MoE（如 DeepSeek-V3 风格）目前在能力竞赛中占优。

### 竞争格局

| Model | Family | Scale | Unique claim |
|-------|--------|------|-------------|
| Mamba-2 | pure SSM | 3B | 线性时间、常数内存 |
| Jamba | hybrid | 52B/12B | 80GB 单卡支持 256k |
| Jamba 1.5 Large | hybrid | 398B/94B | 企业级长上下文 |
| Mamba-3 | pure SSM | 1.5B (paper) | 恢复的状态追踪 |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | 前沿能力 |

2026 年的格局：纯 Transformer + MoE 在前沿能力上占优，但混合架构占据了 256k+ 上下文的利基。Mamba-3 的状态追踪胜出可能会在下一代中推动混合比例向更低注意力（更多 SSM）倾斜。

```figure
swiglu-ffn
```

## 使用方法

`code/main.py` 是一个混合架构的内存计算器。给定 SSM-Transformer 比例以及 hidden-size / layer-count 配置，它会计算：

- 目标上下文下的 KV 缓存。
- SSM 状态内存。
- 在一系列模型形状下，序列长度 N 时的总内存。

计算器支持：

- 纯 Transformer 基线（KV 缓存随 N 增长）。
- Jamba 风格的 1:7 混合。
- 纯 SSM（根本没有 KV 缓存）。

数值直接来源于 Jamba-1 与 Jamba-1.5 论文的已发表形状，并对假设变体进行了外推。

真实部署的集成注意事项：

- 大多数生产推理服务器（如 vLLM、SGLang）支持 Jamba 与 Mamba。请检查具体版本。
- 在 256k 上下文下，Jamba 的内存优势会体现在并发请求吞吐量上。在相同 VRAM 下，你能跑更多的 Jamba 序列而非 Transformer 序列。
- Mamba-3 作为独立模型尚未在生产中部署——在 1.5B 规模上仍是研究预览。

## 投产建议

本课生成 `outputs/skill-hybrid-picker.md`。给定工作负载规格（上下文长度分布、任务混合、内存预算），它会在纯 Transformer、Jamba 风格混合与纯 SSM 之间进行推荐，并就内存与质量的权衡给出明确理由。

## 练习

1. 运行 `code/main.py`，计算在 256k 上下文下，32 层纯 Transformer（hidden 4096，32 heads）与同样形状的 Jamba-1 混合的 KV 缓存。验证 AI21 论文声称的大约 8 倍内存减少。

2. 修改计算器以模拟 1:3 混合（4 Mamba : 1 Attention）和 1:15 混合（14 Mamba : 1 Attention）。绘制 KV 缓存相对于比例的曲线。在何种比例下 KV 缓存等于 SSM 状态内存？

3. 阅读 Jamba 论文第 3 节（arXiv:2403.19887）。解释 AI21 为何更倾向使用 Mamba-1 而非 Mamba-2，尽管 Mamba-2 更快。提示：混合消融章节记录了这一点。

4. 计算 Jamba 1.5 Large（398B 总，94B 活跃）中每隔一层使用 MoE 的参数开销。将活跃参数比与 DeepSeek-V3（37B/671B）比较，并解释为何 Jamba 的架构提升了活跃比。

5. 阅读 Mamba-3 论文第 3 节（arXiv:2603.15569）。用三句话解释为何复数值状态更新等价于对状态的“数据相关旋转嵌入（rotary embedding）”。将答案与 Phase 7 · Lesson 04 的 RoPE 推导联系起来。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State space model (SSM) | "Recurrence with a fixed state" | 一种具有学习递归 `h_t = A h_{t-1} + B x_t` 的层；每个 token 的内存是常数 |
| Selective SSM | "Mamba's trick" | 数据相关的 A、B、C 参数，赋予模型类似门控的选择性，同时保持线性时间 |
| Attention-to-Mamba ratio | "How many attention layers" | 在 Jamba 中，`l = 8` 表示每 7 层 Mamba 配 1 层 Attention |
| Jamba block | "The 8-layer group" | 一个包含 1 层 Attention + 7 层 Mamba，并在交替位置使用 MoE 的 8 层组 |
| SSM state | "The hidden buffer" | 每层的固定大小状态，替代 Mamba 层的 KV 缓存 |
| 256k context | "Jamba's flagship number" | Jamba-1 能在单个 80GB GPU 上容纳的序列长度；纯 Transformer 在该规模下不可行 |
| Mamba-3 | "2026 pure SSM" | 当前最优的纯 SSM 架构，包含复数状态 + MIMO；混合架构将以此为 SSM 端进行重建 |
| MIMO | "Multi-input multi-output" | Mamba-3 的创新，使用矩阵值投影而非按特征的标量投影 |
| Exponential-trapezoidal discretization | "Mamba-3's recurrence" | 更具表现力的递归，包含并代替 Mamba-2 的 Euler 离散化 |
| Hybrid architecture | "Mix attention and SSM" | 任意将 Transformer 与 SSM 层交错的模型；Jamba 是生产级样板 |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 原始 Jamba 论文，比例消融，256k 上下文声明  
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — 扩展后家族，398B/94B 与 12B/52B 的公开发布  
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba 所基于的选择性 SSM 论文  
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — 简化的结构化状态空间继任者  
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — 复数状态、MIMO，2026 年的纯 SSM 前沿  
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 论文，是 SSM 系谱在 LLM 中的起点