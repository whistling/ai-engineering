# Flamingo 和 门控交叉注意力用于少样本 VLMs

> DeepMind 的 Flamingo（2022）在别人之前做了两件事。它展示了单一模型可以处理任意交错的图像、视频和文本序列。并且它展示了 VLM 可以进行上下文学习——给出一个包含三个示例（图像，说明）对的少样本提示词，模型在不做任何梯度更新的情况下对新图像进行描述。机制是：在冻结的 LLM 现有层之间插入门控交叉注意力层，使用一个可学习的 tanh 门控，初始为零，这样在初始化时可以保留 LLM 的文本能力。本课讲解 Flamingo 的 Perceiver 重采样器和门控交叉注意力架构——它是 Gemini 的交错输入和 Idefics2 的视觉 token 的祖先。

**Type:** 学习  
**Languages:** Python（标准库，门控交叉注意力 + Perceiver 重采样器 演示）  
**Prerequisites:** Phase 12 · 03（BLIP-2 Q-Former）  
**Time:** ~120 分钟

## 学习目标

- 解释门控交叉注意力如何通过 tanh(gate) = 0 在初始化时保留冻结 LLM 的文本能力。
- 演示 Perceiver 重采样器：N 个图像 patch → 通过交叉注意力得到 K 个固定“潜变量”查询。
- 描述 Flamingo 如何通过尊重图像位置的因果掩码处理交错的图像-文本序列。
- 重现一个少样本多模态提示结构（3 个图像-说明示例，然后是一个查询图像）。

## 问题背景

BLIP-2 将 32 个视觉 token 送入冻结的 LLM 输入层。对于单图像提示有效。但如果你想像“这是图像 A，为它生成说明；这是图像 B，为它生成说明；现在这是图像 C，为它生成说明”那样在文本中交错多张图像怎么办？LLM 的自注意力需要在单个流里处理图像 token 和文本 token，哪些位置可以注意到哪些图像会变得很复杂。

Flamingo 的答案：完全不改变 LLM 的输入流。在现有 LLM block 之间插入额外的交叉注意力层。文本 token 依然按原样通过 LLM 的因果自注意力。在每隔几个 LLM block 之间，文本 token 还通过一个新的门控层对图像特征做交叉注意力。门（初始化为零）意味着在第零步时这些新层是无操作的——模型在初始化时完全表现为预训练的 LLM。随着训练进展，门打开，视觉信息开始流入。

Flamingo 回答的第二个问题是：如何处理每个提示中可变数量的图像（0、1 或多张）？使用 Perceiver 重采样器——一个小的交叉注意力模块，它将任意数量的 patch 转换为固定数量的视觉潜变量 token。LLM 的交叉注意力层看到的形状与图像数量无关。

## 概念

### 冻结的 LLM

Flamingo 从冻结的 Chinchilla 70B LLM 开始。70B 的权重全部保持不动。现有的文本自注意力和 FFN 照常运行。

### Perceiver 重采样器

对于提示中的每张图像，ViT 会产生 N 个 patch token。Perceiver 重采样器有 K 个固定的可学习潜变量（Flamingo 使用 K=64）。每个重采样器块有两个子步骤：

1. 交叉注意力：K 个潜变量对 N 个 patch token 做注意力（Q 来自潜变量，K/V 来自 patch）。
2. 潜变量内的自注意力 + FFN。

经过 6 个重采样器块后，输出为 K=64 个维度为 1024 的视觉 token，无论 ViT 产生多少 patch。224x224 图像（196 个 patch）和 480x480 图像（900 个 patch）都输出为 64 个重采样器 token。

对于视频，重采样器按时间应用：每帧的 patch 生成 64 个潜变量，并加入时间位置编码使模型能区分 t=0 与 t=N。完整视频变为 T * 64 个视觉 token。

### 门控交叉注意力

在每隔 M 层的冻结 LLM（Flamingo 使用 M=4）之间，插入一个新的门控交叉注意力块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个可学习的标量，初始化为零。  
- `tanh(0) = 0`，因此在初始化时门控分支贡献为零。  
- 随着 `alpha` 偏离零，交叉注意力的贡献平滑增长。  
- 残差连接意味着即使门完全打开也不会覆写 LLM 的文本表示；它只是将视觉信息叠加上去。

这是 Flamingo 最重要的设计选择：视觉条件化是可加的、有门控的，并在初始化时为零。初始化时的 Flamingo 在仅文本输入上完全等同于 Chinchilla 70B。

### 用于交错输入的掩码交叉注意力

在类似 "<image A> caption A <image B> caption B <image C> ?" 的提示中，每个文本 token 应该只能看到序列中在它之前出现的图像。交叉注意力掩码强制执行：文本位置 `t` 只能注意那些图像索引 `i < i_t` 的重采样器 token，其中 `i_t` 是位置 `t` 之前最近的图像索引。是“只看到最后一个前置图像”或“看到所有前置图像”都可以；Flamingo 选择了前者。

### 上下文内少样本学习

一个 Flamingo 提示看起来像：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到完成模式并输出 "bird"（或 image3 所示的内容）。没有梯度步骤。冻结的 LLM 的上下文内学习能力通过门控交叉注意力得以保留——这是论文的要点以及其重要性所在。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb (M3W)：4300 万个带图像与文本交错的网页，重建阅读顺序。  
2. 图像-文本对（ALIGN + LTIP）：44 亿对。  
3. 视频-文本对（VTP）：2700 万个短视频片段。

OBELICS（2023）是对交错网页语料的开源复现，Idefics、Idefics2 以及大多数开源“Flamingo-like”模型都在其上训练。

### OpenFlamingo 和 Otter

OpenFlamingo（2023）是开源复现。架构相同（Perceiver 重采样器 + 冻结的 LLaMA 或 MPT 上的门控交叉注意力）。提供 3B、4B、9B 的检查点。由于基模型较小且数据较少，质量落后于 Flamingo。

Otter（2023）在 OpenFlamingo 基础上进行指令微调（使用 MIMIC-IT 多模态指令数据集），展示了门控交叉注意力也适用于指令跟随任务。

### 后续工作

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力谱系，逐步简化（Idefics2 为了直接使用 patch token 并采用自适应池化而舍弃重采样器）。  
- Flamingo 到 Chameleon 的转变：到 2024 年，许多团队转向早融合（Lesson 12.11）；在需要冻结骨干网络的场景中，Flamingo 风格的门控交叉注意力仍然在生产中使用。  
- Gemini 的交错输入：在概念上继承了 Flamingo 的交错格式灵活性，尽管具体机制是专有的。

### 与 BLIP-2 的比较

| | BLIP-2 | Flamingo |
|---|---|---|
| Visual bridge | Q-Former 一次性在输入处 | 每隔 M 层的门控交叉注意力 |
| Visual tokens | 每图 32 个 | 每次交叉注意力每图 64 个 |
| Frozen LLM | 是 | 是 |
| Few-shot in-context | 弱 | 强 —— 论文核心 |
| Interleaved inputs | 无原生支持 | 有，设计目标 |
| Training data | 1.3 亿对 | 13 亿对 + 4300 万交错页面 |
| Parameter count | 训练参数 1.88 亿 | 训练参数约 100 亿（交叉注意力层） |
| Compute | 在 8 个 A100 上数天 | 在数千个 TPUv4 上数周 |

预算有限做单图像 VQA 选 BLIP-2。需要交错、少样本或多图像推理选 Flamingo / Idefics2。

## 使用示例

`code/main.py` 演示了：

1. 在 36 个虚假 patch token 上的 Perceiver 重采样器，使用 8 个可学习潜变量（纯 Python 实现的交叉注意力）。  
2. 一个门控交叉注意力步骤，`alpha = 0` → 输出等于输入（LLM 未改变），然后 `alpha = 2.0` → 视觉贡献被混入。  
3. 一个交错掩码构建器，生成 "(image 1) (text 1) (image 2) (text 2)" 序列的 2D 注意力掩码。

## 交付物

本课产出 `outputs/skill-gated-bridge-diagnostic.md`。给定一个开源 VLM 的配置（是否有重采样器、交叉注意力频率、门控方案），它会识别 Flamingo 血统的要素并解释冻结策略。对于排查为何微调导致文本性能下降很有用（答案通常是：门打开得太快）。

## 练习

1. 计算 Flamingo-9B 的视觉参数量：9B LLM + 14 亿门控交叉注意力参数 + 6400 万重采样器参数。训练参数占总参数的比例是多少？  
2. 用 PyTorch 实现门控残差 `y = tanh(alpha) * cross + x`。实验性地展示在 `alpha=0` 时，初始状态下 `y==x` 精确成立。  
3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390）关于当每个提示的图像数量不同批处理中如何处理多图像的做法。描述其填充策略。  
4. 为什么 Flamingo 的交叉注意力掩码让文本 token 只注意到最近的前置图像而不是所有前置图像？阅读 Flamingo 论文第 2.4 节并解释权衡。  
5. 上下文内少样本：为一个新的 Flamingo 变体构造一个包含 4 个“图像 → 主要物体颜色”示例的提示。描述当示例数从 0 变到 8 时预期的准确率变化模式。

## 术语表

| 术语 | 大家如何说 | 实际含义 |
|------|----------------|------------------------|
| Perceiver resampler | “固定潜变量交叉注意力” | 一个模块，从可变数量的输入 patch 生成 K 个固定 token |
| Gated cross-attention | “Tanh 门控桥” | 残差层 `y = tanh(alpha)*cross + x`，可学习的 alpha，初始化为 0 |
| Interleaved input | “混合序列” | 图像与文本按阅读顺序自由混合的提示格式 |
| Frozen LLM | “LLM 无梯度” | 文本 LLM 的权重不更新；仅训练重采样器和交叉注意力层 |
| Few-shot | “上下文示例” | 在提示中给出少量（图像，答案）对；模型在不微调的情况下泛化 |
| OBELICS | “交错网页语料” | 一个包含 1.41 亿网页、按阅读顺序排列图像与文本的开源数据集 |
| Chinchilla | “70B 冻结基模型” | Flamingo 使用的冻结文本 LLM，来自 DeepMind 的 Chinchilla 工作 |
| Gate schedule | “alpha 的变化曲线” | 训练期间交叉注意力门如何打开的速率 |
| Cross-attn frequency | “每 M 层插入频率” | 表示多常插入一个门控交叉注意力块；Flamingo 使用 M=4 |
| OpenFlamingo | “开源复现” | MosaicML/LAION 的开源检查点，3-9B，架构与 Flamingo 一致 |

## 进一步阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。  
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现。  
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网页语料。  
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — Perceiver 通用架构。  
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 指令微调的 Flamingo 后继。  
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — 对 Flamingo 方法的现代简化。