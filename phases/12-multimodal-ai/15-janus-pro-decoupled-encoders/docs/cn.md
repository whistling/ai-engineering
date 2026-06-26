# Janus-Pro: 用于统一多模态模型的解耦编码器

> 统一多模态模型存在不可避免的张力。理解任务需要语义化特征 —— SigLIP 或 DINOv2 输出的向量，富含概念级信息。生成任务需要便于重建的编码 —— 可以组合回清晰像素的 VQ 令牌。单一编码器无法同时兼顾这两者。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1 月）提出的解决办法是停止尝试：解耦两个编码器。任务之间共享 transformer 主体，但理解走 SigLIP，生成走 VQ tokenizer。在 7B 参数下，Janus-Pro 在 GenEval 上击败了 DALL-E 3，同时在 MMMU 上与 LLaVA 匹配。这个经验说明了为什么两个编码器在一个失败的地方能成功。

**Type:** 构建  
**Languages:** Python（stdlib，双编码器路由 + 共享主体信号）  
**Prerequisites:** 阶段 12 · 13 (Transfusion)，阶段 12 · 14 (Show-o)  
**Time:** ~120 分钟

## 学习目标

- 解释为什么单一共享编码器会牺牲理解或生成质量中的一项。
- 描述 Janus-Pro 的路由：理解侧使用输入端的 SigLIP 特征，生成侧输入与输出均使用 VQ 令牌。
- 追踪使 Janus-Pro 成功的数据混合与规模扩展。
- 比较解耦（Janus-Pro）、耦合-连续（Transfusion）和耦合-离散（Show-o）架构。

## 问题

统一模型在理解和生成之间共享 transformer 主体。先前的尝试（Chameleon、Show-o、Transfusion）都为两个方向使用同一个视觉 tokenizer。这个 tokenizer 是一种折中：

- 为重建（生成）优化：VQ-VAE 捕捉细粒度像素细节，但生成的令牌语义连贯性较差。
- 为语义（理解）优化：SigLIP 嵌入使“猫”类图像在向量空间聚集，但不利于良好重建。

Show-o 和 Transfusion 因此在某一方向上付出了明显的可见质量代价。Janus-Pro 的问题是：既然两个任务需求不同，为什么还要只用一个 tokenizer？

## 概念

### 解耦的视觉编码

Janus-Pro 的架构将两个编码器分离：

- 理解路径。输入图像 → SigLIP-SO400m → 两层 MLP → transformer 主体。
- 生成路径。如果条件化于现有图像：输入图像 → VQ tokenizer → 令牌 ID → transformer 主体。
- 输出生成。由 transformer 预测的图像令牌 → VQ 解码器 → 像素。

transformer 主体是共享的。主体上下游的所有部分都是任务特定的。

输入通过提示格式来消歧：`<understand>` 标签通过 SigLIP 路由；`<generate>` 通过 VQ 路由。或者路由可以由任务隐式决定。

### 为何可行

理解（理解损失）使用 SigLIP 特征，CLIP 风格的预训练已将其调优用于语义相似性。模型在感知类基准上的表现优于 Show-o / Transfusion，因为输入特征更适合该任务。

生成（生成损失）使用 VQ 令牌，tokenizer 针对重建进行了调优。图像质量优于 Show-o，因为 VQ 码能干净地重组成像素。

共享的 transformer 主体会看到两种输入分布（SigLIP 和 VQ），并学会同时处理两者。结论是：只要数据足够 + 参数足够，主体能吸收这种切换。

### 数据规模 — Janus vs Janus-Pro

Janus（原作，arXiv 2410.13848）提出了解耦思想，但规模较小（1.3B 参数，数据有限）。Janus-Pro（arXiv 2501.17811）进行了扩展：

- 7B 参数（对比 1.3B）。
- Stage 1（对齐）使用 9000 万图文对（从 7200 万增长）。
- Stage 2（统一）使用 7200 万（从 2600 万增长）。
- Stage 3 增加了 20 万图像生成指令样本。

结果：Janus-Pro-7B 在 MMMU 上与 LLaVA 匹配（60.3 vs ~58），并在 GenEval 上击败 DALL-E 3（0.80 vs 0.67）。一个开源模型，在统一谱系的两侧都具有竞争力。

### JanusFlow —— 整流流（rectified-flow）变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径替换为整流流（rectified-flow）生成路径（连续）。分支变为 SigLIP 用于理解 + rectified-flow 用于生成。质量上限进一步提升。架构依然是解耦编码器 + 共享主体。

### 共享主体的职责

transformer 主体处理统一的序列，但来自两种输入分布。它的职责是：

- 对于理解：消费 SigLIP 特征 + 文本令牌 → 自回归地产生文本。
- 对于生成：消费文本令牌 +（可选的图像 VQ 令牌）→ 自回归地产生图像 VQ 令牌。

主体的每一层没有按模态区分的特定权重。它是你在 Qwen 或 Llama 中会看到的那种文本风格 transformer，加上两个输入适配器。

有趣的是，这意味着 Janus-Pro 的主体可以从预训练的 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这一选择很重要：预训练的 LLM 为推理能力做出了贡献，而纯从零训练的统一模型往往难以达到相同水平。

### 与 InternVL-U 的比较

InternVL-U（Lesson 12.10）是 2026 年的后续工作。它结合了：

- 原生的多模态预训练（InternVL3 主干）。
- 解耦编码器路由（SigLIP 输入，VQ + diffusion 头输出）。
- 统一的理解 + 生成 + 编辑能力。

InternVL-U 将解耦编码器的选择纳入更大的框架。解耦编码器的思想已成为大规模统一模型的默认做法。

### 局限性

解耦编码器增加了架构复杂度。需要训练两个 tokenizer、维护两条输入路径、面对两类失败模式。对于不需要生成的产品，Janus-Pro 过于复杂 —— 应选择 LLaVA 一类的理解模型。

对于不需要理解的产品，Janus-Pro 也过度配置 —— 应选择 Stable Diffusion 3 / Flux 之类的生成模型。

对于既需二者的产品，Janus-Pro 现在是开源架构的参考方案。

## 使用示例

`code/main.py` 模拟了 Janus-Pro 的路由：

- 两个模拟编码器：类似 SigLIP（输出 256 维语义向量）和类似 VQ（输出整数代码）。
- 一个提示路由器，根据任务标签选择编码器。
- 一个共享主体（占位），处理令牌序列，而不关心哪个编码器生成它们。
- 一个从 stage 1（对齐）到 stage 3（指令微调）的加权采样调度开关。

打印 3 个示例的路由路径：图像问答、文本到图像（T2I）、图像编辑。

## 部署建议

本课产出 `outputs/skill-decoupled-encoder-picker.md`。对于希望在前沿质量上同时实现生成与理解的产品，建议在 Janus-Pro、JanusFlow 或 InternVL-U 中选择一个，并给出具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上击败了 DALL-E 3。解释为什么一个 7B 的开源模型能在生成上匹配一个前沿的专有模型，但在理解上却不能？

2. 实现一个路由函数：给定提示文本，将其分类为 `understand` 或 `generate`。如何处理像 “describe and then sketch” 这种模糊提示？

3. JanusFlow 用整流流替换 VQ 路径。此时 transformer 主体输出什么？损失函数有何变化？

4. 为 Janus-Pro 架构再提出第四个任务，使用另一个解耦编码器。示例：图像分割（DINO 风格）、深度估计（MiDaS 风格）等。

5. 阅读 Janus-Pro 第 4.2 节关于数据规模的讨论。哪个数据阶段对与 Janus 相比的 T2I 质量提升贡献最大？

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Decoupled encoding | "Two visual encoders" | 每个方向使用独立的 tokenizer 或编码器：用于理解的语义编码器，和用于生成的重建编码器 |
| Shared body | "One transformer" | 单个 transformer 处理任一编码器的输出；没有按模态区分的权重 |
| SigLIP for understanding | "Semantic features" | CLIP 家族的视觉塔，提供丰富的概念特征，但重建能力差 |
| VQ for generation | "Reconstruction codes" | 向量量化的令牌，可以干净地解码回像素 |
| JanusFlow | "Rectified-flow variant" | 在生成侧使用连续的 flow-matching 头，取代 VQ 的 Janus-Pro 变体 |
| Routing tag | "Task tag" | 选择输入编码器的提示标记（`<understand>` / `<generate>`） |

## 进一步阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)  
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)  
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)  
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)  
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)