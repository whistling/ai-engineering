# Visual Autoregressive Modeling (VAR): Next-Scale Prediction

> Diffusion 模型在时间上逐步采样（去噪步骤）。VAR 在尺度上逐步采样——它先预测 1x1 令牌，然后是 2x2、4x4，一直到最终分辨率，每个尺度都以先前尺度为条件。2024 年的论文显示 VAR 在图像生成上遵循 GPT 式的缩放律，并在相同计算预算下优于 DiT。本课搭建核心机制。

**Type:** 构建  
**Languages:** Python（使用 PyTorch）  
**Prerequisites:** 第7阶段 第03课（多头注意力），第8阶段 第06课（DDPM）  
**Time:** ~90 分钟

## 问题

自回归生成在语言建模中占主导地位，因为它具有可预测的缩放特性：更多计算、更多参数、更低困惑度、输出更好。2024 年之前，图像生成有两种主要的自回归尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE 代码上逐令牌）。

两者都受困于生成顺序问题。像素和令牌排列在 2D 网格中，但自回归模型必须以 1D 的光栅顺序访问它们。一个早期的角像素不知道最终图像会成为什么样子。生成质量的缩放性弱于 GPT 在文本上的表现，并且在相同计算下从未达到扩散模型的质量。

VAR 通过改变生成对象修复了生成顺序问题。VAR 不再逐空间预测单个图像令牌，而是按越来越高的分辨率预测整张图像。步骤 1：预测 1x1 令牌（图像的整体“摘要”）。步骤 2：预测 2x2 网格的令牌（粗粒度特征）。步骤 3：预测 4x4 网格。第 K 步：预测最终的 (H/8)x(W/8) 网格。

每个尺度对所有先前尺度进行注意（在“尺度顺序”上因果），并且在自身尺度内并行。顺序问题消失了：尺度 k 的整张图像在一次 transformer 前向中产生。

## 概念

### VQ-VAE 多尺度分词器

VAR 需要一个多尺度离散分词器。对于图像 x，它会产生一系列逐步更高分辨率的令牌网格：

```
x -> encoder -> 潜在表示 f
f -> 在 1x1 下量化: 令牌网格 z_1，形状 (1, 1)
f -> 在 2x2 下量化: 令牌网格 z_2，形状 (2, 2)
...
f -> 在 (H/p)x(W/p) 下量化: 令牌网格 z_K，形状 (H/p, W/p)
```

每个 z_k 使用相同的码本（典型大小 4096–16384）。每个尺度的量化不是独立的——它被训练成在各尺度残差相加时可以重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一个**残差 VQ** 变体。尺度 k 捕获尺度 1..k-1 未能表示的部分。解码器将所有尺度的嵌入求和并生成图像。

多尺度 VQ 分词器像 VQGAN 那样训练一次并冻结。所有生成工作都由其上的自回归模型完成。

### 下一尺度预测

生成模型是一个 transformer，它看到所有先前尺度的令牌并预测下一个尺度的令牌。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入同时编码尺度索引和尺度内的空间位置。注意力在尺度顺序上是因果的：尺度 k 的位置 (i, j) 可以关注尺度 1..k 的所有令牌，以及尺度 k 本身在所使用的尺度内顺序中更早出现的那些令牌（VAR 使用固定位置注意力并且在尺度内没有因果性——尺度内的所有位置并行预测）。

训练损失：在每个尺度 k 上，给定所有先前尺度令牌，预测令牌 z_k。对离散 VQ 代码使用交叉熵损失。结构与 GPT 相同，只是“序列”现在具有尺度结构。

### 生成

在推理时：
```
generate z_1 = sample from p(z_1)                    # 1 个令牌
generate z_2 = sample from p(z_2 | z_1)              # 4 个令牌并行生成
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 个令牌并行生成
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于 K = 10 个尺度，生成需要 10 次 transformer 前向。每次前向在一个尺度内并行产生该尺度的全部令牌——尺度内没有逐令牌自回归。对于 256x256 图像，这大约是 10 次前向，而 DiT 约需 28–50 次。

### 为什么下一尺度优于逐令牌自回归

三个结构性优势：
1. 粗到细符合图像的自然统计特性。人类视觉和图像数据集都表现出尺度依赖的规律：低频结构稳定且可预测；高频细节依赖于低频内容。下一尺度预测能利用这一点。
2. 尺度内并行生成。与 GPT 式的逐令牌自回归不同，VAR 在一个步骤内生成尺度上的所有令牌。有效生成长度是对数尺度而不是线性尺度。
3. 无生成顺序偏差。尺度 k 的令牌可以看到整个尺度 k-1；不存在“左边”或“上方”的偏差，迫使早期令牌在获得完整上下文前就做出承诺。

### 缩放律

Tian 等人证明 VAR 在 ImageNet 上的 FID 遵循幂律缩放曲线——就像 GPT 对困惑度所做的那样。参数或计算加倍通常可使误差减半。这是首个在图像生成中像语言模型那样清晰表现出此类缩放行为的模型。结果是，VAR 的尺度预测可以从计算预算可信地推断出来，而不必依赖针对每种架构的经验猜测。

### 与扩散的关系

VAR 与扩散有相同的数据压缩故事：两者都将生成问题分解为一系列更简单的子问题。

- 扩散：逐步添加噪声，学习逆去一步。
- VAR：逐步添加分辨率，学习预测下一尺度。

它们沿不同的轴分解问题。两者都得到可处理的条件分布。实证上 VAR 在推理时更快（更少前向，在尺度内全部并行）并且在类条件 ImageNet 上与或优于 DiT。文本条件 VAR（VARclip、HART）是一个活跃的研究方向。

## 构建它

在 `code/main.py` 中你将：
1. 在合成“图像”数据（二维高斯环）上构建一个小型的**多尺度 VQ 分词器**。
2. 训练一个 **VAR 风格 transformer** 来进行下一尺度预测。
3. 通过调用 transformer 4 次（4 个尺度）进行采样并解码。
4. 验证按尺度排序的训练使得尺度内能够并行生成。

这是一个玩具实现。目的是观察尺度结构化的注意力掩码和尺度内并行生成的实际工作方式。

## 交付

本课将产生 `outputs/skill-var-tokenizer-designer.md` — 一个用于设计多尺度分词器的技能文档：尺度数量、尺度比率、码本大小、残差共享、解码器架构等。

## 练习

1. 缩放层数消融。分别训练具有 4、6、8、10 个尺度的 VAR。测量重建质量与自回归前向次数的关系。更多尺度 = 更细的残差 = 更好质量但更多前向次数。
2. 码本大小。训练码本大小为 512、4096、16384 的分词器。更大的码本能带来更好的重建但更难预测。找出拐点。
3. 尺度内并行性检查。对于训练好的 VAR，明确测量注意力模式。在尺度 k 内，模型是否注意到跨尺度位置但不注意尺度内的因果位置？验证掩码实现。
4. VAR vs DiT 缩放。针对相同的 ImageNet 类条件任务，在相同参数预算（例如 33M、130M、458M）下训练 VAR 和 DiT。绘制 FID vs 计算量。VAR 应在各规模上领先于 DiT——在小规模上复现论文结果。
5. 文本条件。将 VAR 扩展为通过 adaLN 接受文本嵌入（CLIP 池化）作为额外条件输入。这是 HART 的做法。文本对齐采样会使 FID 提高多少？

## 关键术语

| Term | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | 通过在 VQ 令牌网格金字塔上进行逐层（下一尺度）预测来进行图像生成 |
| Next-scale prediction | "Predict coarser, then finer" | 模型按递增分辨率尺度预测令牌，每一层以所有先前尺度为条件 |
| Multi-scale VQ tokenizer | "Residual VQ" | 产生 K 个递增分辨率令牌网格的 VQ-VAE，解码器将所有尺度相加重建 |
| Scale k | "Pyramid level k" | K 个分辨率级中的一个，从 k=1 的 1x1 到 k=K 的 (H/p)x(W/p) |
| Parallel-within-scale | "One forward per scale" | 尺度 k 的所有令牌在一次 transformer 前向中预测，而非按令牌自回归 |
| Causal-across-scales | "Scale-ordered attention" | 尺度 k 的令牌可以注意到尺度 1..k 的所有令牌，但不能注意到 k+1..K |
| Residual VQ | "Additive tokenization" | 每个尺度的令牌编码由较低尺度留下的残差；解码器对所有尺度的嵌入求和 |
| VAR scaling law | "Image GPT scaling" | 在计算量上 FID 遵循可预测的幂律，就像语言模型的困惑度一样 |
| HART | "Hybrid VAR + text" | 结合 VAR 尺度结构和文本条件的变体（混合自回归 transformer） |
| Scale position embedding | "(scale, row, col) triple" | 位置编码同时携带尺度索引和尺度内的行列坐标 |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，规范参考  
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，扩散对比基线  
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 的多尺度分词器的扩展家族  
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像分词的基础  
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — 文本条件 VAR 的 HART 方法