# Vision Transformers and the Patch-Token Primitive

> 在任何多模态处理之前，图像必须先被转换成变压器能够处理的令牌序列。2020 年的 ViT 论文通过 16x16 像素的补丁、线性投影和位置嵌入解决了这个问题。五年后，所有 2026 年前沿模型（Claude Opus 4.7 原生 2576px、Gemini 3.1 Pro、Qwen3.5-Omni）仍以此为起点 —— 编码器从 ViT 发展到 DINOv2 再到 SigLIP 2，加入了 register tokens，位置方案演进为 2D-RoPE，但这个原语依然成立。本课阅读补丁—令牌流水线的端到端实现，并用 Python 标准库实现一个版本，以便第 12 阶段其余内容对“视觉令牌”有一个具体的心智模型。

**Type:** 学习  
**Languages:** Python（标准库，补丁分词器 + 几何计算器）  
**Prerequisites:** 第7阶段（Transformers）、第4阶段（计算机视觉）  
**Time:** ~120 分钟

## 学习目标

- 将 HxWx3 的图像转换为带有正确位置编码的补丁令牌序列。
- 计算给定（补丁大小、分辨率、隐藏维度、深度）的 ViT 的序列长度、参数量和 FLOPs。
- 说出将 ViT 从 2020 年研究推向 2026 年生产的三项升级：自监督预训练（DINO / MAE）、register tokens、以及原生分辨率打包（native-resolution packing）。
- 在下游任务中在 `CLS` 池化、平均池化和 register tokens 之间做出选择。

## 问题陈述

变压器在向量序列上运行。文本已经是序列（字节或令牌）。图像是一个带有三个颜色通道的二维像素网格 —— 不是序列。如果你把每个像素展平，224x224 的 RGB 图像会变成 150,528 个令牌，而在这个长度上做自注意力是行不通的（关于序列长度是二次复杂度）。

2020 年之前的方法在前端拼接一个 CNN 特征提取器：ResNet 产生 7x7 的 2048 维特征图，把那 49 个令牌送入变压器。这可以工作，但继承了 CNN 的偏置（平移等变性、本地感受野）并失去了变压器对规模的适应性。

Dosovitskiy 等人（2020）提出一个直接的问题：如果我们跳过 CNN 会怎样？把图像分成固定大小的补丁（比如 16x16 像素），对每个补丁做线性投影得到向量，加上位置嵌入，然后把序列送入标准变压器。那时这在视觉领域被视为异端 —— 无卷积的视觉。使用足够的数据（JFT-300M，然后 LAION）后，它在 ImageNet 上击败了 ResNet，并持续改进。

到 2026 年，ViT 原语已成为不容置疑的基础。每个开源权重的 VLM 的视觉塔都是某个后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是“我们是否使用补丁？”，而是“使用什么补丁大小，采用什么分辨率策略，什么预训练目标，什么位置编码”。

## 概念

### 作为令牌的补丁

给定形状为 `(H, W, 3)` 的图像 `x` 和补丁大小 `P`，将图像划分为 `(H/P) x (W/P)` 的不重叠补丁网格。每个补丁是一个 `P x P x 3` 的像素立方体。把每个立方体展平成 `3 P^2` 的向量。应用共享的线性投影矩阵 `W_E`，其形状为 `(3 P^2, D)`，将每个补丁映射到模型的隐藏维度 `D`。

对于 ViT-B/16 的经典配置：
- 分辨率 224，补丁大小 16 → 网格 14x14 → 196 个补丁令牌。
- 每个补丁是 `16 x 16 x 3 = 768` 个像素值，被投影到 `D = 768`。
- 添加一个可学习的 `[CLS]` 令牌 → 序列长度 197。

补丁投影在数学上等价于一个核大小为 `P`、步幅为 `P`、输出通道数为 `D` 的二维卷积。这也是生产代码的实际实现方式 —— `nn.Conv2d(3, D, kernel_size=P, stride=P)`。把它称为“线性投影”是概念性的；以卷积核实现更高效。

### 位置嵌入

补丁本身没有固有顺序 —— 变压器把它们看作一个集合。早期 ViT 加入了可学习的一维位置嵌入（每个位置一向量，ViT-B/16 是 197 个 768 维向量）。这能工作，但把模型绑定到训练分辨率：推理时如果改变网格就必须对位置表进行插值。

现代视觉骨干使用 2D-RoPE（Qwen2-VL 的 M-RoPE，SigLIP 2 的默认）或分解的二维位置编码。2D-RoPE 基于补丁的（行, 列）索引对 query 和 key 向量做旋转，使模型能通过旋转角度推断相对二维位置。无需位置表。模型可以在推理时处理任意网格大小。

### `CLS` 令牌、池化输出与 register tokens

图像级表示是什么？有三种共存的选择：

1. `CLS` 令牌。把一个可学习向量放在补丁序列前面。经过所有变换器块后，CLS 令牌的隐藏态就是图像表示。继承自 BERT。被原始 ViT、CLIP 使用。
2. 平均池化。对补丁令牌的输出隐藏态取均值。被 SigLIP、DINOv2 和大多数现代 VLM 使用。
3. Register tokens。Darcet 等人（2023）观察到，未使用显式“汇聚”令牌训练的 ViT 会发展出高范数的“伪影”补丁，劫持自注意力。添加 4–16 个可学习的 register tokens 可以吸收这种负担并改善密集预测质量（分割、深度估计）。DINOv2 和 SigLIP 2 都带有 register tokens。

选择会影响下游任务。CLS 对分类任务足够。对于将补丁令牌传入 LLM 的 VLM，你通常完全跳过池化 —— 每个补丁都成为 LLM 的输入令牌。Register tokens 在传递给 LLM 之前会被丢弃（它们是支架，而非内容）。

### 预训练：监督、对比、掩码、蒸馏自监督

2020 年的 ViT 是在 JFT-300M 上做监督分类预训练的。很快被以下方法取代：

- CLIP（2021）：基于 4 亿对图文对的对比学习。见 Lesson 12.02。
- MAE（2021，He 等人）：掩码 75% 的补丁，重构像素。纯图像自监督。
- DINO（2021）/ DINOv2（2023）：基于 student-teacher 的自蒸馏，无标签、无标题。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉骨干，默认用于“密集特征”场景。
- SigLIP / SigLIP 2（2023，2025）：用 sigmoid 损失的 CLIP，并使用 NaFlex 处理原生长宽比。是 2026 年开源 VLM（Qwen、Idefics2、LLaVA-OneVision）中主导的视觉塔。

你选择的预训练决定了骨干擅长的任务：CLIP/SigLIP 适合与文本的语义匹配，DINOv2 适合密集视觉特征，MAE 是下游微调的良好起点。

### 缩放规律

ViT 的缩放规律（Zhai 等人 2022）表明 ViT 的质量在模型规模、数据规模和计算量上遵循可预测的规律。在固定计算下：
- 更大的模型 + 更多的数据 → 更好的质量。
- 补丁大小是序列长度与细节保真度之间的杠杆。Patch 14（DINOv2/SigLIP SO400m 常用）比 patch 16 在每张图像上产生更多令牌；对 OCR 和密集任务更好，但速度更慢。
- 分辨率是另一个重要杠杆。从 224 到 384 再到 512 几乎总是有利，但 FLOPs 成二次代价增长。

ViT-g/14（约 10 亿参数？/ 实际 1B params，patch 14，分辨率 224 → 256 令牌）和 SigLIP SO400m/14（4 亿参数，patch 14）是 2026 年开源 VLM 的两款主力编码器。

### ViT 的参数计数

完整计算在 `code/main.py` 中。对于 224 的 ViT-B/16：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载检查点之前，先以这种方式粗估每个 ViT。骨干大小决定了你在任何下游 VLM 中的显存下限。

### 2026 年生产配置

到 2026 年，大多数开源 VLM 所搭载的编码器是 SigLIP 2 SO400m/14 的原生分辨率（NaFlex）配置。它具有：
- 4 亿参数。
- 补丁大小 14，默认分辨率 384 → 每张图像 729 个补丁令牌。
- 对图像级任务使用平均池化；对于 VQA，所有 729 个补丁都流入 LLM。
- 4 个 register tokens，在传递给 LLM 前被丢弃。
- 使用带有图像级缩放的 2D-RoPE 来支持原生长宽比。

该配置中的每个决策都可以追溯到可读的论文。

```figure
image-patch-tokens
```

## 使用方法

`code/main.py` 是一个补丁分词器和几何计算器。它接收（图像 H, W, 补丁 P, 隐藏 D, 深度 L）并输出：

- 补丁化后的网格形状和序列长度。
- 对一个合成的 8x8 像素玩具图像的令牌序列（演示展平 + 投影路径）。
- 按补丁嵌入、位置嵌入、变换器块和 head 分类的参数计数。
- 目标分辨率下单次前向的 FLOPs。
- 对比表：ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384。

运行它。把参数计数与公开的数字匹配。通过修改补丁大小和分辨率来感受令牌数量的代价。

## 交付物

本课生成 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT 配置（补丁大小、分辨率、隐藏维度、深度），它生成令牌计数、参数计数和带有理由的显存估算。每次你为 VLM 选择视觉骨干时都使用这个技能 —— 它能避免“令牌爆炸导致我的 LLM 上下文耗尽”的意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入、补丁大小 14 时的补丁-令牌序列长度。与仅使用 CLS 表示相比如何？

2. 一个 1080p 帧（1920x1080）在补丁 14 下会产生多少令牌？在 30 FPS、5 分钟的视频中，总共会有多少视觉令牌？哪种成本节省对你最有帮助：池化、帧抽样，还是令牌合并？

3. 用纯 Python 实现对补丁令牌的平均池化。验证对 196 个令牌的平均池化是否与你在请求池化嵌入时模型 `forward` 返回的结果一致（例如 DINOv2 的输出）。

4. 阅读 “Vision Transformers Need Registers” (arXiv:2309.16588) 的第 3 节。用两句话描述 registers 吸收了什么伪影以及这对下游密集预测为何重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定不同分辨率的图像列表，产生单个打包序列和块对角注意力掩码。在你学习到 Lesson 12.06 时对照验证。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Patch | "16x16 pixel square" | 输入图像的固定大小不重叠区域；成为一个令牌 |
| Patch embedding | "Linear projection" | 一个共享学习矩阵（或 stride=P 的 Conv2d），将展平的补丁像素映射到 D 维向量 |
| CLS token | "Class token" | 预置的可学习向量，其最终隐藏态表示整张图像；在 2026 年可选 |
| Register token | "Sink token" | 额外的可学习令牌，用于吸收 ViT 在预训练中产生的高范数注意力伪影 |
| Position embedding | "Positional info" | 每位置向量或旋转，使序列具备顺序感；现代默认是 2D-RoPE（位置嵌入） |
| Grid | "Patch grid" | 给定分辨率和补丁大小时的 `(H/P) x (W/P)` 补丁二维数组 |
| NaFlex | "Native flexible resolution" | SigLIP 2 的特性：单模型支持多种长宽比和分辨率，无需重训 |
| Backbone | "Vision tower" | 预训练的图像编码器，其补丁令牌输出会被送入 VLM 的 LLM |
| Pooling | "Image-level summary" | 将补丁令牌转为单个向量的策略：CLS、均值、注意力池化或基于 register 的方法 |
| Patch 14 vs 16 | "Finer vs coarser grid" | Patch 14 在每张图像上产生更多令牌，对 OCR 更友好但更慢；patch 16 是经典默认 |

## 延伸阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无标签。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register tokens 与伪影分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认的视觉塔。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验性缩放规律。