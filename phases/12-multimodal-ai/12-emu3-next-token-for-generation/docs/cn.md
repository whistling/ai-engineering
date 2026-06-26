# Emu3: 用于图像和视频生成的下一标记预测

> BAAI 的 Emu3（Wang 等人，2024 年 9 月）是 2024 年应当终结扩散与自回归争论的成果。一个单一的 Llama 风格的仅解码器 Transformer，仅使用下一标记预测（next-token prediction，NTP）目标，在一个统一的词表上包含文本 + VQ 图像标记 + 3D VQ 视频标记，在图像生成上击败了 SDXL，在感知任务上击败了 LLaVA-1.6。没有 CLIP 损失。没有扩散调度。推理时为提高质量使用无分类器引导（Classifier-free guidance，CFG），但核心训练目标仍是带有 teacher forcing 的下一标记预测。发表于 Nature。本课阅读 Emu3 论文 —— 为什么更好的分词器加上规模就足够 —— 并与扩散方法进行对比。

**Type:** 学习  
**Languages:** Python（stdlib、3D 视频分词器数学 + 自回归采样器骨架）  
**Prerequisites:** Phase 12 · 11 (Chameleon)  
**Time:** ~120 分钟

## 学习目标

- 解释为什么 Emu3 的单一损失下一标记目标在长期以来认为必须使用扩散才能获得高质量图像的观点下仍然有效。
- 描述 3D 视频分词器：什么是时空 VQ 码本，以及为什么 patch 要跨越时间维度。
- 比较 Emu3 与 Stable Diffusion XL（在训练算力、推理成本、质量上限方面）的差异。
- 说出同一 Emu3 模型扮演的三种角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题背景

到 2024 年的主流观点：图像生成需要扩散。论点是：离散图像标记损失过多信息，无法重建细节；而自回归采样在数千个标记上累计误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 等都使用某种扩散形式。Chameleon（第 12.11 课）在小规模上部分反驳了这一点，但未能达到 SDXL 的质量。

Emu3 直接挑战了这一观点。其主张是：更好的视觉分词器 + 足够的规模 + 下一标记损失 = 在同一模型中超过扩散的图像生成能力，同时还能处理感知任务。

该赌注在发表时颇具争议。两年后，开源的统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）成为研究的默认路径；生产端的前沿模型也似乎采用某种变体。

## 概念

### Emu3 分词器

关键成分是视觉分词器。Emu3 训练了一个定制的 IBQ 类分词器（Inverse Bottleneck Quantizer，SBER-MoVQGAN 系列），每个标记做 8x8 的分辨率压缩。一个 512x512 的图像变为 64x64 = 4096 个标记，码本大小为 32768。

这比 Chameleon 在 512x512 下的 1024 个标记（K=8192）多，但每个标记更便宜（更小的码本查找、更简单的编解码器）。关键指标是重建 PSNR 为 30.5 dB，可与 Stable Diffusion 的连续潜空间 32 dB 竞争。

对于视频：一个 3D VQ 分词器将一个时空 patch（4x4x4 像素）编码为一个整数。一个 4 秒的剪辑在 8 FPS 下有 32 帧；在 256x256 分辨率并做 4x 空间和 4x 时间压缩时，标记数为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 个标记。

分词器质量决定了天花板。Emu3 的贡献部分在于“我们训练了一个很优秀的分词器”。

### 单一损失训练

Emu3 使用一个目标：在文本标记、2D 图像标记和 3D 视频标记共享词表上做下一标记预测。训练时按模态乘以特定权重来平衡各模态的贡献，但损失函数一致。

训练数据混合示例：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：类似
- 仅文本：标准 NTP

模型从数据分布中学习何时输出图像标记或文本标记。生成是在模型在 `<image>` 标签之后预测图像标记时自发出现的。

### 无分类器引导（CFG）与温度

自回归图像生成在推理时使用无分类器引导会大幅提升效果。Emu3 使用这一技巧：生成两次，一次带完整标题（条件），一次带空标题（无条件），用一个引导权重将 logits 混合（典型 3.0–7.0）。这是从扩散领域借用到自回归设置的同样 CFG 手法。

温度很重要：太高会出现伪影；太低会导致模式坍缩。Emu3 推荐的温度为感知任务 1.0，图像生成 0.8。

### 三种角色，一个模型

Emu3 以三种功能上不同的 API 提供，但底层权重相同：

- Emu3-Gen：图像生成。输入文本，输出图像标记。
- Emu3-Chat：视觉问答与字幕。输入图像（标记），输出文本。
- Emu3-Stage2：视频生成与视频 VQA。输入文本或视频，输出文本或视频。

没有任务专用头。只需不同的提示模板。使用相同的 checkpoint。

### 基准

来自 Emu3 论文（2024 年 9 月）：

- 图像生成：在 MJHQ-30K FID 上优于 SDXL（5.4 vs 5.6），在 GenEval 综合指标上接近（0.54 vs 0.55 — 统计上并列），在 Deep-Eval 复合指标上也不落下风。
- 图像感知：在 VQAv2 上优于 LLaVA-1.6（75.1 vs 72.4），在 MMMU 上大致持平。
- 视频生成：4 秒剪辑质量在 FVD 上与 Sora 时代公开基准模型竞争。

这些数字并非在每项上都压倒性胜出——Emu3 在某些点让一分，在其他点赢一分——但“下一标记预测就是全部所需”的论断在多模态上是有可辩护性的。

### 计算成本

Emu3 在 ~3000 亿（300B）多模态标记上训练，模型规模为 7B 参数。GPU 小时大致可与 Llama-2-7B 预训练相当（在 A100 级芯片上约 2k–4k GPU 年）。像 Stable Diffusion 这类扩散模型在类似预算下训练，但需要单独的文本编码器和更复杂的流水线。

在推理时，Emu3 每图像比 SDXL 慢：4096 个图像标记以 30 tok/s 速度生成约需 ~2 分钟来产出一张 512x512 图像，而 SDXL 仅需 2–5 秒。投机性解码（speculative decoding）和 KV-cache 优化能缩小差距但无法完全弥补。自回归图像生成计算密集；这是当前的权衡。

### 为什么重要

Emu3 的深远贡献在于概念层面。如果下一标记预测在图像生成上可扩展到匹敌扩散，那么统一模型路径（一种损失、一个骨干、任意模态）是可行的。未来模型不再需要独立的文本编码器、独立的扩散调度器、独立的 VAE。一个 Transformer，加上每种模态一个优秀的分词器，就够了。

Show-o、Janus-Pro 和 InternVL-U 都在基于或挑战这一论断。到 2025 年，中文科研机构（BAAI、DeepSeek）在这一方向上比美方机构更积极地发表成果。

## 使用方法

`code/main.py` 构建了两段玩具代码：

- 一个 2D 与 3D VQ 分词器标记计数计算器：给定（分辨率、patch、剪辑长度、FPS），计算图像与视频的标记数。
- 一个带有无分类器引导和温度的自回归图像标记采样器骨架。

CFG 的实现遵循 Emu3 的配方 —— 将有条件与无条件 logits 按引导权重混合。

## 部署

本课会产出 `outputs/skill-token-gen-cost-analyzer.md`。给定一个生成产品规格（图像或视频、目标分辨率、质量等级、延迟预算），它会计算标记数、推理成本，并从 Emu3 系列与扩散方法中做出选择。

## 练习

1. Emu3 在 8x8 压缩下对 512x512 图像产生 4096 个标记。计算 1024x1024 和 2048x2048 的等价标记数。推理延迟会如何变化？

2. 阅读 Emu3 论文第 3.3 节关于视频分词器的内容。描述 3D VQ patch 的形状以及为什么是 4x4x4 而不是 8x8x1。

3. 无分类器引导权重为 5.0 与 3.0 会产生什么视觉效果？在 `code/main.py` 中追踪其数学原理。

4. 计算 Emu3-7B 在 300B 标记上的训练 FLOPs，并与 Stable Diffusion 3 比较。哪一个训练成本更高？

5. Emu3 在 FID 上优于 SDXL，但在 VQAv2 上不及专用 VLM（视觉语言模型）。解释为什么统一损失方法在不同基准上表现出不同的强项与弱项。

## 关键术语

| 术语 | 常说的说法 | 实际含义 |
|------|------------|----------|
| Next-token prediction | "NTP" | 标准的自回归损失：在给定 token[0..i] 的情况下预测 token[i+1]；当所有模态都被标记化后对每种模态都适用 |
| IBQ tokenizer | "Inverse bottleneck quantizer" | 一类 VQ-VAE，使用更大的码本（32768+），比 Chameleon 有更好的重建能力 |
| 3D VQ | "Spatiotemporal quantizer" | 按（时间、行、列）索引的码本；一个标记覆盖 4x4x4 像素立方体 |
| Classifier-free guidance | "CFG" | 将有条件与无条件的 logits 按权重 γ 混合；在推理时提升图像质量 |
| Unified vocabulary | "Shared tokens" | 文本 + 图像 + 视频共享同一个整数空间；模型预测接下来出现的任意模态标记 |
| MJHQ-30K | "Image gen benchmark" | 一个包含 30k 提示的 Midjourney 质量基准；Emu3 在此报告 FID |

## 延伸阅读

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)  
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)  
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)  
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)  
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)