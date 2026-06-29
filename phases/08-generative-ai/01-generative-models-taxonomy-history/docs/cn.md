# 生成模型 — 分类与历史

> 每个图像模型、文本模型、视频模型和 3D 模型都归入五个桶中的一个。选错桶你会和数学较劲好几周；选对了，过去十二年的进展会在你脑中清晰地叠加起来。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 2（机器学习基础）, Phase 3（深度学习核心）, Phase 7 · 14（Transformers）  
**Time:** ~45 分钟

## 问题概述

生成模型只有一个工作：给定从某个未知分布 `p_data(x)` 抽取的训练样本，输出看起来像是来自同一分布的新样本。人脸、句子、MIDI 文件、蛋白质结构——如果你眯着眼看，都是同一个问题。

难点在于 `p_data` 存在于一个维度数百万的空间（一个 512x512 的 RGB 图像约为 786k 维），样本位于该空间内的一个薄流形上，而你可能只有大约 1000 万个样本。暴力估计密度是绝望的。每一种生成模型都是在用一个困难的问题换取一个稍微不那么困难的问题。

过去十二年里存活下来的有五个家族。知道每个家族作出的折中决策，能告诉你它为何在某些任务上获胜而在另一些任务上崩溃。

## 概念

![生成模型的五个家族 — 按建模对象的分类](../assets/taxonomy.svg)

**1. 显式密度，可求解。** 将 `log p(x)` 写成一个你实际能评估的和。自回归模型（PixelCNN、WaveNet、GPT）将 `p(x) = ∏ p(x_i | x_<i)` 因式分解。可逆流（RealNVP、Glow）把 `p(x)` 构造成对一个简单基分布的可逆变换。优点：精确似然、训练损失干净。缺点：自回归推理是顺序的（长序列慢），流模型需要可逆架构（架构受限）。

**2. 显式密度，近似解。** 从下界约束 `log p(x)`（ELBO）并优化该下界。VAE（Kingma 2013）使用带变分后验的编码器—解码器。扩散模型（DDPM，Ho 2020）训练一个去噪器，隐式地优化加权 ELBO。扩散在 2026 年成为图像、视频与 3D 的主干。

**3. 隐式密度。** 完全跳过密度；学习一个生成器 `G(z)` 产生样本，和一个判别器 `D(x)` 区分真伪。GAN（Goodfellow 2014）。推理快（一次前传），但训练时 notoriously 不稳定。StyleGAN 1/2/3 在固定域的逼真度（人脸、卧室）上即便到 2026 年仍然是最好的。

**4. 基于分数 / 连续时间。** 直接学习对数密度的梯度 `∇_x log p(x)`（即 score）。Song & Ermon（2019）证明了分数匹配把扩散推广到 SDE。Flow matching（Lipman 2023）是 2024–2026 年的热点：无仿真训练、更直的路径，比 DDPM 快 4–10 倍的采样速度。Stable Diffusion 3、Flux、AudioCraft 2 都使用 flow matching。

**5. 基于离散编码的标记自回归。** 用 VQ-VAE 或残差量化器把高维数据压缩成短序列的离散标记，然后用 Transformer 对标记序列建模。Parti、MuseNet、AudioLM、VALL-E、Sora 的 patch tokenizer 都用这个。这个桶相当于桶 1 加上一个学习到的 tokenizer。

## 简要历史

| 年份 | 模型 | 意义 |
|------|-------|------|
| 2013 | VAE (Kingma) | 第一个具有可用训练损失的深度生成模型。 |
| 2014 | GAN (Goodfellow) | 隐式密度、没有似然 — 生成了令人震惊的清晰样本。 |
| 2015 | DRAW, PixelCNN | 顺序图像生成。 |
| 2017 | Glow, RealNVP | 可逆流；随深度给出精确似然。 |
| 2017 | Progressive GAN | 首次生成兆像素级的人脸。 |
| 2019 | StyleGAN / StyleGAN2 | 在特定域的人脸生成仍难以被超越。 |
| 2020 | DDPM (Ho) | 扩散开始变得实用。 |
| 2021 | CLIP, DALL-E 1, VQGAN | 文本到图像进入主流。 |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | 潜在扩散 + 文本条件化成为商品化技术。 |
| 2022 | ControlNet, LoRA | 对预训练扩散模型进行精细控制。 |
| 2023 | SDXL, Midjourney v5, Flow matching | 规模化 + 更好的训练动态。 |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | 视频扩散；flow matching 占优。 |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | 生产级别的视频生成。 |
| 2026 | Consistency + Rectified Flow | 从扩散主干实现一步采样。 |

## 五问分诊

当一篇新的生成模型论文发布时，在读方法部分之前先回答这五个问题。

1. **建模的对象是什么？** 像素、潜变量、离散标记、3D 高斯、网格、波形？  
2. **密度是显式还是隐式？** 他们有没有把 `log p(x)` 写出来？  
3. **采样：一次性还是迭代？** 迭代意味着推理更慢；一次性通常意味着对抗式或蒸馏。  
4. **条件化：无条件、类别、文本、图像、姿态？** 这决定了损失和架构支撑。  
5. **评估：FID、CLIP 分数、IS、人类偏好、任务准确率？** 每种指标都有已知的失败模式（见第 14 课）。

你将在本阶段的每一课中反复回答这五个问题。到最后，它们会成为你的反射动作。

## 构建它

本课的代码是一个轻量级可视化：用三种玩具方法（核密度估计、离散直方图和最近样本的“类 GAN”生成器）从样本中拟合一维高斯混合，这样你可以在一屏上看到显式密度与隐式密度的区别。

运行 `code/main.py`。它从一个双峰高斯混合中绘制 2000 个样本，然后打印：

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

注意：前两种方法可以让你问“这个点有多可能？”，第三种不能。这就是以后每节课都会重要的*显式 vs 隐式* 区分。

## 应用场景

在 2026 年，哪个家族适合哪个任务？

| 任务 | 最佳家族 | 原因 |
|------|---------|------|
| 窄域的逼真人脸 | StyleGAN 2/3 | 仍然是最清晰、推理最快的方案。 |
| 通用文本到图像 | 潜在扩散 + flow matching | SD3、Flux.1、DALL-E 3。 |
| 快速文本到图像 | Rectified flow + 蒸馏 | SDXL-Turbo、SD3-Turbo、LCM。 |
| 文本到视频 | 扩散 Transformer + flow matching | Sora、Veo 2、Kling。 |
| 语音与音乐 | 基于标记的自回归（AudioLM、VALL-E、MusicGen）或 flow matching（AudioCraft 2） | 离散标记的扩展性更经济。 |
| 3D 场景 | Gaussian Splatting 拟合、扩散先验 | 3D-GS 用于重建，扩散用于新视角合成。 |
| 密度估计（只需要估计不需要采样） | Flows | 唯一能给出精确 `log p(x)` 的家族。 |
| 仿真 / 物理 | Flow matching、score SDE | 直线路径、光滑的向量场。 |

## 部署输出

保存为 `outputs/skill-model-chooser.md`。

该技能接收一个任务描述并输出： (1) 应使用哪个家族，(2) 排名的三个开源选项和三个托管选项，(3) 你应注意的可能失败模式，(4) 计算/时间预算估计。

## 练习

1. 简单题。对于下列五个产品，识别其家族和主干：ChatGPT image、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。证据应来自公开技术报告。  
2. 中等题。你将要在明天阅读的一篇论文声称比扩散快 100 倍采样速度。写下三个问题，用来检查该速度提升在有条件化和高分辨率时是否仍然成立。  
3. 困难题。选择一个你关心的领域（例如蛋白质结构、CAD、分子、轨迹）。对该领域当前 SOTA 模型回答五问分诊，并勾画一个更好模型将会改变什么。

## 术语解释

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Generative model | "它能生成新东西" | 学习一个 `p_data(x)` 的采样器，可选地暴露 `log p(x)`。 |
| Explicit density | "你可以评估它" | 模型提供闭式或可解的 `log p(x)`。 |
| Implicit density | "GAN 风格" | 只有采样器 — 无法评估给定点的 `p(x)`。 |
| ELBO | "Evidence lower bound" | `log p(x)` 的一个可处理下界；VAE 和扩散优化它。 |
| Score | "对数密度的梯度" | `∇_x log p(x)`；扩散和 SDE 模型学习这个场。 |
| Manifold hypothesis | "数据存在于一个曲面上" | 高维数据集中在低维流形上；这就是降维有效的原因。 |
| Autoregressive | "预测下一个部分" | 将联合分布分解为条件乘积。 |
| Latent | "压缩编码" | 能够由解码器重构输入的低维表示。 |

## 生产注意：五个家族，五种推理形态

每个家族对应不同的推理服务器成本曲线。生产推理文献将 LLM 推理分为 prefill + decode；相同的分解也适用于这里：

- **自回归（桶 1 和 5）。** 顺序解码主导延迟；KV-cache、连续批处理和投机性解码都可直接应用。  
- **VAE / 扩散 / flow-matching（桶 2 和 4）。** 没有 LLM 意义上的 decode。成本 = `num_steps × step_cost`，其中 `step_cost` 是在全分辨率潜变量上的 transformer 或 U-Net 前向。生产可调节点是步数（DDIM / DPM-Solver / 蒸馏）、批量大小和数值精度（bf16 / fp8 / int4）。  
- **GAN（桶 3）。** 一次前传。没有时间表、没有 KV-cache。TTFT ≈ 总延迟。这就是为什么在窄域 UX 上 StyleGAN 仍然占优。

当你在论文摘要看到“比扩散更快”时，翻译成“更少的步数 × 相同的步成本”或“相同步数 × 更便宜的步成本”。其他都是市场营销。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — GAN 论文。  
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文。  
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM 论文。  
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — 将扩散视为 SDE 的工作。  
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching 论文。  
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3 相关工作。