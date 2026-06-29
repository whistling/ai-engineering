# Conditional GANs & Pix2Pix

> 2014–2017 年的第一个重大突破是能够控制 GAN 的输出。为其附加标签、图像或句子。Pix2Pix 做了图像版本，并且即使到 2026 年，在窄域的图像到图像任务上它仍然优于所有通用的文本到图像模型。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)  
**Time:** ~75 分钟

## 问题描述

无条件 GAN 会采样任意人脸。对演示有用，但在生产中没用。你想要的是：*将素描映射为照片*、*将地图映射为空中照片*、*将白天场景映射为夜间*、*为灰度图像上色*。在这些任务中，你会得到一个输入图像 `x`，必须输出与之有语义对应的 `y`。对于每个 `x` 存在许多合理的 `y`。均方误差会把它们平均成一坨。对抗损失不会，因为“看起来真实”是尖锐的。

条件生成对抗网络（Mirza & Osindero, 2014）将条件 `c` 作为 `G` 和 `D` 的输入。Pix2Pix（Isola 等，2017）对其进行了专门化：条件是完整的输入图像，生成器使用 U-Net，判别器是基于补丁的分类器（PatchGAN），损失是对抗损失 + L1。因为它在配对数据上训练——你恰好拥有所需的信号——这个配方即使在 2026 年也能在窄域图像到图像任务上胜过从零开始的文本到图像模型。

## 概念

![Pix2Pix：U-Net 生成器，PatchGAN 判别器](../assets/pix2pix.svg)

**条件 G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有显式输入噪声——Isola 发现显式噪声会被忽略）。

**条件 D。** `D(x, y) → [0, 1]`。输入是*成对的*(条件, 输出)。这是关键差别：D 必须判断 `y` 是否与 `x` 一致，而不仅仅是 `y` 看起来是否真实。

**U-Net 生成器。** 编码-解码结构，瓶颈处有跳跃连接。对于输入和输出共享低级结构（边缘、轮廓）的任务至关重要。没有跳跃连接时，高频细节会消失。

**PatchGAN 判别器。** D 不输出单一的真/假分数，而是输出一个 `N×N` 网格，每个单元判断约 70×70 像素的感受野。对这些单元取平均。这是一种马尔可夫随机场假设：真实感是局部的。训练更快、参数更少，输出更锐利。

**Loss.**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练并把 G 推向已知目标。与 L2（平均）相比，L1 给出更锐利的边缘（中位数而非均值）。Pix2Pix 的默认 `λ = 100`。

## CycleGAN — 在没有配对数据时

Pix2Pix 需要配对的 `(x, y)` 数据。CycleGAN（Zhu 等，2017）在牺牲额外损失的代价下去掉了这个要求：*循环一致性*损失。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使得 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这使你能在没有配对示例的情况下进行马→斑马、夏→冬的转换。

到 2026 年，无配对图像到图像大多通过扩散模型完成（如 ControlNet、IP-Adapter），但循环一致性的思想在几乎所有无配对域适应论文中仍然存在。

## 实现它

`code/main.py` 实现了一个在 1 维数据上的微小条件 GAN。条件 `c` 是一个类别标签（0 或 1）。任务是：为给定类别生成来自该条件分布的样本。

### 步骤 1：将条件附加到 G 和 D 的输入上

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

独热编码是一种最简单的方式。更大的模型会使用学习到的嵌入、FiLM 调制，或交叉注意力。

### 步骤 2：训练条件模型

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配给定条件下的真实分布，而不是边际分布。

### 步骤 3：验证每类输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 陷阱

- **条件被忽视。** G 学会边缘化，D 也不会惩罚，因为条件信号太弱。修复：更积极地在 D 中加入条件（在早期层而非仅在最后），或使用投影判别器（Miyato & Koyama 2018）。
- **L1 权重太低。** G 浮动到任意真实外观的输出而非忠实重建。对于 Pix2Pix 风格任务，初始 λ≈100。
- **L1 权重太高。** G 产生模糊输出，因为 L1 依然是一个 L_p 范数。训练稳定后逐步退火。
- **D 中的真值泄露。** 将 `(x, y)` 串联作为 D 的输入，而不是仅 `y`。没有 `x`，D 无法检查一致性。
- **每类的模式崩塌（mode collapse）。** 每个类别可以独立崩塌。运行类别条件下的多样性检查。

## 使用场景

2026 年图像到图像任务的主流选择：

| Task | Best approach |
|------|---------------|
| Sketch → photo, same domain, paired data | Pix2Pix / Pix2PixHD（仍然快速且锐利） |
| Sketch → photo, unpaired | 使用带有涂鸦条件模型的 ControlNet |
| Semantic seg → photo | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| Style transfer | 使用 IP-Adapter 或 LoRA 的扩散；GAN 方法为遗留方案 |
| Depth → photo | 在 Stable Diffusion 上的 ControlNet-Depth |
| Super-resolution | Real-ESRGAN（GAN）、ESRGAN-Plus，或 SD-Upscale（扩散） |
| Colorization | ColTran、基于扩散的上色器，或 Pix2Pix-color |
| Daytime → nighttime, seasons, weather | CycleGAN 或 基于 ControlNet 的方法 |

当你有数千个配对样本、任务狭窄且可重复，并且需要低延迟推理时，Pix2Pix 仍然是合适的工具。在通用开域任务上，扩散胜出。

## 部署

保存为 `outputs/skill-img2img-chooser.md`。该 Skill 接受任务描述、数据可用性（配对 vs 非配对，样本数 N）以及延迟/质量预算，然后输出：方法（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据需求、推理成本和评估协议（LPIPS、FID、任务特定评估）。

## 练习

1. **简单。** 修改 `code/main.py`，添加第三个类别。确认 G 仍然将每个类别的噪声映射到正确的模态。
2. **中等。** 在 1 维设置中将 L1 替换为感知式损失（例如使用一个小型冻结的 D 作为特征提取器）。它是否改变了条件分布的锐利度？
3. **困难。** 在 1 维设置中勾画一个 CycleGAN：两个分布、两个生成器、循环损失。展示它如何在没有配对数据的情况下学会在它们之间映射。

## 术语表

| 术语 | 人们如何描述 | 实际含义 |
|------|-------------|---------|
| 条件 GAN (Conditional GAN) | “带标签的 GAN” | `G(z, c), D(x, c)`。两个网络都看到条件。 |
| Pix2Pix | “图像到图像的 GAN” | 配对的 cGAN，使用 U-Net 生成器和 PatchGAN 判别器 + L1 损失。 |
| U-Net | “带跳跃连接的编码-解码器” | 对称卷积网络；跳跃连接保留高频信息。 |
| PatchGAN | “局部真实性分类器” | D 输出每个补丁的评分，而非全局评分。 |
| CycleGAN | “无配对图像翻译” | 两个 G + 循环一致性损失；不需要配对数据。 |
| SPADE | “GauGAN” | 使用语义图对中间激活进行空间自适应归一化；语义图到图像。 |
| FiLM | “按特征线性调制” | 来自条件的按特征仿射变换；廉价的条件方式。 |

## 生产注意：作为延迟受限基线的 Pix2Pix

当你有配对数据且任务狭窄（素描 → 渲染、语义图 → 照片、白天 → 夜间）时，Pix2Pix 的一次性前向推理在延迟上比扩散快一个数量级。常见的生产比较是：

| Path | Steps | Typical latency at 512² on a single L4 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint or SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

在静态批次（每个请求 FLOPs 相同）下，Pix2Pix 在吞吐量上占优。扩散在质量和泛化上获胜。现代做法通常是为狭窄任务发布一个 Pix2Pix 风格的蒸馏模型，并为长尾输入提供扩散回退方案。

## 拓展阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — 条件 GAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 投影判别器。