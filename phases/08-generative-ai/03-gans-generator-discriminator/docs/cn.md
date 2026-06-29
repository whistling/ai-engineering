# GANs — Generator vs Discriminator

> Goodfellow 在 2014 年的巧思是完全跳过密度估计。两个网络。一个制造伪样本。一个识别伪样本。它们互相博弈，直到伪样本和真实样本无法区分。按理说这不应该有效。它经常失败。但当它成功时，对于狭窄领域的样本，仍然是文献中锐利度最高的结果之一。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 3 · 02 (反向传播), Phase 3 · 08 (优化器), Phase 8 · 02 (VAE)  
**Time:** ~75 分钟

## 问题

VAE 生成的样本模糊，因为它们的 MSE 解码器损失在贝叶斯意义上对*像素均值*是最优的 —— 多个合理的数字的均值就是一个模糊的数字。你想要的是奖励*逼真性*（plausibility）的损失，而不是像素上接近某一个目标的损失。逼真性的解析解不存在，你必须去学习它。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像和伪造图像；训练一个生成器 `G(z)` 去欺骗 `D`。对 `G` 的损失信号来自 `D` 当前认为哪些特征看起来像真实样本。随着 `G` 的改进，这个信号会更新，`G` 在追逐一个移动的目标。如果两个网络收敛，`G` 就在没有显式写出 `log p(x)` 的情况下学到了数据分布。

这就是对抗训练。数学上是一个极小极大（minimax）博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

到 2026 年，GAN 已不再是生成器的 SOTA（扩散模型和流匹配夺走了这个王冠）。但 StyleGAN 2/3 仍然是迄今为止人脸生成最锐利的模型，GAN 的判别器在扩散训练中常被用作*感知损失*，对抗训练也驱动了那些让实时扩散成为可能的快速一步蒸馏（如 SDXL-Turbo、SD3-Turbo、LCM）。

## 概念

![GAN 训练：生成器和判别器在极小极大博弈中](../assets/gan.svg)

**生成器 `G(z)`。** 将噪声向量 `z ~ N(0, I)` 映射为样本 `x̂`。通常是一个解码器形状的网络（全连接或转置卷积）。

**判别器 `D(x)`。** 将样本映射为标量概率（或得分）。真实 → 1，伪造 → 0。

**损失。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。对真实=1、伪造=0 做二元交叉熵。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和*形式（原始的 `log(1 - D(G(z)))` 在 `D` 自信时会饱和，导致梯度消失）。

**训练循环。** 一步 `D`，一步 `G`。重复。

**为什么有效。** 如果 `G` 完全匹配了 `p_data`，那么 `D` 无法做得比随机好，输出到处都是 0.5；此时 `G` 不再得到梯度。平衡点。

**为什么会失败。** 模式崩溃（`G` 找到一个 `D` 无法区分的模式并一直生成它），梯度消失（`D` 学得太快导致 `log D` 饱和），训练不稳定（学习率、批大小等任何因素都可能破坏平衡）。

## 让 GAN 工作的变种

| Year | Innovation | Fix |
|------|------------|-----|
| 2015 | DCGAN | 卷积/反卷积、批归一化、LeakyReLU — 第一个稳定的架构。 |
| 2017 | WGAN, WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替换 BCE。修复梯度消失问题。 |
| 2017 | Spectral normalization | 对判别器做 Lipschitz 约束。到 2026 年仍在判别器中使用。 |
| 2018 | Progressive GAN | 先训练低分辨率，再逐步增加层。首个百万像素结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化（AdaIN）。固定域照片真实感的最优解。 |
| 2021 | StyleGAN3 | 无混叠、平移等变性 — 到 2026 年仍是人脸生成的金标准。 |
| 2022 | StyleGAN-XL | 条件式、类别感知、更大规模。 |
| 2024 | R3GAN | 通过更强的正则化重新命名；在 1024² 上无需奇技淫巧也能工作。 |

```figure
gan-minimax
```

## 构建它

`code/main.py` 在一维数据上训练一个微型 GAN：两个高斯混合分布。生成器和判别器都是单隐层 MLP。我们手工实现前向、反向与极小极大循环。目标是亲眼观察两种关键失效模式（模式崩溃 + 梯度消失）发生时的行为。

### 步骤 1：非饱和损失

原始 Goodfellow 的 BCE 里 `log(1 - D(G(z)))` 在 `D` 把 `G` 的伪样本判断为伪造且非常自信时会趋于 0。此时对 `G` 的梯度基本为零 —— `G` 无法改进。非饱和形式 `-log D(G(z))` 在 `D` 自信时会发散，给 `G` 一个强信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

（注：代码内的注释已保留为英文说明其数学意义；如需中文注释，请在本地编辑。）

### 步骤 2：每步生成器对应一步判别器

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

为 `G` 生成“新鲜”的伪样本，否则梯度信息将过时。

### 步骤 3：观察模式崩溃

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

典型症状：两个真实模式中的一个停止被生成。判别器不再纠正它，因为它从未在伪样本中出现过。

## 陷阱

- **判别器过强。** 将 D 的学习率降低 2-5 倍，或者给 D 的输入加实例/层噪声。如果 D 的准确率超过 95%，G 基本已死亡。
- **生成器记忆化一个模式。** 给 D 的输入加噪声，使用 minibatch-discriminator 层，或切换到 WGAN-GP。
- **批量归一化泄漏统计。** 真实批 + 伪造批共同通过同一 BN 层会混合它们的统计量。改用实例归一化或谱归一化。
- **玩弄 Inception-score。** FID 和 IS 在样本数量少时噪声很大。评估时使用 ≥10k 个样本。
- **条件任务的一次性采样是个谎言。** 你仍然需要 CFG 比例、截断技巧和重采样才能得到可用输出。

## 使用场景

2026 年的 GAN 技术栈：

| Situation | Pick |
|-----------|------|
| 照片级真实的人脸、固定姿态 | StyleGAN3（最锐利、最小） |
| 二次元 / 风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像翻译 | Pix2Pix / CycleGAN（Phase 8 · 04）或 ControlNet（Phase 8 · 08） |
| 快速一步文本到图像 | 扩散模型的对抗性蒸馏（SDXL-Turbo、SD3-Turbo） |
| 扩散训练中的感知损失 | 在图像裁剪上使用小型 GAN 判别器 |
| 任何多模态、开放式任务 | 不要用 GAN — 使用扩散或流匹配 |

GAN 很锐利但狭窄。一旦你的领域变大 —— 照片、任意文本提示、视频 —— 就切换到扩散模型。对抗技巧作为一个组件（感知损失、蒸馏）仍然存在，但不再作为独立生成器的首选。

## 上线建议

保存 `outputs/skill-gan-debugger.md`。Skill 会读取一次失败的 GAN 运行（损失曲线、样本网格、数据集大小），输出按优先级排序的可能原因、单行修复建议和重跑方案。

## 练习

1. **简单。** 用默认设置运行 `code/main.py`。然后设定 `D_LR = 5 * G_LR` 并重跑。G 的损失会多快塌到常数？
2. **中等。** 将 Goodfellow 的 BCE 换成 WGAN 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并将 D 的权重裁剪到 `[-0.01, 0.01]`。训练更稳定吗？比较真实收敛时间。
3. **困难。** 将一维例子扩展到二维数据（环上 8 个高斯混合）。在 1k、5k、10k 步时跟踪生成器捕获了多少个模式。实现 minibatch discrimination 并重新测量。

## 术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Generator | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| Discriminator | "D" | 分类器 `D: x → [0, 1]`，真实 vs 伪造。 |
| Minimax | "The game" | 对联合目标的 `min_G max_D` 博弈。 |
| Non-saturating loss | "The fix" | 对 G 使用 `-log D(G(z))` 而不是 `log(1 - D(G(z)))`。 |
| Mode collapse | "G memorized one thing" | 尽管数据多样，生成器只产生少数不同的输出。 |
| WGAN | "Wasserstein" | 用 Earth-Mover 距离 + 梯度惩罚替代 BCE；梯度更平滑。 |
| Spectral norm | "Lipschitz trick" | 约束 D 的权重范数以限制其斜率；稳定训练。 |
| StyleGAN | "The one that works" | 映射网络 + AdaIN；在人脸任务中表现最佳，至 2026 年仍然领先。 |

## 生产注记：一次性推理是 GAN 的持久优势

GAN 在开放域生成上已不再在样本质量上占优，但在推理成本上仍具优势。在生产推理术语里，GAN 有：

- **无预填充，无解码阶段。** 单次 `G(z)` 前向传递。TTFT ≈ 总延迟。
- **没有 KV-cache 压力。** 唯一的状态是权重。批大小受激活内存限制，而不是缓存。
- **极其简单的连续批处理。** 每个请求的 FLOPs 都相同，在服务器目标负载下使用静态批通常最优。无需在途调度器。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）在 2026 年成为快速文本到图像的主流技术：它把 20–50 步的扩散流水线折叠成 1–4 次类 GAN 的前向，同时保持扩散基模型的分布。对抗损失作为训练时的旋钮仍保留，用于把慢速生成器变成快速生成器。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 原始 GAN 论文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 第一个稳定架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — 谱归一化。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo。