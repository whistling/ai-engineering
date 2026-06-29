# StyleGAN

> 大多数生成器会把 `z` 同时注入到每一层。StyleGAN 将其拆开：先把 `z` 映射到中间表征 `w`，然后通过 AdaIN 在每个分辨率层注入 `w`。这一单一改变解开了潜在空间的纠缠，使得真实感面孔在接下来的七年里成为已解决的问题。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**Time:** ~45 分钟

## 问题

一个 DCGAN 通过一堆转置卷积将 `z` 映射到图像。问题在于：`z` 控制一切 — 姿态、光照、身份、背景 — 相互纠缠。沿着 `z` 的某个轴移动，四项都会变化。你无法对模型提出“同一个人，不同姿态”的请求，因为表示并没有以这种方式因式分解。

Karras 等人（2019，NVIDIA）提出：停止把 `z` 直接送入卷积层。用一个学习得到的常量 `4×4×512` 张量作为网络输入。学习一个 8 层 MLP，将 `z ∈ Z → w ∈ W`。通过自适应实例归一化（AdaIN）在每个分辨率注入 `w`：先对每个卷积特征图归一化，然后通过 `w` 的仿射投影来缩放和平移。对每层添加噪声以获得随机细节（皮肤毛孔、发丝）。

结果：`W` 的轴大致上将“高层风格”（姿态、身份）与“细节风格”（光照、颜色）正交分离。你可以通过在低分辨率使用图像 A 的 `w`、在高分辨率使用图像 B 的 `w` 来交换风格。这解锁了编辑、跨域风格化以及整条“StyleGAN 反演”研究线。

## 概念

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**映射网络（Mapping network）。** `f: Z → W`，一个 8 层 MLP。`Z = N(0, I)^512`。`W` 不被强制为高斯分布 — 它学习一个适应数据的形状。

**合成网络（Synthesis network）。** 从一个学习得到的常量 `4×4×512` 开始。每个分辨率块：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。分辨率成倍增长：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。按特征图进行归一化，然后重样式化。这里的“风格”是特征图的一阶和二阶统计量。

**逐层噪声（Per-layer noise）。** 向每个特征图添加单通道高斯噪声，并由每通道的可学习因子缩放。它控制随机细节而不影响全局结构。

**截断技巧（Truncation trick）。** 在推理阶段，采样 `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是多次采样得到的 `w` 的均值。`ψ < 1` 在多样性和质量之间做权衡。几乎所有 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| Version | Year | Innovation |
|---------|------|------------|
| StyleGAN | 2019 | 映射网络 + AdaIN + 噪声 + 逐步生长。 |
| StyleGAN2 | 2020 | 权重去调制（weight demodulation）替代 AdaIN（修复 droplet 伪影）；跳跃/残差架构；路径长度正则化。 |
| StyleGAN3 | 2021 | 无别名卷积 + 等变核；消除纹理粘连像素栅格的问题。 |
| StyleGAN-XL | 2022 | 类条件化，1024²，ImageNet。 |
| R3GAN | 2024 | 以更强的正则化重塑品牌；在 FFHQ-1024 上用 20× 更少参数缩小与扩散模型的差距。 |

截至 2026 年，StyleGAN3 在以下场景仍是默认选择：(a) 窄域真实感高帧率生成，(b) 少样本域自适应（用 100 张图训练新数据集时冻结映射网络），(c) 基于反演的编辑（找到能重建真实照片的 `w`，然后编辑该 `w`）。对于开放域的文本到图像任务，它并不是首选 —— 扩散模型更适合。

## 实现

`code/main.py` 实现了一个 1 维的轻量 “style-GAN lite”：一个映射 MLP，一个从学习到的常量向量出发并用 `w` 导出的 scale/bias 对其进行调制的合成函数，以及逐层噪声。它展示了通过仿射调制注入 `w` 在性能上能匹配或优于将 `z` 连接到生成器输入的方法。

### 步骤 1：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 步骤 2：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

每个特征图的 scale 和 bias 来自于 `w` 的线性投影。

### 步骤 3：逐层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每通道的 Sigma 是可学习的。

## 陷阱

- **Droplet 伪影。** StyleGAN 1 在特征图中产生块状 droplet，因为 AdaIN 将均值置零。StyleGAN 2 的权重去调制通过缩放卷积权重来修复此问题。
- **纹理粘连（Texture sticking）。** StyleGAN 1 和 2 的纹理会跟随像素坐标而非物体坐标（在插值时可见）。StyleGAN 3 的无别名卷积通过窗函数化的 sinc 滤波器解决了这一点。
- **模式覆盖（Mode coverage）。** 截断 ψ < 0.7 看起来更干净，但样本来自一个较窄的圆锥；如果需要多样性，请使用 ψ = 1.0。
- **反演存在信息损失。** 将真实照片反演到 `W` 通常通过优化或编码器完成（e4e、ReStyle、HyperStyle）。多次迭代后结果会漂移。

## 使用场景

| Use case | Approach |
|----------|----------|
| 真实人脸（动漫、人像、窄域） | StyleGAN3 FFHQ / 自定义微调 |
| 从照片进行人脸编辑 | e4e 反演 + StyleSpace / InterFaceGAN 方向 |
| 换脸 / 驱动重建 | StyleGAN + 编码器 + 融合 |
| 头像流水线 | StyleGAN3 配合 ADA 进行少样本微调 |
| 用几张图进行域自适应 | 冻结映射网络，微调合成网络 |
| 多模态或文本条件生成 | 不推荐 — 使用扩散模型 |

对于要求“人的面部照片”的产品级演示，StyleGAN 在推理成本（单次前向，4090 上 <10ms）和相同质量下的清晰度上优于扩散模型。

## 上线要点

保存为 `outputs/skill-stylegan-inversion.md`。这个技能接收一张真实照片并输出：反演方法（e4e / ReStyle / HyperStyle）、预期的潜在损失、编辑预算（在 `W` 空间可以移动多远会开始出现伪影）、以及一份已知可用的编辑方向清单（年龄、表情、姿态）。

## 练习

1. 简单：运行 `code/main.py`，分别设置 `adain_on=True` 和 `adain_on=False`。比较固定潜在与扰动潜在下输出的分布。
2. 中等：实现混合正则化（mixing regularization）：对一个训练批次，计算 `w_a`、`w_b`，并在合成的前半部分使用 `w_a`，后半部分使用 `w_b`。解码器是否学会了解耦风格？
3. 困难：拿一个预训练的 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。通过在有标签样本上训练 SVM 找到控制“微笑”的 `w` 方向；报告在身份开始漂移之前可以推动多远。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Mapping network | "The MLP" | `f: Z → W`，8 层，将潜在几何与数据统计解耦。 |
| W space | "The style space" | 映射网络的输出；大致解耦的风格空间。 |
| AdaIN | "Adaptive instance norm" | 对特征图归一化，然后由 `w` 的投影进行缩放 + 平移。 |
| Truncation trick | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 在多样性与质量之间权衡。 |
| Path-length regularization | "PL reg" | 惩罚图像随 `w` 单位变化的过大变化；使 `W` 更平滑。 |
| Weight demodulation | "The StyleGAN2 fix" | 对卷积权重进行归一化而不是激活；消除 droplet 伪影。 |
| Alias-free | "StyleGAN3's trick" | 窗函数化的 sinc 滤波器；消除纹理粘连像素栅格问题。 |
| Inversion | "Find w for a real image" | 优化或编码 `x → w`，使得 `G(w) ≈ x`。 |

## 生产说明：为什么 StyleGAN 在 2026 年仍在使用

在 4090 上，StyleGAN3 可以在 10 ms 以下生成一个 1024² 的 FFHQ 面孔 —— `num_steps = 1`，没有 VAE 解码，也没有交叉注意力步。在生产层面，这就是任何图像生成器的最低延迟。相同分辨率下，一个 50 步的 SDXL + VAE 解码流水线大约需要 ~3 秒。那是一个大约 300× 的差距，对于窄域产品（头像服务、身份证件流水线、图库人脸生成）来说在总体拥有成本（TCO）上胜出。

两个运营后果：

- 无需调度器、无需批处理器。以目标占用率的静态批量最优。连续批处理（对 LLM 和扩散模型至关重要）没有任何益处，因为每个请求消耗相同的 FLOPs。
- 截断 ψ 是安全旋钮。在高峰负载下降低 ψ（ψ < 0.7）可以让样本来自映射网络范围的窄锥。对高级用户提高 ψ 以获得更高多样性。

## 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 反演。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代极简 GAN 实践。