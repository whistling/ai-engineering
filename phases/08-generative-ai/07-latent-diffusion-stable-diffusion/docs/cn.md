# Latent Diffusion & Stable Diffusion

> 在 512×512 像素空间上做扩散模型的计算代价简直是战争罪行。Rombach 等人（2022）注意到生成一张图像并不需要全部 786k 维——只需要足够抓住语义结构，然后用一个单独的解码器重建其余部分。在 VAE 的潜在空间中运行扩散。这个想法就是 Stable Diffusion。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)  
**Time:** ~75 分钟

## 问题

在 512² 的像素空间上做扩散意味着 U-Net 在形状为 `[B, 3, 512, 512]` 的张量上运行。对于一个 5 亿参数的 U-Net，每一步采样大约是 100 GFLOPS。五十步就是每张图像 5 TFLOPS。若在十亿张图像上训练，算力账单会荒谬地高。

大部分 FLOPs 都花在通过网络传递对感知不重要的细节上——高频纹理，这些是一个有损 VAE 可以压缩掉的。Rombach 的想法：先训练一个 VAE（*第一阶段*），冻结它，然后在 4 通道 64×64 的潜在空间中完全运行扩散（*第二阶段*）。同样的 U-Net。像素数量是原来的 1/16。相同质量下 FLOPs 约减少 64 倍。

这就是 Stable Diffusion 的配方。SD 1.x / 2.x 使用一个在 `64×64×4` 潜在上运行的 8.6 亿参数 U-Net，SDXL 在 `128×128×4` 上用了一个 26 亿参数 U-Net，SD3 用 DiT（Diffusion Transformer）并配合 flow matching。Flux.1-dev（Black Forest Labs，2024）发布了一个 120 亿参数的 DiT-MMDiT。它们都运行在同一个两阶段基底上。

## 概念

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**两阶段，分别训练。**

1. **Stage 1 — VAE。** 编码器 `E(x) → z`、解码器 `D(z) → x`。目标压缩比：每个空间轴下采样 8×，并调整通道使得潜在总大小约为像素数量的 1/16。损失 = 重构（L1 + LPIPS 感知）+ KL（权重较小，不会强制 `z` 完全服从高斯，因为我们不需要从 `z` 做精确采样）。通常会加入对抗损失以让解码图像更清晰。

2. **Stage 2 — 在 `z` 上做扩散。** 将 `z = E(x_real)` 视为数据。训练一个 U-Net（或 DiT）去去噪 `z_t`。推理时：通过扩散采样出 `z_0`，然后 `x = D(z_0)`。

**文本条件化。** 还需要两个额外组件。一个冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。以及一个交叉注意力注入：每个 U-Net 模块都接受 `[Q = 图像特征, K = V = 文本 tokens]` 并进行混合。tokens 是文本影响图像的唯一通路。

**损失函数与第 06 课相同。** 相同的 DDPM / flow matching 的噪声 MSE。你只是把数据域换成了潜在空间。

## 架构变体

| Model | Year | Backbone | Latent shape | Text encoder | Params |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | same | 1-4 step sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

趋势：用 DiT（在潜在补丁上运行的 transformer）替换 U-Net，扩展文本编码器（T5 在提示遵从性上优于 CLIP），增加潜在通道（4 → 16 给细节留出更多空间）。

```figure
noise-schedule
```

## 构建它

`code/main.py` 将一个玩具的 1-D “VAE”（示范用的恒等编码器 + 解码器；真实 VAE 会是卷积网络）堆在第 06 课的 DDPM 之上，并添加了带有 classifier-free guidance 的类条件。它展示了相同的扩散损失无论是在原始 1-D 值域还是在编码后的值域上都能工作——这是关键洞察。

### 第 1 步：编码器/解码器

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真实 VAE 有训练好的权重。为了教学目的，这个线性映射足以展示扩散如何在 `z` 上运作，而不关心原始数据域。

### 第 2 步：在 `z` 空间做扩散

与第 06 课相同的 DDPM。网络看到的数据是 `z = E(x)`。采样出 `z_0` 后，用 `D(z_0)` 解码。

### 第 3 步：classifier-free guidance

训练时，10% 的时间丢弃类别标签（替换为空标记）。推理时，计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（最大多样性），`w = 3` = 默认，`w = 7+` = 饱和 / 过于锐利。

### 第 4 步：文本条件化（概念，不是代码）

用一个冻结的文本编码器输出替换类别标签。通过交叉注意力将文本嵌入喂给 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这就是类条件扩散模型与 Stable Diffusion 之间的唯一实质性差别。

## 陷阱

- **VAE 缩放不匹配。** SD 1.x 的 VAE 在编码后会应用一个缩放常数（`scaling_factor ≈ 0.18215`）。忘记这一点会导致 U-Net 在方差严重错误的潜在上训练。每个检查点都会附带这个常数。
- **文本编码器静默出错。** SD3 需要 T5-XXL 且 token 数 ≥128，退回到仅 CLIP 会损失提示忠实度。务必检查 `use_t5=True`，否则提示忠实度会暴跌。
- **混合潜在空间。** SDXL、SD3、Flux 都使用不同的 VAE。在 SDXL 潜在上训练的 LoRA 无法在 SD3 上工作。Hugging Face diffusers 0.30+ 会拒绝加载不匹配的检查点。
- **CFG 过高。** `w > 10` 会产生饱和、油腻的图像，并以牺牲多样性为代价过拟合提示。最佳区间是 `w = 3-7`。
- **负面提示泄露。** 空的负面提示成为空的 null token；非空的负面提示会成为 `ε_uncond`。两者并不相同；有些流水线会默默地默认为 null。

## 使用它

到 2026 年的生产堆栈：

| Target | Recommended backbone |
|--------|----------------------|
| 狭窄领域、有配对数据、从头训练模型 | SDXL 微调（LoRA / 全量）——最快上线 |
| 开域文本到图像、开放权重 | Flux.1-dev（12B，Apache / 非商业）或 SD3.5-Large |
| 最快推理、开放权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 最好提示遵从性、托管服务 | GPT-Image / DALL-E 3（仍然），Midjourney v7，Imagen 4 |
| 编辑工作流 | Flux.1-Kontext（2024 年 12 月）——原生接受图像 + 文本 |
| 研究、基准 | SD 1.5 —— 古老但研究充分 |

## 部署

保存 `outputs/skill-sd-prompter.md`。该技能接收文本提示 + 目标风格并输出：模型 + 检查点、CFG scale、采样器、负面提示、分辨率、可选的 ControlNet/IP-Adapter 组合，以及逐步 QA 清单。

## 练习

1. 简单：运行 `code/main.py`，引导权重 `w ∈ {0, 1, 3, 7, 15}`。记录按类的样本均值。在哪个 `w` 下类均值开始偏离真实数据均值？
2. 中等：将玩具线性编码器换成 tanh-MLP 的编码器/解码器对并加入重构损失。在新的潜在上重新训练扩散。样本质量有变化吗？
3. 困难：用 diffusers 设置真实的 Stable Diffusion 推理：加载 `sdxl-base`，用 30 步 Euler、CFG=7，计时。然后换成 `sdxl-turbo`，4 步，CFG=0。主题相同、质量不同——描述发生了什么以及原因。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| First stage | "The VAE" | 训练好的编码器/解码器对；将 512² 压缩到 64²。 |
| Second stage | "The U-Net" | 在潜在空间上的扩散模型。 |
| CFG | "Guidance scale" | `(1+w)·ε_cond - w·ε_uncond`；调整条件强度。 |
| Null token | "Empty prompt embed" | 用于 `ε_uncond` 的无条件嵌入。 |
| Cross-attention | "How text gets in" | 每个 U-Net 模块将文本 tokens 作为 K 和 V 进行注意。 |
| DiT | "Diffusion Transformer" | 用 transformer 在潜在补丁上替代 U-Net；更易扩展。 |
| MMDiT | "Multi-modal DiT" | SD3 的架构：文本和图像流的联合注意力。 |
| VAE scaling factor | "Magic number" | 将潜在除以约 5.4，使扩散在单位方差空间中运行。 |

## 生产注记：在 8GB 消费级 GPU 上运行 Flux-12B

下面是参考的 Flux 集成，回答了“我只有消费级 GPU，我能部署吗？”的问题。技巧与生产推理文献中对 DiT 的三旋钮配方一致：

1. **分阶段加载。** Flux 有三类网络，它们不需要同时驻留在显存中：T5-XXL 文本编码器（fp32 约 10 GB）、CLIP-L（小）、12B 的 MMDiT、以及 VAE。先编码提示，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。消费级 8GB GPU 同一时间只能放下一个阶段。
2. **通过 bitsandbytes 做 4-bit 量化。** 在 T5 编码器和 DiT 上使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存降至约 1/8，Aritra 的基准显示文本到图像的质量下降可以忽略（笔记本中有链接）。
3. **CPU 交换（CPU offload）。** `pipe.enable_model_cpu_offload()` 会在每次前向过程中自动在 CPU/GPU 之间交换模块。会增加 10-20% 的延迟，但让流水线能跑起来。

内存估算：T5 的 10 GB / 8 ≈ 1.25 GB（量化后），12B 参数 × 0.5 bytes ≈ ~6 GB（量化后 DiT），外加激活。按 stas00 的说法，这是 TP=1 推理的极限端——没有模型并行、最大程度的量化。用于生产你会在 H100 上跑 TP=2 或 TP=4；对于单台开发笔记本，这是可行的配方。

## 深入阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 家族。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 上面所有检查点的参考实现。