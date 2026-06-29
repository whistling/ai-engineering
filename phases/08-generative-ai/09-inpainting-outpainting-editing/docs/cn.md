# Inpainting, Outpainting & Image Editing

> Text-to-image makes new things. Inpainting fixes old ones. In production, 70% of billable image work is editing — swap a background, remove a logo, extend the canvas, regenerate a hand. Inpainting is where diffusion earns its keep.

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 8 · 07（潜在扩散）, Phase 8 · 08（ControlNet & LoRA）  
**Time:** ~75 分钟

## The Problem

客户发来一张完美的产品照片，但背景中有一个分散注意力的标志。你想把标志擦掉，并让其余像素保持像素级一致。你不能从头跑一次 text-to-image —— 结果会有不同的颜色、不同的光照、不同的产品角度。你希望只重新生成被遮罩的区域，并且期望重建尊重周围上下文。

这就是 inpainting。变体包括：

- **Inpainting（修补/局部重建）**：在遮罩内部重建，保持遮罩外的像素不变。
- **Outpainting（扩画/外拓）**：在遮罩外（或画布之外）重建，保持遮罩内不变。
- **Image editing（图像编辑）**：重新生成整张图像，但保持对原图的语义或结构一致性（如 SDEdit、InstructPix2Pix）。

到 2026 年，几乎所有的扩散流水线都带有 inpainting 模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit 都遵循相同的原理。

## The Concept

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### The naive approach (and why it's wrong)

用一个遮罩去跑标准的 text-to-image。在每一步采样时，将未遮罩区域的噪声潜码替换为前向扩散过的干净图像。这种方法能工作……但效果很差。边界伪影会渗出，因为模型不知道被遮罩区域内部的内容。

### The proper inpainting model

训练一个修改过的 U-Net，使其接受 9 通道输入而不是 4 通道：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是 VAE 编码的源图像副本加上单通道的遮罩。在训练时，随机遮罩图像的某些区域，并训练模型只去去噪被遮罩的区域，同时把未遮罩区域作为干净的条件信号提供。在推理时，模型可以“看到”遮罩区域的周围，从而生成连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种 9 通道（或等价）输入。Diffusers 包含 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit (Meng et al., 2022) — free editing

对源图像加入噪声直到某个中间时间点 `t`，然后从 `t` 反向采样到 0，同时使用新的提示词。不需要重训练。起始时间 `t` 的选择在保真度与创意自由之间做权衡：

- `t/T = 0.3` → 几乎与源图像相同，仅有小的风格变化
- `t/T = 0.6` → 中度编辑，保留粗糙结构
- `t/T = 0.9` → 几乎从噪声生成，最小程度保留源图像

### InstructPix2Pix (Brooks et al., 2023)

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。推理时同时以输入图像和文本指令为条件（例如“改成日落”或“加一条龙”）。有两个 CFG 比例：图像尺度和文本尺度。

### RePaint (Lugmayr et al., 2022)

保留标准的无条件扩散模型。在每个反向步骤中进行重采样 —— 偶尔跳回更噪的状态并重新生成。能避免边界伪影。当你没有训练过的 inpainting 模型时使用该方法。

## Build It

`code/main.py` 实现了一个在 5 维数据上的玩具 1-D inpainting 方案。我们在由两个簇组成的混合数据上训练一个 DDPM：每个样本是来自两个簇之一的 5 个浮点数。在推理时，我们“遮罩”这 5 维中的 2 个维度，在每一步将未遮罩的 3 个维度注入干净图像的前向噪声版本，并只重建被遮罩的维度。

### Step 1: 5-D DDPM data

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Step 2: train denoiser over all 5 dims

标准 DDPM。网络对 5 维噪声输入输出 5 维噪声预测。

### Step 3: at inference, mask-aware reverse

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # 将未遮罩的维度替换为对干净源的最新加噪版本
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...然后在 x_t 上运行正常的反向一步
```

这是天真的方法，在玩具 1-D 数据上可以工作。真正的图像 inpainting 使用 9 通道输入，因为纹理连贯性更为重要。

### Step 4: outpainting

Outpainting 就是将遮罩取反后的 inpainting：遮罩新（之前不存在的）画布区域，保留其余部分。训练目标完全相同。

## Pitfalls

- **Seams（接缝/缝隙）**。天真的方法会留下可见边界，因为梯度信息无法穿过遮罩流动。修复方法：对遮罩进行膨胀 8–16 像素，或使用真正的 inpainting 模型。
- **Mask leakage（遮罩泄露）**。如果用于条件的未遮罩区域质量低或带噪声，会污染遮罩内的生成。稍微去噪或模糊一下。
- **CFG interacts with mask size（CFG 与遮罩大小交互）**。对小遮罩使用高 CFG 会导致过度饱和的补丁。对小修改降低 CFG。
- **SDEdit fidelity cliff（SDEdit 保真度悬崖）**。从 `t/T = 0.5` 到 `t/T = 0.6` 可能会丢失主体身份。需要扫参并保存检查点。
- **Prompt mismatch（提示词不匹配）**。提示词应描述整张图像，而不仅仅是新增内容。写“a cat sitting on a chair”（一只坐在椅子上的猫），而不是仅写“a cat”。

## Use It

| Task | Pipeline |
|------|----------|
| Remove object, small mask | SD-Inpaint or Flux-Fill，标准提示词 |
| Replace sky | SD-Inpaint + "blue sky at sunset" |
| Extend canvas | SDXL outpaint mode（8px 羽化）或使用 Flux-Fill 的 outpaint 遮罩 |
| Regenerate hand / face | SD-Inpaint，使用重新描述主体的提示词 + ControlNet-Openpose |
| Change style of one region | 在遮罩区域使用 SDEdit，`t/T=0.5` |
| "Make it sunset" | InstructPix2Pix 或 Flux-Kontext |
| Background replacement | 使用 SAM 生成遮罩 → SD-Inpaint |
| Ultra-high-fidelity | Flux-Fill 或 托管的 GPT-Image 处理最困难的案例 |

SAM（Meta 的 Segment Anything，2023）+ 扩散 inpaint 是 2026 年的背景移除流水线。SAM 2（2024）支持视频。

## Ship It

保存为 `outputs/skill-editing-pipeline.md`。该技能接受原始图像 + 编辑描述 + 可选遮罩（或 SAM 提示），输出：遮罩生成方法、基础模型、CFG 比例（图像 + 文本）、SDEdit-t 或 inpainting 模式，以及 QA 检查表。

## Exercises

1. **Easy.** 在 `code/main.py` 中，将被遮罩维度的比例从 0.2 变化到 0.8。在哪个比例下，遮罩维度的 inpaint 质量（被遮罩维度的残差）等同于无条件生成？
2. **Medium.** 实现 RePaint：在每第 10 个反向步骤，回跳 5 个步骤（添加噪声）并重新去噪。测量它是否减少遮罩边缘的边界残差。
3. **Hard.** 使用 Hugging Face diffusers 进行对比：在 20 个人脸重生成任务上比较 SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill。分别评分姿态遵从性和身份保持度。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Inpainting | "Fill the hole" | 在遮罩内部填充；保持遮罩外像素不变。 |
| Outpainting | "Extend the canvas" | 在画布之外或遮罩外生成；保持遮罩内不变。 |
| 9-channel U-Net | "Proper inpainting model" | U-Net 以 `noisy \| encoded-source \| mask` 作为输入。 |
| SDEdit | "Img2img with noise level" | 对源图像加噪到时间 `t`，用新提示词去噪。 |
| InstructPix2Pix | "Text-only edits" | 在（图像、指令、输出）三元组上微调的扩散模型。 |
| RePaint | "No retraining" | 在反向过程中周期性地重新加噪以减少接缝。 |
| SAM | "Segment Anything" | 通过点击或框生成遮罩；与 inpaint 配合使用。 |
| Flux-Kontext | "Edit with context" | Flux 的变体，接受参考图像 + 指令进行编辑。 |

## Production note: edit pipelines are latency-sensitive

用户在编辑图像时期望子 5 秒的往返时间。一个 1024²、30 步的 SDXL-Inpaint 在一块 L4 上大约需要 3–4 秒，加上 SAM 遮罩生成（~200 ms）和 VAE 编解码（合计 ~500 ms）。在生产考量中，这更多受 TTFT（time-to-first-token）约束，而不是吞吐量约束 —— 单批 1、低并发，尽量最小化每个阶段的时间：

- **SAM-H is the slow one.** SAM-H 在 1024² 下约 200 ms；SAM-ViT-B 在有轻微质量损失的情况下约 40 ms。SAM 2（视频）增加了时间开销；不要将其用于单张图像编辑。
- **Skip the encode when possible.** `pipe.image_processor.preprocess(img)` 会编码为潜码。如果你有之前生成的潜码（在迭代编辑 UI 中常见），直接通过 `latents=...` 传入以跳过一次 VAE 编码。
- **Mask dilation matters for throughput too.** 小遮罩意味着大部分 U-Net 前向计算被浪费（未遮罩像素反正会被固定）。`diffusers` 的 `StableDiffusionInpaintPipeline` 无论如何都会跑完整个 U-Net；只有真正的 9 通道 inpaint 变体才能利用遮罩做计算优化。
- **Flux-Kontext is the 2025 answer.** 对 `(source_image, instruction)` 做一次前向就完成编辑 —— 无需单独遮罩、无需 SDEdit 的噪声扫参。在一块 H100 上大约 1.5 秒完成。架构教训：合并阶段可以带来显著加速。

## Further Reading

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 无需训练的 inpainting 方法。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit 方法论文。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 基于文本指令的图像编辑。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，遮罩生成方法来源。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 支持图像与视频的 SAM 2。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 基于注意力的细粒度编辑方法。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 年相关工具。