# Video Generation

> 一张图像是一个 2 维张量。一个视频是一个 3 维张量。理论相同；计算量却大 10-100 倍。OpenAI 的 Sora（2024 年 2 月）证明这是可行的。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 可以在 1080p 生产级别从文本生成视频 —— 而开源权重栈（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）落后约 12 个月。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 8 · 07 (潜在扩散), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)  
**Time:** ~45 分钟

## The Problem

一个 10 秒、24fps 的 1080p 视频包含 240 帧，每帧 1920×1080×3 像素。每个剪辑大约是 ~1.5 GB 的原始数据。像素空间的扩散不可行。你需要：

1. **时空压缩（Spatiotemporal compression）。** 一个对视频（而非单帧）进行编码的 VAE，将其压缩为一系列时空补丁。
2. **时间连贯性（Temporal coherence）。** 多帧之间需要在若干秒内共享内容、光照和物体身份。网络必须建模运动。
3. **计算预算（Compute budget）。** 与图像相比，视频训练使相同模型规模的计算成本高 10-100 倍。
4. **条件控制（Conditioning）。** 文本、图像（首帧）、音频或另一段视频。大多数生产模型都接受这四种条件。

解决方案架构是将 **Diffusion Transformer (DiT)** 应用于时空补丁，基于大规模 (prompt, caption, video) 数据集训练。损失函数与第 06 课相同（扩散损失）。

## The Concept

![视频扩散：分块、DiT、解码](../assets/video-generation.svg)

### Patchify（分块）

用 3D VAE 编码视频（学习的时空压缩）。潜在表示形状为 `[T_latent, H_latent, W_latent, C_latent]`。按 `[t_p, h_p, w_p]` 大小切分成补丁。对于 Sora 样式的模型，`t_p = 1`（按帧补丁）或 `t_p = 2`（每两帧一补丁）。一个 10 秒的 1080p 视频被压缩后大约是 ~20,000-100,000 个补丁。

### 时空 DiT（Spatiotemporal DiT）

Transformer 处理平展的补丁序列。每个补丁有 3 维位置嵌入（时间 + y + x）。注意力通常被因式分解：

- **空间注意力（Spatial attention）** 在每帧的补丁内进行。
- **时间注意力（Temporal attention）** 在相同空间位置跨帧进行。
- **完整 3D 注意力（Full 3D attention）** 代价高 16-100 倍；通常只在低分辨率或研究场景使用。

### 文本条件（Text conditioning）

用大型文本编码器做交叉注意力（Sora 使用 T5-XXL，CogVideoX-5B 使用 T5-XXL）。长提示很重要 —— Sora 的训练集使用 GPT 生成的、平均 200 个 token 的密集重标注（re-captions）。

### 训练

对时空潜变量做标准扩散损失（ε 或 v 预测）。数据来源：网络视频 + ~1 亿个精编剪辑 + 合成文本说明。计算：即便是小规模研究也需要 10,000+ GPU 小时；Sora 级别需 100,000+。

## The 2026 production landscape

| Model | Date | Max duration | Max res | Open weights? | Notable |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | 否 | 首个在规模上展示“世界模拟器”属性的模型 |
| Sora Turbo | 2024-12 | 20s | 1080p | 否 | 生产级 Sora，推理速度快 5 倍 |
| Veo 2 (Google) | 2024-12 | 8s | 4K | 否 | 2025 年画质与物理表现最佳 |
| Veo 3 | 2025 Q3 | 15s | 4K | 否 | 原生音频与更强的摄像机控制 |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | 否 | 2025 Q1 最佳人体运动表现 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | 否 | 专业视频工具生态 |
| Pika 2.0 | 2024-10 | 5s | 1080p | 否 | 最强的人物一致性 |
| CogVideoX (THUDM) | 2024 | 10s | 720p | 是 (2B, 5B) | 首个公开的 5B 级别视频模型 |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | 是 (13B) | 2024 年底的开源 SOTA |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | 是 (10B) | 许可最宽松 |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | 是 | 2025 年中表现最强的开源模型 |

开源权重正在比图像领域更快地缩小差距：HunyuanVideo + WAN 2.2 LoRA 到 2026 年中已能驱动大多数开源工作流。

## Build It

`code/main.py` 模拟了核心的时空 DiT 思路：对一个小的合成视频做分块，为每个补丁添加位置嵌入，并用一个类似 Transformer 的注意力在补丁上对整个序列进行去噪。没有使用 numpy；纯 Python 实现。我们展示即使在 1-D 场景下，当相邻帧的补丁共享去噪器和位置嵌入时，时间连贯性也会出现。

### Step 1: patchify a synthetic 1-D "video"

```python
def make_video(T_frames=8, rng=None):
    # 一个“视频”是遵循平滑轨迹的一维值序列
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Step 2: position embedding per frame

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Step 3: denoiser sees the whole sequence

我们的微小网络不是独立去噪每一帧，而是将所有帧的数值及其位置嵌入拼接起来，并对所有帧的噪声联合预测。

### Step 4: temporal coherence test

训练后采样一个视频。测量帧到帧的增量（delta）。如果模型学到了时间结构，增量会比独立采样每帧时更小。

## Pitfalls

- **独立逐帧采样 = 闪烁（flicker）。** 对每帧单独运行图像扩散会导致输出闪烁，因为每帧的噪声是独立的。视频扩散通过注意力或共享噪声将帧耦合以解决此问题。
- **简单的 3D 注意力 = 内存不足（OOM）。** 在 10 秒 1080p 潜变量上做完整 3D 注意力需要数千亿次运算。应将注意力因式分解为空间 + 时间。
- **数据标注（captioning）比模型规模更重要。** Sora 相对于以往工作的主要提升在于训练时使用了约 10 倍更细致的标注（用 GPT-4 重标注剪辑）。OpenAI 的技术报告对此有明确说明。
- **首帧条件（First-frame conditioning）。** 大多数生产模型也接受图像作为首帧。这就是“图像到视频（image-to-video）”模式；训练包含这种变体。
- **物理漂移（Physics drift）。** 较长剪辑（>10s）会积累细微不一致。滑动窗口生成 + 关键帧锚定有助于缓解。

## Use It

| Use case | 2026 pick |
|----------|-----------|
| 最高质量的文本到视频（托管） | Veo 3 或 Sora |
| 摄像机控制的电影级生成 | Runway Gen-3（带运动画笔） |
| 跨剪辑的人物一致性 | Pika 2.0 或 Kling 2.1 |
| 开源权重、快速微调 | WAN 2.2 + LoRA |
| 图像到视频（I2V） | WAN 2.2-I2V、Kling 2.1 I2V 或 Runway |
| 音频到视频的口型同步 | Veo 3（原生音频）或专用口型同步模型 |
| 视频编辑 | Runway Act-Two、Kling Motion Brush、Flux-Kontext（静帧） |

在质量相当的情况下，单秒视频成本在 2024 到 2026 年间下降了约 20 倍。

## Ship It

保存为 `outputs/skill-video-brief.md`。该 Skill 接受一个视频简报（时长、长宽比、风格、摄像计划、主体一致性、音频），并输出：模型 + 托管方案、提示脚手架（摄像机语言、主体描述、运动描述词）、随机种子 + 可复现协议，以及逐帧 QA 清单。

## Exercises

1. **Easy.** 在 `code/main.py` 中比较（a）独立逐帧采样 和（b）联合序列采样的帧间增量。报告增量的均值和方差。
2. **Medium.** 加入首帧条件：将第 0 帧固定为给定值，然后采样其余帧。测量固定值如何传播。
3. **Hard.** 使用 HuggingFace diffusers 在本地 GPU 上运行 CogVideoX-2B。对一个 6 秒的片段在 720p 下做 20 步推理并计时。分析时空注意力以识别瓶颈。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Video VAE | "3-D VAE" | 将 `(T, H, W, C)` → 时空潜变量的编码器。 |
| Patches | "The tokens" | 固定大小的 3D 潜变量块；DiT 的输入。 |
| Factorized attention | "Spatial + temporal" | 先对空间做注意力，再对时间做注意力；避免完整 3D 注意力。 |
| Image-to-video (I2V) | "Animate this photo" | 模型接受图像 + 文本，输出以该图像为起点的视频。 |
| Keyframe conditioning | "Anchor frames" | 固定特定帧以控制视频走向。 |
| Motion brush | "Directional hint" | UI 输入，用户在图像上绘制运动向量。 |
| Re-captioning | "Dense captions" | 使用 LLM 为训练剪辑生成详细的提示描述。 |
| Flicker | "Temporal artifact" | 帧间不一致；通过耦合去噪修复。 |

## Production note: video latents are a memory-bandwidth problem

一个 10 秒 1080p 剪辑在 24 fps 下是 240 帧 × 1920 × 1080 × 3 ≈ 1.5 GB 的原始像素。经过 4× 的视频 VAE 压缩（`2 × 空间 × 2 × 时间`）后，潜变量大约是每次请求 ~100 MB。若用时空 DiT 在 batch 1 上做 30 个步骤，你每步需要在 HBM 中移动大约 ~3 GB —— 瓶颈是内存带宽，而非 FLOPs。

三个生产级调优点，均来自推理实践：

- **TP（模型并行）跨 DiT。** 文本到视频模型通常 ≥10B 参数。TP=4 跨 4 块 H100 是常态；在 405B 级模型上则用 PP=2 × TP=2。到达 all-reduce 瓶颈之前，TP 对每步延迟大致呈线性降低。
- **帧批处理 = 连续批处理（Frame batching = continuous batching）。** 在生成时，视频在概念上是由相互通过注意力连接的帧组成的一个批次。连续批处理（在途调度）策略：如果模型架构允许滑动窗口生成，则在返回帧 `t` 时可以开始渲染帧 `t+1`。
- **剪辑级预填充缓存（Clip-level prefill cache）。** 对于图像到视频，首帧条件类似于 LLM 的 prompt prefill：计算一次并在后续的时间解码器步骤重用。这实际上是一个视频的 KV-cache。

## Further Reading

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora 技术报告。
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX。
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo。
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1。
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — 2025 年中开源 SOTA。
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — 开创性的视频扩散论文。
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion 的前身。