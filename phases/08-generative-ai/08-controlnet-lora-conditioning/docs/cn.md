# ControlNet、LoRA 与 条件控制

> 仅靠文本是一个笨拙的控制信号。ControlNet 让你克隆预训练的扩散模型，并用深度图、姿态骨架、涂鸦或边缘图来引导它。LoRA 让你通过训练 1000 万参数来微调一个 20 亿参数级别的模型。它们一起把 Stable Diffusion 从玩具变成了 2026 年每个机构都会部署的图像流水线。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — for LoRA foundation)  
**Time:** ~75 分钟

## 问题

像 “a woman in a red dress walking a dog on a busy street” 这样的提示并不能告诉模型狗在*哪里*、女人处于*什么姿态*或街道的*视角*。文本能够固定大约 10% 的图像信息；其余是视觉信息，无法用语言高效描述。

为每种条件信号（姿态、深度、canny、分割）从头训练一个新的条件模型是不可行的。你想保留 26 亿参数的 SDXL 主干（backbone）冻结状态，附加一个读取条件信息的小侧网络，并让它对主干的中间特征施加微调。这就是 ControlNet。

你还想在不重训完整模型的情况下教模型新概念（你的人脸、你的产品、你的风格）。你希望增量小 100 倍。这就是 LoRA —— 插入到现有注意力权重中的低秩适配器。

ControlNet + LoRA + 文本 = 2026 年的工程师工具箱。大多数生产图像流水线在 SDXL / SD3 / Flux 基础上叠加 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## 概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet（Zhang 等，2023）

取一个预训练的 SD。*克隆* U-Net 的编码器半部分。把原始编码器冻结。训练克隆使其接受额外的条件输入（边缘、深度、姿态）。用 *零初始化卷积*（1×1 conv 初始化为零 —— 初始为无操作，学习一个增量）将克隆连接回原始解码器的一侧跳跃连接。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 起始时是恒等的 —— 在训练前没有风险。用标准的扩散损失在 100 万个（提示、条件、图像）三元组上训练。

每种模态的 ControlNet 作为小型侧模型发布（SDXL 约 ~360M，SD 1.5 约 ~70M）。在推理时你可以把它们组合起来：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu 等，2021）

对于模型中的任意线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。对注意力而言，秩 4-16 是常见设置；对大规模微调，秩 64-128。新增参数数目为 `2 · d · r`，而不是 `d²`。以 SDXL 注意力层 `d=640`, `r=16` 为例：每个适配器约 2 万参数而非 41 万 —— 减少约 20x。整模型来看：一个 LoRA 通常为 20-200MB，而基模型约为 5GB。

在推理时你可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是常态。多个 LoRA 可相加堆叠（但需注意它们会以非线性方式相互作用）。

### IP-Adapter（Ye 等，2023）

一个小型适配器，接受一张*图片*作为条件（与文本并行）。使用 CLIP 图像编码器生成图像 token，将其注入到与文本 token 并行的交叉注意力中。每个基模型约 ~20MB。可以在没有 LoRA 的情况下做到“以这张参考图的风格生成图像”。

## 可组合性矩阵

| Tool | What it controls | Size | When to use |
|------|------------------|------|-------------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格化 |
| IP-Adapter | 从参考图获取风格或主体 | 20MB | 文本无法描述的外观 |
| Textual Inversion | 将单个概念作为新 token | 10KB | 旧方法，多被 LoRA 取代 |
| DreamBooth | 对特定主体的完整微调 | 2-5GB | 强身份保持，高计算成本 |
| T2I-Adapter | 轻量级的 ControlNet 替代 | 70MB | 边缘设备、推理预算受限 |

ControlNet ≈ 空间控制。LoRA ≈ 语义控制。两者结合使用。

## 实现

`code/main.py` 在 1 维上模拟了这两种机制：

1. LoRA：一个预训练线性层 `W`。将其冻结。训练一个低秩 `B @ A`，使得 `W + BA` 匹配目标线性层。展示 `r = 1` 足以完美学习一个秩 1 的修正。
2. ControlNet-lite：一个“冻结的基线”预测器和一个读取额外信号的“侧网络”。侧网络的输出由一个可学习的标量门控，初始化为零（我们的零卷积版本）。训练并观察门值逐渐升起。

### 第 1 步：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W被冻结；A、B 是可训练的低秩因子。
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 第 2 步：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate 初始化为 0
h = base(x) + gated
```

在第 0 步输出与 base 完全相同。训练初期 `gate` 会缓慢更新 —— 不会发生灾难性漂移。

## 陷阱

- 过度缩放 LoRA。将 `α` 设为 2 或 3 是常见的 “增强效果” 脸谱化技巧，但会产生过度风格化或损坏的输出。保持 `α ≤ 1.5`。
- ControlNet 权重冲突。将 Pose ControlNet 权重设为 1.0，同时 Depth ControlNet 也设为 1.0 通常会超量。权重总和 ≈ 1.0 是一个安全默认值。
- 在错误基模型上加载 LoRA。SDXL 的 LoRA 在 SD 1.5 上会静默无效，因为注意力维度不匹配。diffusers 0.30+ 会给出警告。
- Textual Inversion 漂移。一个检查点上训练的 token 在另一个检查点上会严重漂移。LoRA 更可移植。
- LoRA 权重合并与存储。你可以把 LoRA 融合进基模型权重以加速推理（不用运行时相加），但你会失去运行时缩放 `α` 的能力。最好同时保留两种版本。

## 使用场景

| Goal | 2026 pipeline |
|------|---------------|
| 复现某品牌的艺术风格 | 在 ~30 张精心挑选的图像上以秩 32 训练 LoRA |
| 在生成图像中放入我的脸 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 指定姿态 + 提示 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知的构图 | ControlNet-Depth + SD3 |
| 参考图 + 提示 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + Inpainting（第 09 课） |
| 快速一步风格化 | 在 SDXL-Turbo 上的 LCM-LoRA |

## 交付

保存 `outputs/skill-sd-toolkit-composer.md`。该 Skill 接受一个任务（输入资产：提示、可选参考图、可选姿态、可选深度、可选涂鸦），输出工具栈、权重和可重现的随机种子协议。

## 练习

1. 简单：在 `code/main.py` 中，将 LoRA 秩 `r` 从 1 变化到 4。哪个秩能使 LoRA 精确匹配一个秩 2 的目标增量？
2. 中等：针对两个目标变换分别训练两个 LoRA。一起加载它们并展示它们的加性相互作用。何时这种相互作用会偏离线性？
3. 困难：使用 diffusers 堆叠：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。随着堆叠权重变化，测量 FID 与提示遵从性的权衡。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ControlNet | "空间控制" | 克隆的编码器 + 零初始化卷积跳跃；读取条件图像。 |
| Zero convolution | "开始时是恒等的" | 1×1 卷积初始化为零；ControlNet 初始为无操作。 |
| LoRA | "低秩适配器" | `W + B @ A`，`r << d`；比完整微调少 100 倍参数。 |
| rank r | "调节旋钮" | LoRA 压缩的秩；典型 4-16，个性化时 64+。 |
| α | "LoRA 强度" | 运行时对 LoRA 增量的缩放系数。 |
| IP-Adapter | "参考图像" | 通过 CLIP 图像 token 的小型图像条件适配器。 |
| DreamBooth | "对主体的完整微调" | 在约 30 张主体照片上训练整个模型。 |
| Textual Inversion | "新 token" | 仅学习一个新词向量；为旧方法，多被取代。 |

## 生产注意：LoRA 热插、ControlNet 通道、多租户服务

一个真正的文本到图像 SaaS 会在同一基准上为数百个 LoRA 和十余个 ControlNet 提供服务。其服务问题很像 LLM 的多租户（生产文献在连续批处理和 LoRAX / S-LoRA 下讨论了 LLM 情形）：

- 热插 LoRA，不要合并。把 `W' = W + α·B·A` 融合进基模型能带来约 3-5% 的每步加速，但会固定 `α` 和基模型。将 LoRA 保持为 VRAM 中的 rank-r 增量；diffusers 提供 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 以实现按请求激活。切换代价为 `2 · d · r · num_layers` 权重 —— MB 级别，子秒级。
- ControlNet 作为第二条注意力通道。克隆的编码器与基线并行运行。两个 ControlNet 各以权重 1.0 激活 = 每步多两个前向传递，而不是合并为一个传递。批处理大小的余量呈二次下降。每个活跃 ControlNet 预算约为 1.5× 的步骤成本。
- 量化后的 LoRA 也可行。如果你把基模型量化（参见第 07 课，Flux 在 8GB 上），LoRA 增量也可干净地量化到 8-bit 或 4-bit。QLoRA 风格的加载让你在 4-bit Flux 基础上堆叠 5-10 个 LoRA 而不会耗尽内存。

Flux 专属提示：Niels 的 Flux-on-8GB 笔记本把基模型量化到 4-bit；在该量化基上以 `pipe.load_lora_weights("user/style-lora")` 方式加载风格 LoRA（`weight_name="pytorch_lora_weights.safetensors"`）依然有效。这是 2026 年大多数 SaaS 机构的配方。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLM；后移植到扩散模型）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — 相较 ControlNet 更轻量的替代方案。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参考流水线文档。