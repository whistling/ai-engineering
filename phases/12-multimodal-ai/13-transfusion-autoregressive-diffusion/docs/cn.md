# Transfusion: 在同一 Transformer 中同时进行自回归文本与扩散图像

> Chameleon 和 Emu3 把一切押在离散令牌上。它们可行，但量化瓶颈明显——图像质量在连续空间的扩散模型之下出现平台期。Transfusion（Meta，Zhou 等，2024 年 8 月）走相反的路：保持图像为连续表示，完全摒弃 VQ-VAE，用一个 Transformer 和两个损失来训练。文本令牌使用下一个令牌预测（NTP）；图像补丁使用流匹配 / 扩散损失。两个目标优化相同的权重。Stable Diffusion 3（MMDiT）背后的架构是一个近亲。本课阅读 Transfusion 论文，构建一个玩具级的两损失训练器，并追踪使一个 Transformer 同时完成两项任务的注意力掩码。

**Type:** 构建  
**Languages:** Python（标准库，适用于 MNIST 级别的两损失训练器）  
**Prerequisites:** 第12·11阶段（Chameleon）、第8阶段（生成式 AI）  
**Time:** ~180 分钟

## 学习目标

- 连接一个在同一骨干上运行两种损失的 Transformer（文本令牌上的 NTP，图像补丁上的扩散 MSE）。
- 解释为什么对图像补丁使用双向注意力且对文本令牌使用因果注意力是正确的掩码选择。
- 在计算量、质量和代码复杂性上比较 Transfusion 风格（连续图像、扩散损失）与 Chameleon 风格（离散图像、NTP）。
- 说明 MMDiT 的贡献：每个块的模态特定权重、残差流上的联合注意力。

## 问题背景

关于离散与连续图像令牌的争论早于大型语言模型。连续表示（原始像素、VAE 潜变量）能保留细节。离散令牌（VQ 索引）符合 Transformer 的原生词表，但在量化步骤中损失细节。

Chameleon / Emu3 选择了离散：一个损失、一个架构，但图像保真度被分词器质量限制住。

扩散模型选择了连续：图像质量卓越，但通常与 LLM 是分离的模型，需要复杂的噪声调度工程，且与文本生成的整合不够干净。

Transfusion 提出：能否两者兼得？保持图像为连续表示，同时训练一个模型，在同一次梯度步中拼接两个损失。

## 概念

### 两损失架构

一个解码器式的 Transformer 处理一个包含以下内容的序列：

- 文本令牌（离散，来自 BPE 词表）。
- 图像补丁（连续，16x16 像素块通过线性投影到隐藏维 — 与 ViT 编码器的输入相同）。
- 用于标记连续补丁所在位置的 `<image>` 和 `</image>` 标签。

前向计算运行一次。损失针对每个令牌选择两个头之一：

- 对于文本令牌：在词表 logits 头上做标准交叉熵。
- 对于图像补丁：在连续补丁上做扩散损失 —— 预测加在每个补丁上的噪声。

梯度通过共享的 Transformer 主体流动。两个损失同时改进共享权重。

### 注意力掩码：文本因果 + 图像双向

文本令牌必须是因果的——不能让文本令牌关注未来文本，否则教师强制（teacher forcing）会被破坏。图像补丁代表一个快照；它们应该在同一图像块内相互双向关注。

掩码：

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # 对文本采用因果
  OR (i is image and j is image and same_image_block(i, j))   # 在同一图像内双向
  OR (i is text and j is image and j < i_image_end)   # 文本可以关注先前的图像
  OR (i is image and j is text and j < i_image_start)   # 图像可以关注之前的文本
```

在训练与推理时实现为块三角形（block-triangular）掩码。

### 在 Transformer 内部的扩散损失

扩散损失是标准的：在图像补丁上加入噪声，要求模型预测噪声（或等价地预测干净补丁）。Transfusion 的版本使用流匹配 —— 预测从有噪到干净的速度场（velocity field）。

训练过程中：
1. 对每个图像补丁 x0，采样随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（对流匹配使用线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列的文本 NTP 损失一起反向传播。

推理时，生成过程为：
- 文本令牌：标准自回归采样。
- 图像补丁：在条件化文本令牌下进行扩散采样循环（通常 10-30 步）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等，2024 年 3 月）推出了 MMDiT（Multimodal Diffusion Transformer），与 Transfusion 是同期的近亲架构。

MMDiT 的关键差异：

- 每块具有模态特定权重。每个 Transformer 块对文本令牌与图像补丁使用独立的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其余部分按模态区分。
- 修正流（rectified flow）训练。一种特定的流匹配变体，具有已知的采样方式，数学上比 DDPM 更简单。
- 规模。MMDiT 是 SD3 的主干（2B 和 8B 参数变体）。Transfusion 的论文扩展至 7B。

两者收敛到相同的核心想法：一个 Transformer 同时对文本做 NTP、对连续图像表示做扩散。

### 为什么这比 Chameleon 风格更优

连续-扩散与离散-NTP 在图像生成上的质量差距是可测量的。Transfusion 论文报告：

- 在 7B 参数规模下，优于同等规模的 Chameleon 风格模型 3-5 个 FID 点。
- 无需训练分词器——图像编码器更简单（线性投影到隐藏维，等同 ViT 的输入层）。
- 推理时能对图像补丁去噪并行化，而不是像自回归图像令牌那样串行。

缺点：Transfusion 是双损失模型，训练动态更复杂。损失权重需要调优。NTP 与扩散损失之间的步幅不匹配可能导致某一头占主导。

### 下游演化

Janus-Pro（第 12.15 课）通过为理解与生成解耦视觉编码器来改进 Transfusion 的思想——一个用于理解（SigLIP），一个用于生成（VQ），同时共享 Transformer 主体。Show-o（第 12.14 课）将扩散替换为离散扩散（掩码预测）。Transfusion 之后，统一生成家族迅速分化。

到 2026 年，能够输出图像的生产级 VLM（如 Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成通路）几乎肯定使用了这个家族的某个衍生版本。具体细节为专有信息。

## 使用方法

`code/main.py` 在一个微型的类似 MNIST 的问题上构建了一个玩具 Transfusion：

- 文本标题是描述数字（0-9）的短整数序列。
- 图像是 4x4 的字节网格。
- 一对共享权重的线性投影作为 Transformer 的替代；文本上使用 NTP 损失，图像补丁上使用有噪 MSE 损失。
- 训练循环交替处理两个损失，注意力掩码是显式构造的。
- 生成会在一次前向传递中产出文本标题和一个 4x4 图像。

这个 Transformer 是玩具级的。两损失的管道、注意力掩码构造和推理循环才是真正的产物。

## 交付物

本课产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本+图像、文本+音频、文本+视频），它会设计两损失调度（损失权重、掩码形状、共享与模态特定块的划分）并指出实现风险。

## 练习

1. 一个 Transfusion 风格模型训练数据中 70% 为文本令牌、30% 为图像补丁。图像扩散损失的数量级约为文本 NTP 损失的 10 倍。应该如何设置损失权重以平衡它们？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现块三角掩码。标出每个条目的 0 或 1。

3. MMDiT 具有模态特定的 QKV 权重。与 Transfusion 的完全共享 Transformer 相比，这会带来多少参数量开销？在 7B 参数规模时，这样做值得吗？

4. 生成：给定一个文本提示，模型先运行 NTP 生成 50 个令牌，然后遇到 `<image>`，接着对 256 个补丁在 20 个去噪步上运行扩散。总共需要多少次前向传递？

5. 阅读 SD3 论文第 3 节。描述修正流（rectified flow）以及它为什么比 DDPM 在推理步骤上收敛更少。

## 关键词

| 术语 | 人们常说 | 实际含义 |
|------|---------:|---------|
| 两损失训练（Two-loss training） | “NTP + diffusion” | 一个 Transformer 在同一次梯度步中同时优化文本令牌的交叉熵（NTP）与连续图像补丁的 MSE（扩散） |
| 流匹配（Flow matching） | “Rectified flow” | 一种扩散变体，预测从噪声到干净数据的速度场；在数学上比 DDPM 更简单 |
| MMDiT | “Multimodal DiT” | Stable Diffusion 3 的架构：联合注意力、模态特定的 MLP 与归一化 |
| 块三角掩码（Block-triangular mask） | “Causal text + bidirectional image” | 在文本间采用因果性，在图像区域内采用双向的注意力掩码 |
| 连续图像表示（Continuous image representation） | “No VQ” | 将图像补丁表示为实值向量，而非整数码本索引 |
| 速度预测（Velocity prediction / v-parameterization） | “v-parameterization” | 网络输出为噪声与数据之间的速度场，而不是仅仅预测噪声 |

## 延伸阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)