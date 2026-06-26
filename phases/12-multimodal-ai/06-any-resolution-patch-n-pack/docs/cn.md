# Any-Resolution Vision: Patch-n'-Pack and NaFlex

> 真实图像不是 224x224 的正方形。收据是 9:16，图表是 16:9，医学扫描可能是 4096x4096，手机截图是 9:19.5。2024 年之前的 VLM 答案——将所有图像缩放到固定正方形——丢弃了使 OCR、文档理解和高分辨率场景解析生效的关键信号。NaViT（Google，2023）展示了可以使用块对角掩码将可变分辨率的 patch 打包到单个 transformer 批次中。Qwen2-VL 的 M-RoPE（2024）完全舍弃了绝对位置表。LLaVA-NeXT 的 AnyRes 将高分辨率图像切分为基础 + 子图像。SigLIP 2 的 NaFlex 变体（2025）现在是希望用单个检查点服务所有纵横比的开源 VLM 的默认编码器。本课程端到端实现了 patch-n'-pack。

**Type:** 构建  
**Languages:** Python（标准库，patch 打包器 + 块对角注意力掩码）  
**Prerequisites:** Phase 12 · 01（ViT patches）、Phase 12 · 05（LLaVA）  
**Time:** ~120 分钟

## 学习目标

- 将一批可变分辨率图像的 patch 打包成一个序列，并构建块对角注意力掩码。  
- 在 AnyRes（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间为给定任务选择合适策略。  
- 在不缩放的情况下计算 OCR、图表和摄影的 token 预算。  
- 说出将图像缩放到正方形的三种失败模式：文字被压扁、内容被裁剪、为填充浪费 token。

## 问题陈述

Transformer 期望输入为序列。一个批次是多个相同长度序列的堆栈。如果你的图像都是 224x224，你每次都得到 196 个 patch token，无需填充，工作完成。在训练时用 224，推理时也用 224，从此再也不用考虑分辨率了。

现实世界不配合。文档是纵向（8.5x11 英寸，约 2:3）。图表截图是横向（16:9）。收据又高又窄（约 1:3）。医学影像往往是 2048x2048 或更大。移动设备截图是 1170x2532（约 0.46:1）。

2024 年前的三种选项及其弊端：

1. 缩放到固定正方形（224x224 或 336x336）。变形会扭曲文字和面部。下采样会破坏图表标签和 OCR 内容。这是 LLaVA-1.5 之前的常规做法。  
2. 裁剪到固定纵横比。你会丢弃大部分图像，并且选择裁剪位置本身又是一个视觉问题。  
3. 填充到最长边。解决了变形问题，但会在纵向图像上浪费 50% 以上的 token。大量填充 token 会带来二次方级的注意力开销。

2024–2025 年的答案：让 transformer 在图像的原生分辨率下直接吃 patch，并想办法将异构批次打包成一个序列而不浪费计算。

## 概念

### NaViT 和 patch-n'-pack

NaViT（Dehghani 等，2023）证明了该方法在规模化下是可行的。思路很机械：

1. 对批次中每张图像，按照选定的 patch 大小（例如 14）计算其原生 patch 网格。  
2. 将每张图像的 patch 展平为各自的可变长度序列。  
3. 将所有图像的 patch 串联为批次的一个长序列。  
4. 构建块对角注意力掩码，使得图像 A 的 patch 仅在图像 A 内部相互注意。  
5. 保留每个 patch 的位置编码（二维 RoPE 或分数位置嵌入）。

例如，三张图像分别为 336x336（576 个 token）、224x224（256 个 token）和 448x336（768 个 token），打包后变为一个 1600-token 的序列，并配有 1600x1600 的块对角掩码。无填充、无浪费计算。Transformer 可以处理任意纵横比。

NaViT 还引入了训练期间的分数 patch 丢弃——在批次内随机丢弃 50% 的 patch——既作为正则化又加速训练。SigLIP 2 继承了这一点。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实的替代方案。当你有高分辨率图像但编码器是固定的（例如 CLIP 或 SigLIP 在 336 分辨率），可以按如下方式平衡：

1. 从预定义布局集合中选一个网格（1x1、1x2、2x1、1x3、3x1、2x2 等）以匹配图像纵横比。  
2. 将整张图像切成该网格；每个 tile 都成为一个 336x336 的裁剪。  
3. 额外产生一张缩略图：整个图像重采样到 336x336，作为全局上下文 token。  
4. 将每个 tile 通过冻结的 336 编码器编码，拼接 tile token + 缩略图 token。

例如对 672x672 图像用 2x2 网格加缩略图：4 * 576 + 576 = 2880 个视觉 token。代价高但有效——LLM 能同时看到局部细节和全局语境。

当你的编码器是冻结且只支持一个分辨率时，AnyRes 是首选路径。它会在大图像上激增 token（例如 1344x1344 的 4x4 网格会产生 9216 + 576 ≈ 9800 token，几乎占满一个 8k LLM 的上下文）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 提出了 Multimodal Rotary Position Embedding。与 NaViT 的分数位置或 AnyRes 的切片+缩略图不同，每个 patch 带有一个三维位置（时间、行、高度、列、宽度 —— 文中以 t,r,c 表示）。查询/键的旋转处理任意 H、W 和时间长度。

M-RoPE 原生支持动态分辨率且无需重训。在推理时，你可以输入任何 HxW 图像，patch 嵌入器会产生 H/14 x W/14 的 token，每个 token 获得它的 (t=0, r=row, c=col) 位置，RoPE 用恰当的频率旋转注意力，完成。Qwen2.5-VL 和 Qwen3-VL 继续沿用该思路。InternVL3 的 V2PE 也是类似的按模态可变编码。

不同于 AnyRes，M-RoPE 在原生分辨率下的 token 数为 O(H x W / P^2)——没有乘性的 tile 额外开销。不同于 NaViT，它仍然期望一次前向传递处理单张图像。跨分辨率批处理仍然需要在其之上使用 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 检出点的 native-flex 模式。单个模型在推理时支持多个序列长度（256、729、1024 token）。训练时内部采用 NaViT 风格的 patch-n'-pack，并对每个 patch 使用绝对分数位置。卖点是：一个检查点，根据任务在推理时选择 token 预算，无需重训。

语义任务（分类、检索）用 256 token。OCR 或图表理解用 1024 token。无需重训。

### 打包掩码

块对角掩码是大多数实现容易出错的地方。对于一个长度为 `N_total` 的打包序列，覆盖了图像 i=0..B-1，且每张图像的长度为 `n_i`，掩码 `M` 的形状是 `(N_total, N_total)`，当且仅当两个索引都落在同一张图像的块内时为 1，否则为 0。可以从累积长度列表构建它：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

在 PyTorch 中这可以用一行实现（`torch.block_diag`）或者显式的 gather。FlashAttention 的可变长度路径（`cu_seqlens`）完全跳过了密集掩码，而是直接使用累积长度张量在序列内部实现注意力——对于典型批次比密集掩码快约 10 倍。

### Token 预算

按任务选择策略：

- OCR / 文档：1024–4096 token。使用 SigLIP 2 的 NaFlex（1024）或 AnyRes 的 3x3 + 缩略图。  
- 图表和 UI：729–1024 token，原生分辨率在 384–448 区间。使用 Qwen2.5-VL 的动态分辨率并设置像素上限。  
- 自然照片：256–576 token 足够。下游 LLM 能看到足够的信息。在内容密集处付费以获得更多 token。  
- 视频：每帧在空间池化后为 64–128 token，帧率为 2–8 FPS。课程 12.17 涵盖此内容。

2026 年的生产规则：为每个任务选择一个最大像素上限，在不超过该上限的前提下按原生纵横比编码，将批次打包并跳过填充。Qwen2.5-VL 暴露了 `min_pixels` 和 `max_pixels` 以精确控制这一调节项。

## 使用方法

`code/main.py` 实现了对具有整数像素坐标的异构图像批次的 patch-n'-pack。它会：

- 接受一组 (H, W) 图像尺寸列表。  
- 在 patch 大小为 14 时计算每张图像的原生 patch 序列长度。  
- 将它们打包成总长度为 `sum(n_i)` 的序列。  
- 构建块对角注意力掩码（为清晰起见使用密集表示）。  
- 比较打包后的代价与方形缩放和 AnyRes 切片的代价。  
- 为混合批次（收据、图表、截图、照片）打印 token 预算表格。

运行它。输出的数字就是每个 2026 年开源 VLM 都在使用 patch-n'-pack 的原因。

## 上线交付

本课程会生成 `outputs/skill-resolution-budget-planner.md`。给定一个混合纵横比工作负载（OCR、图表、照片、视频帧）和一个总 token 预算，它会为每个请求选择合适策略（NaFlex、AnyRes、M-RoPE 或 固定正方形）并输出每次请求的配置。在为产品为 VLM 规模化时使用该 skill——它能防止沉默的 10 倍 token 爆炸杀死延迟预算。

## 练习

1. 一张收据是 600x1500（约 1:2.5）。在 patch 大小为 14 时，有多少原生分辨率 token？缩放到 336 后有多少 token？在实践中哪种方式更容易丢失 OCR 精度？  

2. 为长度为 256、576、729、1024 的 4 张图像构建块对角掩码。验证注意力矩阵是 2585x2585 并且恰好有 `256^2 + 576^2 + 729^2 + 1024^2` 个非零条目。  

3. 对 1792x896 图像以 patch=14 比较三种方式： (a) 缩放到 336 后编码；(b) AnyRes 2x1 + 缩略图；(c) M-RoPE 在原生分辨率编码。哪一种使用的 token 最少？哪一种保留的细节最多？  

4. 实现分数 patch 丢弃：给定一个已打包序列，均匀随机丢弃 50% 的 token，并相应更新块对角掩码。测量掩码稀疏性的变化。  

5. 阅读 Qwen2-VL 论文的第 3.2 节（arXiv:2409.12191）。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么以及为何两个边界都重要。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Patch-n'-pack | "NaViT-style packing" | 将来自不同图像的可变长度 patch 序列串联到同一批次维度 |
| Block-diagonal mask | "Packing mask" | 限制每张图像的 patch 仅在自身内部相互注意的注意力掩码 |
| AnyRes | "LLaVA-NeXT tiling" | 将高分辨率图像分割为固定大小的网格 tile 加上全局缩略图；对每个 tile 使用固定编码器编码 |
| NaFlex | "SigLIP 2 native-flex" | 单个 SigLIP 2 检查点在推理时无需重训即可服务 256/729/1024 token 预算 |
| M-RoPE | "Multimodal RoPE" | 三维 Rotary 位置编码（时间、行、列），支持任意 H、W、T 而无需位置表 |
| cu_seqlens | "FlashAttention packing" | FlashAttention 的可变长度路径使用的累积长度张量，代替密集的块对角掩码 |
| min_pixels / max_pixels | "Resolution bounds" | Qwen2.5-VL 的每次请求调节项，用来对非常小或非常大的输入限制 token 数 |
| Visual token budget | "How many tokens per image" | 每张图像发出的 patch token 粗略计数；决定 LLM 的 prompt 预算和注意力开销 |

## 延伸阅读

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)  
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)  
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)  
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)  
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)