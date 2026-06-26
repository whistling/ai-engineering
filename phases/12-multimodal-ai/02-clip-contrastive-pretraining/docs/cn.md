# CLIP 和对比视觉-语言预训练

> OpenAI 的 CLIP (2021) 证明了一个足够重要的想法，足以主导接下来的五年：仅用嘈杂的网络图像-说明文本对和对比损失，就将图像编码器和文本编码器对齐到同一向量空间。没有监督标签。4 亿对样本。得到的嵌入空间可以做零样本分类、图像-文本检索，并作为每个 2026 年 VLM 的视觉塔。SigLIP 2 (2025) 用 sigmoid 取代 softmax，并以更低成本将规模推过 CLIP。本课从 InfoNCE 推导到 sigmoid 成对损失的数学，并用标准库 Python 构建训练步骤。

**Type:** 构建  
**Languages:** Python（stdlib、InfoNCE + sigmoid 损失的实现）  
**Prerequisites:** Phase 12 · 01（ViT patches），Phase 7（Transformers）  
**Time:** ~180 分钟

## 学习目标

- 从互信息推导 InfoNCE 损失并实现数值稳定的向量化版本。
- 解释为什么 sigmoid 成对损失（SigLIP）能在没有 softmax all-gather 开销的情况下扩展到 32768+ 的批量大小。
- 通过构建文本模板（`a photo of a {class}`）并对余弦相似度取 argmax 来运行 ImageNet 的零样本分类。
- 说出 CLIP / SigLIP 预训练给你的四个调节杆：批量大小、温度、提示模板、数据质量。

## 问题陈述

在 CLIP 出现之前，视觉是监督训练的。收集带标签的数据集（ImageNet：120 万图像，1000 类），训练 CNN，上线发布。标签很昂贵，标签带有标注者一致性的偏差，并且标签在没有微调的情况下无法迁移到新任务。

网络上的图像-说明文本对超过十亿对，且免费。一个金毛的图片配上 alt 文本 “my dog Max in the park” 就包含监督信号——文本描述了图像。问题是：能否把这些数据变成有用的训练信号？

CLIP 的答案是：把图像-说明对当作匹配任务。给定一个批次的 N 个图像和 N 个说明，学习把每张图像和它自己的说明在 N-1 个干扰项中匹配出来。监督就是“这两个属于同一对；其他 N-1 个不是”。没有类别标签。没有人工标注。只有对比损失。

得到的嵌入空间能做的比 CLIP 训练时的任务更多。ImageNet 零样本有效是因为 “a photo of a cat” 的嵌入会靠近那些从未被明确标注为猫的猫图像。这一押注催生了每个 2026 年的 VLM。

## 概念

### 双塔编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，输出每张图像的 D 维向量。
- 文本编码器 `g`：小型 transformer，输出每个说明的 D 维向量。

两个塔都将输出归一化到单位长度。相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`，因为两者都是单位范数。

对于 N 对（图像，说明）的批次，构造形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是一个可学习的温度（CLIP 初始化为 0.07；以对数空间学习）。

### InfoNCE 损失

CLIP 使用行与列上的对称交叉熵：

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。交叉熵中的 softmax 强制每张图像比批次中的每个其他说明更匹配它自己的说明。所有其他批内样本都是“负样本”。更大的批次 = 更多负样本 = 更强的信号。CLIP 在批量 32k 上训练；规模很重要。

### 温度

`tau` 控制 softmax 的尖锐度。低 `tau` → 分布更尖锐，产生困难负样本挖掘的效果。高 `tau` → 分布更平滑，所有样本都贡献梯度。CLIP 学习 `log(1/tau)`，并做裁剪以防止塌陷。SigLIP 2 固定初始 `tau` 并使用可学习偏置代替。

### 为什么 sigmoid 更易扩展（SigLIP）

Softmax 需要整个相似度矩阵保持同步。在分布式训练中，你必须对每个副本做 all-gather，将所有嵌入收集到每个副本，然后再做 softmax。这在通信上对全量世界规模是二次的。

SigLIP 用逐元素 sigmoid 代替 softmax：对每对 `(i, j)`，损失变成二分类“这对是匹配吗？”的二元交叉熵。正类标签为对角线，其他全部为负。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当且仅当 `i == j`，否则为 0。每对的损失相互独立。不需要 all-gather。每张 GPU 计算其本地块并求和。SigLIP 2 可以廉价地扩展到批量 32k–512k，而 CLIP 则需要成比例更多的通信。

### 零样本分类

给定 N 个类别名称，对每个类别构建一个文本模板：

```
"a photo of a {class}"
```

用文本编码器对每个模板做嵌入。用图像编码器对图片做嵌入。对余弦相似度取 argmax 即为预测类别。对目标类别没有任何训练。

提示模板很重要。CLIP 原论文对每个类别使用了 80 个模板（普通、艺术、照片、绘画等）并对嵌入求平均，提升 ~3 个 ImageNet 点。现代用法通常选用一两个模板。

### 线性探测与微调

零样本只是一个基线。线性探测（在冻结的 CLIP 特征上只训练一个线性层）在同域任务上通常优于零样本。完整微调在同域上胜过线性探测，但可能损害零样本的迁移能力。三种方案各有权衡。

### SigLIP 2：NaFlex 与密集特征

SigLIP 2 (2025) 增加了：
- NaFlex：单模型处理可变长宽比和分辨率。
- 更好的密集特征用于分割和深度估计，目标是作为 VLM 中冻结的骨干网络使用。
- 多语言：在 100+ 语言上训练，而 CLIP 原为英语。
- 参数规模到 1B，而 CLIP 在 400M 处达到上限。

在 2026 年的开源 VLM 中，SigLIP 2 SO400m/14 是默认的视觉塔。CLIP 仍是纯图像-文本检索时的默认选择，尤其当 LAION-2B 的训练分布符合你的查询模式时。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同的想法，规模 18 亿对，90% 噪声。证明了噪声数据能扩展。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开源复现，多种规模，是开源检查点的首选。EVA-CLIP：从掩码图像建模初始化；是 VLM 强大的骨干。BASIC：Google 的 CLIP+ALIGN 混合。都是同一家族，数据和调优不同。

### 零样本上限

CLIP 类模型在 ImageNet 零样本上约束在 ~76%（CLIP-G、OpenCLIP-G）。要提升需要更大的数据（SigLIP 2 可达到 80%+）或架构变化（监督头、更多参数）。基准趋于饱和；真正的价值在于下游 VLM 可利用的嵌入空间。

```figure
multimodal-fusion
```

## 使用方法

`code/main.py` 实现了：

1. 一个玩具双塔编码器（基于哈希的图像特征、基于字符的文本特征），让你在没有 numpy 的情况下也能看到 InfoNCE 的形状。
2. 纯 Python 实现的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定）。
3. 用于比较的 sigmoid 成对损失实现。
4. 零样本分类例程：对一组文本提示计算余弦相似度并取 argmax 做预测。

运行它并观察损失曲线。绝对数值是玩具级别；曲线形状与真实 CLIP 训练器输出相匹配。

## 部署说明

本课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和目标类别列表，它用 CLIP 模板构建文本提示，使用指定的检查点（例如 `openai/clip-vit-large-patch14`）对双方进行嵌入，并返回 top-1 / top-5 的预测及相似度得分。该技能拒绝对提示列表外的类别做出断言。

## 练习

1. 手动实现批次大小为 4 的 InfoNCE。构造 4x4 的相似度矩阵，运行 softmax，取出对角线，计算交叉熵。用你的 Python 实现与手工计算结果校验。

2. SigLIP 除了温度外还用了一个偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当批次存在严重类别不平衡（每行负样本远多于正样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 构建一个猫 vs 狗 的零样本分类器。尝试两个提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板集合（ensemble）是否优于单一模板？

4. 计算在 512-GPU 运行、批量 32k 时 softmax InfoNCE 与 sigmoid 成对损失的通信成本。哪种随 N 线性增长，哪种随 N^2 增长？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 的缩放律论文（arXiv:2212.07143，Cherti 等）。用图中的结论复现数据缩放的关系：在固定模型大小下，ImageNet 零样本准确率与训练数据大小之间是什么样的对数线性关系？

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| InfoNCE | "Contrastive loss" | 对一个批次的相似度矩阵做交叉熵；每个样本的正例是与之配对的样本，负例是其他所有样本 |
| Sigmoid loss | "SigLIP loss" | 每对样本的二元交叉熵；没有 softmax、没有 all-gather，在分布式训练中廉价可扩展 |
| Temperature | "tau" | 在 softmax/sigmoid 前缩放 logits 的标量；控制分布的尖锐度 |
| Zero-shot | "no-finetune classification" | 使用文本提示构建类别嵌入，并通过余弦相似度分类；不对目标类别进行训练 |
| Prompt template | "a photo of a ..." | 围绕类别名称的文本骨架；会影响零样本准确率约 1–5 个点 |
| Dual encoder | "Two-tower" | 一个图像编码器 + 一个文本编码器，输出共享的 D 维空间 |
| Hard negative | "Tough distractor" | 与正例相似度足够高，需要模型努力将其分离的负样本 |
| Linear probe | "Frozen + one layer" | 在冻结特征上只训练一个线性分类器；用于衡量特征质量 |
| NaFlex | "Native flexible resolution" | SigLIP 2 能力：在不改变输入的情况下支持任意长宽比和分辨率 |
| Temperature scaling | "log-parametrized tau" | CLIP 用 `log(1/tau)` 参数化，以便梯度良好；并做裁剪以防 tau 过小 |

## 深入阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。  
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。  
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。  
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用嘈杂网络数据做尺度验证。  
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 的缩放律。