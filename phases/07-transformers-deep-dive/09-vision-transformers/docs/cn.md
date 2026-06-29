# Vision Transformers (ViT)

> 一张图像是一个补丁网格。一句句子是一个令牌网格。相同的 Transformer 可以同时处理二者。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 7 · 05（完整 Transformer）、Phase 4 · 03（卷积神经网络，CNNs）、Phase 4 · 14（Vision Transformers 介绍）  
**Time:** ~45 分钟

## 问题

在 2020 年之前，计算机视觉意味着卷积。ImageNet、COCO 和检测基准上的所有 SOTA 都使用 CNN 作为骨干网络。Transformer 是为语言设计的。

Dosovitskiy 等人（2020）— “An Image is Worth 16x16 Words” — 展示了可以完全去掉卷积。把图像切成固定大小的补丁，将每个补丁线性投影到嵌入空间，然后把序列输入到一个原生的 Transformer 编码器。在足够大的规模（例如在 ImageNet-21k 或更大的预训练上），ViT 能够与基于 ResNet 的模型匹敌甚至超越。

ViT 是 2026 年更广泛模式的起点：一种架构，多模态。Whisper 将音频分词；ViT 将图像分词；用于机器人学的动作令牌；视频的像素令牌。Transformer 不在乎——给它一个序列，它就能学习。

到 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）主导了大部分视觉领域。CNN 在边缘设备和对延迟敏感的任务上仍然胜出。其它几乎所有场景的堆栈中都能找到 ViT。

## 概念

![图像 → 补丁 → 令牌 → Transformer](../assets/vit.svg)

### 第 1 步 — patchify

将一个 `H × W × C` 的图像拆分为一个 `N × (P·P·C)` 的平铺补丁序列。典型配置：`224 × 224` 的图像，`16 × 16` 的补丁 → 196 个补丁，每个补丁有 768 个值。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

补丁大小是一个杠杆。补丁越小 = 令牌越多，分辨率更高，但自注意力的计算代价呈二次增长。补丁越大 = 粗糙但更便宜。

### 第 2 步 — 线性嵌入

用一个可学习的矩阵将每个平铺补丁投影到 `d_model`。等价于卷积核大小为 `P`、步幅为 `P` 的卷积。在 PyTorch 中这实际上就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` —— 两行代码即可实现。

### 第 3 步 — 前置 `[CLS]` 令牌，加入位置嵌入

- 前置一个可学习的 `[CLS]` 令牌。它的最终隐藏状态用于分类时作为图像表征。  
- 加上可学习的位置嵌入（ViT 原始做法），或者使用二维正弦位置编码（后续变体）。  
- 在 2024 年之后，RoPE 被扩展到二维用于位置编码，有时不再需要显式的位置嵌入。

### 第 4 步 — 标准 Transformer 编码器

堆叠 L 个块，模式为 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`。与 BERT 完全相同。没有任何视觉专用层。这是论文的教育性要点。

### 第 5 步 — 头部

用于分类：取 `[CLS]` 的隐藏状态 → 线性层 → softmax。对于 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用补丁嵌入。

### 有影响的变体

| Model | Year | Change |
|-------|------|--------|
| ViT | 2020 | 原始模型。固定补丁大小，全局注意力。 |
| DeiT | 2021 | 蒸馏；可以仅在 ImageNet-1k 上训练。 |
| Swin | 2021 | 分层结构并使用移位窗口。固定的次二次复杂度。 |
| DINOv2 | 2023 | 自监督（无标签）。最佳通用视觉特征。 |
| ViT-22B | 2023 | 22B 参数；遵循扩展规律。 |
| SigLIP | 2023 | ViT + 语言对，使用 sigmoid 对比损失。 |
| SAM 3 | 2025 | Segment Anything；ViT-Large + 可提示的掩码解码器。 |

### 为什么过了这么久才普及

ViT 需要大量数据才能与 CNN 匹敌，因为它没有 CNN 的归纳偏置（平移不变性、局部性）。没有 >1 亿带标签的图像或强大的自监督预训练，在相同计算预算下 CNN 仍然占优。DeiT 在 2021 年通过蒸馏技巧部分解决了这个问题；DINOv2 在 2023 年通过自监督长期解决了它。

## 构建

参见 `code/main.py`。纯标准库实现的 patchify + 线性嵌入 + 完整性检查。不包含训练 —— 任何现实规模的 ViT 需要 PyTorch 和数小时的 GPU 时间。

### 第 1 步：伪造图像

用行列表表示的 24 × 24 RGB 图像，每个元素是 `(R, G, B)` 元组。我们使用 6×6 的补丁 → 16 个补丁，每个补丁的平铺向量维度为 108。

### 第 2 步：patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

按栅格的行主序（raster order）展开：逐行遍历网格。每个 ViT 都使用这种顺序。

### 第 3 步：线性嵌入

将每个平铺向量乘以一个随机的 `(patch_flat_size, d_model)` 矩阵。验证在前置 `[CLS]` 之后输出形状为 `(N_patches + 1, d_model)`。

### 第 4 步：为现实的 ViT 统计参数量

打印 ViT-Base 的参数量：12 层，12 个头，d=768，patch=16。与 ResNet-50（约 2500 万）比较。ViT-Base 大约是 ~8600 万。ViT-Large ~3.07 亿。ViT-Huge ~6.32 亿。

## 使用

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 嵌入在 2026 年成为图像特征的默认选择。** 冻结骨干网络，训练一个小的头部。适用于分类、检索、检测、图像描述。Meta 的 DINOv2 检查点在每个非文本视觉任务上都优于 CLIP。

补丁大小的选择。小模型使用 16×16（ViT-B/16）。密集预测（分割）使用 8×8 或 14×14（SAM、DINOv2）。非常大的模型使用 14×14。

## 发布

参见 `outputs/skill-vit-configurator.md`。该技能根据数据集大小、分辨率和计算预算为新视觉任务选择 ViT 变体和补丁大小。

## 练习

1. **简单。** 运行 `code/main.py`。验证补丁数量等于 `(H/P) * (W/P)`，平铺补丁维度等于 `P*P*C`。  
2. **中等。** 实现二维正弦位置嵌入 —— 为每个补丁的 `row` 和 `col` 各自生成独立的正弦编码，然后拼接。把它们输入一个小型的 PyTorch ViT，在 CIFAR-10 上比较与可学习位置嵌入的准确率差异。  
3. **困难。** 构建一个 3 层 ViT（PyTorch），在 1000 张 MNIST 图像上用 4×4 补丁训练。测量测试准确率。现在在相同的 1000 张图像上加入 DINOv2 预训练（简化版：只训练编码器在被遮挡补丁上预测补丁嵌入）。准确率是否提升？

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Patch | "The vision-transformer token" | 对于 `P × P × C` 区域的像素值展平向量。 |
| Patchify | "Chop + flatten" | 将图像切成不重叠的补丁，并将每个补丁展平为向量。 |
| `[CLS]` token | "The image summary" | 前置的可学习令牌；其最终嵌入是图像的表示。 |
| Inductive bias | "What the model assumes" | ViT 比 CNN 拥有更少的先验；因此需要更多数据来弥补差距。 |
| DINOv2 | "Self-supervised ViT" | 使用图像增强 + 动量教师在无标签下训练。到 2026 年是最佳的通用图像特征。 |
| SigLIP | "CLIP's successor" | ViT + 文本编码器，使用 sigmoid 对比损失；在相同计算下优于 CLIP。 |
| Swin | "Windowed ViT" | 带局部注意力和移位窗口的分层 ViT；次二次复杂度。 |
| Register tokens | "2023 trick" | 一些额外的可学习令牌，用来吸收注意力中的“下沉点”；能提升 DINOv2 的特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。  
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。  
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。  
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。  
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的 register-token 修复。