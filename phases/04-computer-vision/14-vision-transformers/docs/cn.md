# 视觉Transformer (Vision Transformers, ViT)

> 将图像切成补丁，把每个补丁当作一个词，运行标准的 Transformer。别回头看。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** 第7阶段 第02课（Self-Attention），第4阶段 第04课（Image Classification）  
**Time:** ~45 分钟

## 学习目标

- 从零实现 patch embedding、学习的位置嵌入、class token 和 transformer 编码器块，从而构建一个最小 ViT
- 解释为什么早期认为 ViT 需要大规模预训练数据，直到 DeiT 和 MAE 证明并非如此
- 比较 ViT、Swin、ConvNeXt 在架构先验上的差异（无先验、局部窗口注意力、卷积骨干）
- 使用 `timm` 对预训练 ViT 在小数据集上进行微调，并遵循标准的 linear-probe / fine-tune 流程

## 问题背景

十多年来，卷积被视为计算机视觉的同义词。CNN 具有强烈的归纳偏置——局部性、平移等变性——没人认为能被替代。然后 Dosovitskiy 等（2020）展示了：对展开的图像补丁直接应用一个简单的 Transformer，完全不使用卷积结构，在大规模下可以匹配或超越最好的 CNN。

但关键是“在大规模下”。在 ImageNet-1k 上直接训练的 ViT 输给了 ResNet。在用 ImageNet-21k 或 JFT-300M 预训练后再在 ImageNet-1k 上微调，ViT 则超越了 ResNet。结论是 Transformer 缺乏有用的先验，但可以从足够的数据中学到这些先验。后续工作（DeiT、MAE、DINO）表明，通过合适的训练配方——强增强、自监督预训练、蒸馏——ViT 在小数据上也能训练良好。

到 2026 年，纯 CNN 在边缘设备上仍有竞争力（ConvNeXt 最强），但 Transformer 在其他领域占主导：分割（Mask2Former、SegFormer）、检测（DETR、RT-DETR）、多模态（CLIP、SigLIP）、视频（VideoMAE、VJEPA）。ViT 的块结构是必须掌握的。

## 概念

### 流程图

```mermaid
flowchart LR
    IMG["图像<br/>(3, 224, 224)"] --> PATCH["补丁嵌入<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["展平为<br/>(196, 768) 令牌"]
    FLAT --> CAT["前置<br/>[CLS] 令牌"]
    CAT --> POS["添加学习的<br/>位置嵌入"]
    POS --> ENC["N 个 Transformer<br/>编码器块"]
    ENC --> CLS["取 [CLS]<br/>令牌输出"]
    CLS --> HEAD["MLP 分类头"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七个步骤。补丁 -> 令牌 -> 注意力 -> 分类器。每个变体（DeiT、Swin、ConvNeXt、MAE 预训练）都改变这七步中的一两项，其余保持不变。

### Patch embedding

第一个卷积是关键。核大小 16，步长 16，所以 224x224 的图像变成 14x14 的 16x16 补丁网格，每个补丁被投影为 768 维嵌入。那个单卷积既把图像切成补丁也做了线性投影。

```
Input:  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, no padding):
Output: (768, 14, 14)
Flatten spatial: (196, 768)
```

196 个补丁 = 196 个令牌。每个令牌的特征维度是 768（ViT-B）、1024（ViT-L）或 1280（ViT-H）。

### Class token

一个学习得到的向量被前置到序列中：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

经过 N 个 transformer 块后，`[CLS]` 的输出就是全局图像表示。分类头只读取这个向量。

### 位置嵌入

Transformer 本身没有内置的空间位置信息。向每个令牌加上一个学习向量：

```
tokens = tokens + learned_pos_embedding   (also shape (197, 768))
```

位置嵌入是模型的参数；通过梯度训练，它会适应二维图像结构。也存在基于正弦的二维替代方案，但在实践中很少使用。

### Transformer 编码器块

标准块：多头自注意力、MLP、残差连接、pre-LayerNorm。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP is two-layer with GELU: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16 堆叠 12 个这样的块，每个有 12 个注意力头，总参数约 86M。

### 为什么用 pre-LN

早期 Transformer 用的是 post-LN（`x = LN(x + sublayer(x))`），在没有 warmup 的情况下训练超过 6-8 层会遇到困难。pre-LN（`x = x + sublayer(LN(x))`）可以在无 warmup 下稳定训练更深的网络。每个 ViT 和每个现代 LLM 都使用 pre-LN。

### 补丁大小的权衡

- 16x16 补丁 -> 196 个令牌，标准配置。
- 32x32 补丁 -> 49 个令牌，更快但分辨率低。
- 8x8 补丁 -> 784 个令牌，更细粒但 O(n^2) 的注意力成本增长很快。

更大的补丁 = 更少的令牌 = 更快但空间细节少。SwinV2 在分层窗口中使用 4x4 补丁。

### DeiT 的 ImageNet-1k 训练配方

原始 ViT 需要 JFT-300M 才能超过 CNN。DeiT（Touvron 等，2020）在仅用 ImageNet-1k 的情况下，通过四个改动把 ViT-B 训练到 81.8% 的 top-1：

1. 强增强：RandAugment、Mixup、CutMix、Random Erasing。
2. 随机深度（stochastic depth）：训练时随机丢弃整层块。
3. 重复增强（repeated augmentation）：同一张图像在一个 batch 中被多次采样（例如 3 次）。
4. 从 CNN 教师蒸馏（可选，能进一步提升精度）。

每个现代 ViT 的训练配方都来源于 DeiT。

### Swin vs ConvNeXt

- Swin（Liu 等，2021）——基于窗口的注意力。每个块在局部窗口内进行注意力计算；交替块会移动窗口以在窗口之间混合信息。引入了类似 CNN 的局部性先验，但保留了注意力算子。
- ConvNeXt（Liu 等，2022）——重新设计的 CNN，采用了与 Swin 相似的架构选择（depthwise convs、LayerNorm、GELU、反向瓶颈）。表明差距并非“注意力 vs 卷积”，而是“现代训练配方 + 架构设计”。

到 2026 年，ConvNeXt-V2 和 Swin-V2 都已进入生产级；正确选择取决于你的推理栈（ConvNeXt 在边缘设备上更易编译）和预训练语料。

### MAE 预训练

Masked Autoencoder（He 等，2022）：随机遮盖 75% 的补丁，只用可见的 25% 让编码器处理，训练一个小解码器从编码器输出中重建被遮盖的补丁。预训练后丢弃解码器并微调编码器。

MAE 使 ViT 在仅用 ImageNet-1k 时也能训练良好，达到 SOTA，是当前默认的自监督配方。

## 实践构建

### 步骤 1：Patch embedding

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一个卷积，一个展平，一个转置。就是整个图像到令牌的步骤。

### 步骤 2：Transformer 块

Pre-LN、多头自注意力、带 GELU 的 MLP、残差连接。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` 负责分头、缩放点积和输出投影。`batch_first=True` 所以形状为 `(N, seq, dim)`。

### 步骤 3：ViT 模型

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

大约 2.8M 参数——一个可在 CPU 上可行运行的迷你 ViT。真实的 ViT-B 约 86M；只需将 `dim=768, depth=12, num_heads=12` 带入相同类定义即可。

### 步骤 4：健全性检查 — 单张图像推理

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

应当无错误地运行。概率和为 1。

## 使用预训练模型

`timm` 提供了各种 ViT 变体的 ImageNet 预训练权重。只需一行：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

到 2026 年，`timm` 已成为视觉 Transformer 的生产默认工具。它在统一 API 下支持 ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer 等几十种模型。

对于多模态工作（图像 + 文本），`transformers` 提供了 CLIP、SigLIP、BLIP-2、LLaVA。所有这些模型的图像编码器通常都是 ViT 的某个变体。

## 交付产物

本课将产出：

- `outputs/prompt-vit-vs-cnn-picker.md` — 一个提示模板，根据数据集大小、计算资源和推理栈在 ViT、ConvNeXt、Swin 之间做出选择。
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` — 一个技能脚本，验证 ViT 的 patch embedding 和位置嵌入形状是否与模型预期的序列长度匹配，捕捉最常见的移植错误。

## 练习

1. **(Easy)** 打印上面迷你 ViT 前向传播中每个中间张量的形状。确认：输入 `(N, 3, 64, 64)` -> 补丁 `(N, 16, 192)` -> 加 CLS 后 `(N, 17, 192)` -> 分类器输入 `(N, 192)` -> 输出 `(N, num_classes)`。
2. **(Medium)** 在 Lesson 4 的 synthetic-CIFAR 数据集上微调一个预训练的 `timm` ViT-S/16。与在相同数据上微调的 ResNet-18 做比较。报告训练时间和最终精度。
3. **(Hard)** 为迷你 ViT 实现 MAE 预训练：遮盖 75% 的补丁，训练编码器 + 一个小解码器重建被遮盖的补丁。在预训练前后评估 linear-probe 在合成数据上的准确率。

## 关键词

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| Patch embedding | "The first conv" | 一个卷积，核大小 = 步长 = 补丁大小；把图像变成网格化的令牌嵌入 |
| Class token | "[CLS]" | 一个学习向量，前置到令牌序列；它的最终输出是全局图像表示 |
| Positional embedding | "Learned pos" | 一个学习向量，叠加到每个令牌上，使 Transformer 知道每个补丁的位置 |
| Pre-LN | "LayerNorm before sublayer" | 稳定的 Transformer 变体：`x + sublayer(LN(x))` 而非 `LN(x + sublayer(x))` |
| Multi-head attention | "Parallel attention" | 标准 Transformer 注意力分成 num_heads 个独立子空间，并在之后拼接 |
| ViT-B/16 | "Base, patch 16" | 典型配置：dim=768, depth=12, heads=12, patch_size=16, image=224；约 86M 参数 |
| DeiT | "Data-efficient ViT" | 在仅用 ImageNet-1k 的条件下，通过强增强训练 ViT；证明了大规模预训练数据并非绝对必要 |
| MAE | "Masked autoencoder" | 自监督预训练：遮盖 75% 的补丁并重建；主流的 ViT 自监督配方 |

## 延伸阅读

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — ViT 原始论文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) — 如何仅用 ImageNet-1k 训练 ViT
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) — MAE 预训练
- [timm documentation](https://huggingface.co/docs/timm) — 生产中使用的视觉 Transformer 参考文档