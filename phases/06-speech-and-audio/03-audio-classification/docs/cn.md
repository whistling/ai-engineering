# Audio Classification — 从 MFCC 上的 k-NN 到 AST 与 BEATs

> 从“狗叫与警报器”到“这是什么语言”的所有任务都属于音频分类。特征通常是梅尔谱。架构每十年演进一次。评估指标仍然是 AUC、F1 和按类召回率。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 6 · 02（谱图与梅尔），Phase 3 · 06（卷积神经网络），Phase 5 · 08（用于文本的 CNNs 与 RNNs）  
**Time:** ~75 分钟

## 问题

你得到一个 10 秒的音频片段。你想知道：“这是什么？” 城市声音（警报、钻机、狗）、语音指令（是/否/停止）、语言识别（en/es/ar）、说话人情绪（愤怒/中性）、或环境声音（室内/室外、人声背景）。这些都是音频分类。在 2026 年，基线架构已经成熟：对数梅尔谱 → CNN 或 Transformer → softmax。

核心难点不是网络，而是数据。音频数据集有严重的类别不平衡、强烈的域漂移（干净 vs 噪声）和标签噪声（谁来决定“城市喧闹” vs “餐厅噪声”？）。80% 的问题是数据治理、增强和评估，而不是把 CNN 换成 Transformer。

## 概念

![音频分类阶梯：从 MFCC 上的 k-NN 到 AST 再到 BEATs](../assets/audio-classification.svg)

**k-NN 在 MFCC 上（1990 年代基线）。** 将每个片段的 MFCC 展平，计算与带标签样本库的余弦相似度，返回前 K 的多数投票。在干净、小规模数据集（Speech Commands、ESC-50）上出人意料地强。无需 GPU 即可运行。

**对数梅尔谱上的 2D CNN（2015–2019）。** 将 (T, n_mels) 的对数梅尔谱当作一张图像。应用 ResNet-18 或 VGG 风格网络。在时间轴上做全局均值池化。对类别做 softmax。在大多数 2026 年的 Kaggle 比赛中仍是基线。

**音频谱图变换器，AST（2021–2024）。** 将对数梅尔谱分块（例如 16×16 的 patch），加入位置嵌入，输入 ViT。监督学习在 AudioSet 上的 mAP 达到 0.485。

**BEATs 和 WavLM-base（2024–2026）。** 在数百万小时数据上进行自监督预训练。用 1–10% 的有监督数据微调即可达到之前所需数据量的性能。在 2026 年，对于非语音音频，这已是默认的起点。BEATs-iter3 在 AudioSet 上比 AST 高出 1–2 mAP，同时使用 1/4 的计算资源。

**将 Whisper 的 encoder 作为冻结骨干（2024）。** 取 Whisper 的 encoder，丢弃 decoder，接上线性分类头。在语言识别和简单事件分类上（且不做音频增强）可以接近 SOTA，是一种“免费午餐”基线。

### 类不平衡才是真正的挑战

ESC-50：50 类，每类 40 个片段 — 平衡、容易。UrbanSound8K：10 类，存在 10:1 的不平衡。AudioSet：632 类，长尾可达 100,000:1。有效的技术包括：

- 训练时做平衡采样（评估时不要）。
- Mixup：线性插值两段音频（以及它们的标签）作为增强。
- SpecAugment：随机掩盖时间和频率带。简单但关键。

### 评估

- 多类互斥（如 Speech Commands）：top-1 准确率、top-5 准确率。
- 多类多标签（AudioSet、类似 UrbanSound 的设置）：平均精度均值（mAP）。
- 严重不平衡时：按类召回率 + 宏 F1。

你应该知道的 2026 年指标：

| 基准 | 基线 | 2026 年 SOTA | 来源 |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs 论文（2024） |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR 排行榜 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 结果 |

## 实现

### 步骤 1：特征化

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 步骤 2：定长摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但强：对时间维度做均值 + 方差，为 13 维 MFCC 提供 26 维的定长固定嵌入。运行极快。到 2017 年为止，在 ESC-50 上超过很多 NN 基线。

### 步骤 3：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 步骤 4：升级到对数梅尔谱上的 CNN

在 PyTorch 中：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels) 输入：批大小 B，通道 1，时间 T，梅尔数 n_mels
        return self.head(self.body(x).flatten(1))
```

约 3M 参数。在单张 RTX 4090 上对 ESC-50 训练约 10 分钟，准确率 80%+。

### 步骤 5：2026 年默认 — 微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于 BEATs，使用 `microsoft/BEATs-base`（通过 `beats` 库）；transformers API 的输入输出形状相同。

## 使用建议

2026 年的栈：

| 情况 | 起点 |
|-----------|-----------|
| 极小数据集（<1000 片段） | k-NN 在 MFCC 均值（作为基线）+ 音频增强 |
| 中等数据集（1K–100K） | 微调 BEATs 或 AST |
| 大型数据集（>100K） | 从头训练或微调 Whisper-encoder |
| 实时、边缘设备 | 40-MFCC 的 CNN，量化到 int8（KWS 风格） |
| 多标签（AudioSet） | BEATs-iter3 + BCE 损失 + mixup + SpecAugment |
| 语言识别 | MMS-LID，或 SpeechBrain 的 VoxLingua107 基线 |

决策规则：先从冻结的骨干开始，而不是训练全新的模型。微调 BEATs 的分类头能在数小时内拿到 95% 的 SOTA 性能，而不是数周。

## 部署

保存为 `outputs/skill-classifier-designer.md`。为给定的音频分类任务选择架构、增强策略、类平衡策略和评估指标。

## 练习

1. **简单。** 运行 `code/main.py`。它会在一个 4 类合成数据集（不同音高的纯音）上训练 k-NN MFCC 基线。报告混淆矩阵。
2. **中等。** 用 [均值、方差、偏度、峰度] 替换 `summarize`。在同一合成数据集上，4 阶矩池化是否优于均值+方差？
3. **困难。** 使用 `torchaudio` 在 ESC-50 的 fold 1 上训练一个 2D CNN。报告 5 折交叉验证准确率。加入 SpecAugment（time mask = 20，freq mask = 10）并报告变化量。

## 关键词

| 术语 | 通常说法 | 实际含义 |
|------|-----------------|-----------------------|
| AudioSet | 音频的 ImageNet | 谷歌的 2M 片段、632 类弱标注 YouTube 数据集。 |
| ESC-50 | 小型分类基准 | 50 类 × 每类 40 个环境声音片段。 |
| AST | Audio Spectrogram Transformer | 在对数梅尔 patch 上的 ViT；2021 年的一个 SOTA 架构。 |
| BEATs | 自监督音频模型 | 微软的模型，iter3 在 2026 年领导 AudioSet。 |
| Mixup | 配对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于掩盖的增强 | 将谱图的随机时间和频率带置零。 |
| mAP | 主要的多标签指标 | 跨类别和阈值的平均精度均值。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 记录 2021–2024 年的架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024+ 的默认方法。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主导的音频增强方法。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 存在的 50 类基准数据集。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632 类的 YouTube 分类法；仍然是金标准。