# OCR & 文档理解

> OCR 是一个三阶段流水线 —— 检测文本框、识别字符，然后进行布局重建。每个现代 OCR 系统要么重排这些阶段，要么将它们合并。

**Type:** 学习 + 使用  
**Languages:** Python  
**Prerequisites:** Phase 4 Lesson 06（检测），Phase 7 Lesson 02（自注意力）  
**Time:** ~45 分钟

## 学习目标

- 理解经典 OCR 流水线（检测 -> 识别 -> 布局）以及现代端到端替代方案（Donut、Qwen-VL-OCR）
- 实现用于序列到序列 OCR 训练的 CTC（Connectionist Temporal Classification）损失
- 使用 PaddleOCR 或 EasyOCR 在无需训练的情况下进行生产级文档解析
- 区分 OCR、布局解析与文档理解 —— 并为不同任务选择合适的工具

## 问题背景

充满文本的图像随处可见：收据、发票、身份证、扫描书籍、表单、白板、标牌、截图。将它们转换为结构化数据 —— 不仅仅是字符，而是“这是总金额” —— 是计算机视觉中价值很高的应用之一。

该领域可以分为三层技能：

1. **OCR 本体**：把像素变成文本。
2. **布局解析**：将 OCR 输出分组为区域（标题、正文、表格、页眉）。
3. **文档理解**：从布局中提取结构化字段（例如 "invoice_total = $42.50"）。

每一层都有经典与现代方法，而且从“我想从图像得到文本”到“我需要从这张收据里得到总金额”之间的差距比大多数团队意识到的要大得多。

## 概念

### 经典流水线

```mermaid
flowchart LR
    IMG["图像"] --> DET["文本检测<br/>(DB、EAST、CRAFT)"]
    DET --> BOX["单词/行<br/>边界框"]
    BOX --> CROP["裁剪每个区域"]
    CROP --> REC["识别<br/>(CRNN + CTC)"]
    REC --> TXT["文本字符串"]
    TXT --> LAY["布局<br/>排序"]
    LAY --> OUT["按阅读顺序的文本"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测** 输出每行或每词的四边形框。
- **识别** 将每个区域裁剪为固定高度，运行 CNN + BiLSTM + CTC 以产生字符序列。
- **布局** 重建阅读顺序（对拉丁文为从上到下、从左到右；阿拉伯文、日文等则不同）。

### 用一段话理解 CTC

OCR 识别需要从固定长度的特征图产生可变长度的序列。CTC（Graves 等，2006）允许你在没有字符级对齐的情况下训练。模型在每个时间步对（词表 + blank）输出分布；CTC 损失对所有在合并重复并去除 blank 后与目标文本对应的对齐进行边缘化。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC 是 2015 年 CRNN 能够工作的原因，并且在 2026 年仍然是大多数生产 OCR 模型的训练方法。

### 现代端到端模型

- **Donut**（Kim 等，2022）—— ViT 编码器 + 文本解码器；直接读图像并输出 JSON。无需文本检测器、无需布局模块。
- **TrOCR** —— ViT + transformer 解码器用于行级 OCR。
- **Qwen-VL-OCR / InternVL** —— 针对 OCR 任务微调的完整视觉-语言模型；在 2026 年复杂文档上精度最好。
- **PaddleOCR** —— 以经典 DB + CRNN 流水线为基础的成熟生产包；仍然是开源主力。

端到端模型需要更多数据和计算，但可以跳过多阶段流水线中的误差积累。

### 布局解析

对于结构化文档，运行布局检测器（LayoutLMv3、DocLayNet），对每个区域进行标注：标题、段落、图、表、脚注。阅读顺序则变为“按布局顺序遍历区域并拼接”。

对于表单，使用 **键值提取** 模型（视觉丰富文档可用 Donut，纯扫描文档可用 LayoutLMv3）。它们接受图像 + 已检测文本 + 位置信息，并预测结构化的键值对。

### 评估指标

- **Character Error Rate (CER)** —— Levenshtein 距离 / 参考长度。越低越好。生产目标：干净扫描件 < 2%。
- **Word Error Rate (WER)** —— 以单词为单位的同类指标。
- **结构化字段的 F1** —— 针对键值任务；衡量 `{invoice_total: 42.50}` 是否被正确提取。
- **JSON 编辑距离** —— 针对端到端文档解析；Donut 论文提出了归一化的树编辑距离。

## 构建实现

### 第一步：CTC 损失 + 贪心解码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) 包含 blank（索引 0）的词表上的 log-softmax
    targets:        (N, S) 整数型目标（不包含 blanks）
    input_lengths:  (N,) 每个样本使用的时间步长度
    target_lengths: (N,) 每个样本的目标长度
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (去除 blanks 并合并重复)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在可用时会使用高效的 CuDNN 实现。贪心解码比束搜索更简单，通常与束搜索的 CER 相差不超过 1%。

### 第二步：Tiny CRNN 识别器

行 OCR 的最小 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

输入为固定高度（CNN 对高度做了最大池化至 1）。宽度是 CTC 的时间维度。

### 第三步：合成 OCR 数据

生成黑底白字或黑字白底的数字字符串用于端到端冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实的 OCR 数据集会加入字体变化、噪声、旋转、模糊和颜色。上面的流水线在真实场景下是相同的。

### 第四步：训练草图

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这个简单的合成数据上，损失应从约 ~3 降到 ~0.2，在 200 步左右。

## 使用指南

三条生产路线：

- **PaddleOCR** —— 成熟、快速、多语种。一行调用：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR** —— Python 原生、多语种、基于 PyTorch。
- **Tesseract** —— 经典工具；在模型表现不佳的老式扫描件上仍然有用。

对于端到端文档解析，使用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于格式可重复的收据、发票和表单，可微调 Donut。对于任意文档或需要推理能力的 OCR，像 Qwen-VL-OCR 的 VLM 是当前默认选择。

## 部署输出

本课产生：

- `outputs/prompt-ocr-stack-picker.md` — 一个根据文档类型、语言和结构在 Tesseract / PaddleOCR / Donut / VLM-OCR 中选择的提示词。
- `outputs/skill-ctc-decoder.md` — 一个从头实现贪心和束搜索 CTC 解码器的技能说明，包含长度归一化。

## 练习

1. **（简单）** 在 5 位随机数字字符串上训练 TinyCRNN 500 步。报告在保留集上的 CER。
2. **（中等）** 用束搜索（beam_width=5）替换贪心解码。报告 CER 的变化。在哪类输入上束搜索更有优势？
3. **（困难）** 在 20 张收据上使用 PaddleOCR，提取行项目，并针对手工标注的 {item_name, price} 对计算 F1。

## 术语表

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| OCR | "Text from pixels" | 将图像区域转换为字符序列 |
| CTC | "Alignment-free loss" | 在没有每时间步标签的情况下训练序列模型；对齐进行边缘化 |
| CRNN | "Classic OCR model" | 卷积特征提取器 + BiLSTM + CTC；2015 年的基线，仍在生产中使用 |
| Donut | "End-to-end OCR" | ViT 编码器 + 文本解码器；直接从图像输出 JSON |
| Layout parsing | "Find regions" | 在文档中检测并标注标题/表格/图/段落等区域 |
| Reading order | "Text sequence" | 将识别出的区域按顺序排列成文本；对拉丁文简单，但对混合布局并非易事 |
| CER / WER | "Error rates" | 在字符或单词粒度上的 Levenshtein 距离 / 参考长度 |
| VLM-OCR | "LLM that reads" | 为 OCR 任务训练或提示的视觉-语言模型；在复杂文档上当前为 SOTA |

## 深入阅读

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) — 原始的 CNN+RNN+CTC 架构  
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — 原始 CTC 论文；详细介绍了算法思想  
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) — 无 OCR 的端到端文档理解 Transformer  
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 开源的生产级 OCR 栈