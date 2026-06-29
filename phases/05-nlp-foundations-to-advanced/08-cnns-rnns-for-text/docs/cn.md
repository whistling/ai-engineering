# 用于文本的 CNN 与 RNN

> 卷积学习 n-grams。循环负责记忆。两者都被注意力所取代。但在受限硬件上依然重要。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 3 · 11 (PyTorch 入门), Phase 5 · 03 (词嵌入), Phase 4 · 02 (从零实现卷积)  
**Time:** ~75 分钟

## 问题

TF-IDF 和 Word2Vec 产生了忽略词序的平面向量。基于它们构建的分类器无法区分 `dog bites man` 与 `man bites dog`。词序有时携带关键信号。

在变换器出现之前，有两类架构弥补了这一不足。

**用于文本的卷积网络 (TextCNN)。** 在词嵌入序列上应用一维卷积。宽度为 3 的滤波器是一个可学习的三元语法检测器：它覆盖三个词并输出一个得分。堆叠不同宽度（2、3、4、5）以检测多尺度模式。对特征图做最大池化以得到固定大小的表示。结构扁平、并行、速度快。

**循环网络 (RNN、LSTM、GRU)。** 一个一个标记地处理 token，维护一个将信息传递下去的隐藏状态。顺序处理、具有记忆、支持可变长度输入。从 2014 到 2017 年间主导序列建模，之后注意力机制出现并改变了格局。

本课将实现两者，并指出促使注意力机制出现的失败点。

## 概念

**TextCNN**（Kim, 2014）。对 token 做嵌入。宽度为 `k` 的一维卷积在连续的 `k`-gram 嵌入上滑动一个滤波器，产生一个特征图。对该特征图做全局最大池化选取最强激活。将来自多种滤波器宽度的最大池化输出拼接在一起，送入分类头。

为什么有效。滤波器就是可学习的 n-gram 检测器。最大池化位置不变，因此“not good” 无论出现在评论开头还是中间都会触发相同的特征。三个宽度、每个宽度 100 个滤波器就能得到 300 个可学习的 n-gram 检测器。训练可并行进行；没有时间步依赖。

**RNN。** 在每个时间步 `t`，隐藏状态满足 `h_t = f(W * x_t + U * h_{t-1} + b)`。在时间上共享 `W`、`U`、`b`。时间步 `T` 的隐藏状态是前缀的摘要。用于分类时，对 `h_1 ... h_T` 做池化（最大、平均或取最后一个）。

普通 RNN 存在梯度消失问题。**LSTM** 引入了决定遗忘、存储与输出的门控，从而稳定了长序列上的梯度。**GRU** 将 LSTM 简化为两个门；参数更少但性能相近。

**双向 RNN。** 同时运行一个正向和一个反向 RNN，并将隐藏状态拼接。每个 token 的表示能看到左右两侧的上下文。对标注任务至关重要。

```figure
rnn-unroll
```

## 实现

### 步骤 1：在 PyTorch 中实现 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 变形为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道。池化后输出的大小与输入长度无关，固定不变。

### 步骤 2：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

这里在序列上做最大池化，而不是取最后一个状态。用于分类时，最大池化通常比取最后一个隐藏状态效果更好，因为长序列末端的信息往往会主导最后一个状态。

### 步骤 3：梯度消失演示（直观）

没有门控的普通 RNN 无法学习长期依赖。考虑一个玩具任务：预测序列中是否出现过 token `A`。如果 `A` 在位置 1，序列长度为 100，那么从损失到位置 1 的梯度需经过 99 次对循环权重的相乘。如果权重小于 1，梯度会消失；如果大于 1，会爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# 在权重=0.9 且跨 100 步时：
#   0.9 ^ 100 ≈ 2.7e-5
# 从第100步传回第1步的梯度实际上为零。
```

LSTM 通过一个沿网路贯通、仅有加性交互的**单元状态**来解决这个问题（遗忘门会做乘性缩放，但梯度仍沿“高速公路”流动）。GRU 用更少的参数实现了类似效果。两者都能在 100+ 步的序列上实现稳定训练。

### 步骤 4：为什么这仍不足以解决所有问题

即使使用 LSTM，仍存在三个问题。

1. 顺序瓶颈。对长度为 1000 的序列训练 RNN 需要 1000 次串行的正反向传播步骤，无法在时间维度上并行。
2. 编码器-解码器设置下的固定大小上下文向量。解码器只能看到编码器的最终隐藏状态，即对整个输入的压缩。长输入会丢失细节。第 09 课会直接讨论这个问题。
3. 远距离依赖的准确率上限。LSTM 虽然比普通 RNN 更好，但在跨越 200+ 步传递特定信息时仍然困难。

注意力机制解决了这三点。Transformer 完全放弃了循环结构。第 10 课是转折点。

## 使用建议

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 已可用于生产。训练代码与常规流程一致。

Hugging Face 提供的预训练编码器可以作为输入层直接接入：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

何时使用这些方法的清单。

- 边缘 / 设备端推理。使用 GloVe 嵌入的 TextCNN 比 Transformer 小 10–100 倍。如果部署目标是手机，这是优选栈。
- 流式 / 在线分类。RNN 可以逐个 token 处理；Transformer 需要完整序列。对于实时到来的文本，LSTM 仍有优势。
- 作为基线的微型模型。快速迭代新任务。在 CPU 上训练一个 TextCNN 只需 5 分钟。
- 数据有限的序列标注。BiLSTM-CRF（第 06 课）仍是 1k–10k 标注句子的生产级 NER 架构。

其他情况都优先使用 Transformer。

## 发布

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. 简单。训练一个 TextCNN 在一个三分类的玩具数据集上（你自己构造数据）。验证使用滤波器宽度组合 (2, 3, 4) 在平均 F1 上优于单一宽度 (3)。
2. 中等。为 LSTM 分类器实现最大池化、平均池化和取最后状态三种池化方式。在小数据集上比较它们；记录哪个池化方式获胜并推测原因。
3. 困难。构建一个 BiLSTM-CRF 的 NER 标注器（结合第 06 课与本课）。在 CoNLL-2003 上训练。与第 06 课的仅 CRF 基线以及 BERT 微调比较。报告训练时间、内存和 F1。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| TextCNN | CNN for text | 在词嵌入上堆叠一维卷积并做全局最大池化。Kim (2014)。 |
| RNN | Recurrent net | 隐藏状态在每个时间步更新： `h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | Gated RNN | 添加输入门 / 遗忘门 / 输出门 + 一个单元状态。能在长序列上稳定训练。 |
| GRU | Simpler LSTM | 用两个门替代三个门。准确率相近、参数更少。 |
| Bidirectional | Both directions | 正向 + 反向 RNN 拼接。每个 token 可见左右两侧上下文。 |
| Vanishing gradient | Training signal dies | 在普通 RNN 中，<1 的重复乘积会使早期时间步的梯度趋近于零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文。八页，易读。  
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文。意外地通俗易懂。  
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 那些让 LSTM 易于理解的图示。