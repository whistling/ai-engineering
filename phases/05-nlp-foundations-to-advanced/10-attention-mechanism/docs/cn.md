# 注意力机制 — 突破

> 解码器不再只盯着一个压缩的摘要，而开始查看整个源序列。此后的一切都是注意力加工程实现。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 5 · 09（序列到序列模型）
**Time:** ~45 分钟

## 问题

第 09 课以一个受限的失败告终。一个在玩具复制任务上训练的 GRU 编码器-解码器在长度为 5 时能达到 89% 的准确率，但在长度为 80 时接近随机。原因是结构性的，而不是训练错误：编码器得到的每一位信息都必须装进一个固定大小的隐藏状态里，解码器看不到别的东西。

Bahdanau、Cho 和 Bengio 在 2014 年给出了三行代码的修复。不是只把最终编码器状态给解码器，而是保留每个编码器状态。在每个解码步骤，计算对所有编码器状态的加权平均，权重表示“解码器现在需要查看编码器位置 `i` 的程度有多大？”这个加权平均就是上下文，并且每个解码步骤都会变化。

这就是全部思想。Transformer 扩展了它。自注意力把它应用到单个序列上。多头注意力并行运行它。但 2014 年的版本已经打破了瓶颈，一旦有了它，向 Transformer 的转变更多是工程实现而非概念上的飞跃。

## 概念

![Bahdanau 注意力：解码器查询所有编码器状态](../assets/attention.svg)

在每个解码步骤 `t`：

1. 使用上一步的解码器隐藏状态 `s_{t-1}` 作为一个 **query**。
2. 将它与每个编码器隐藏状态 `h_1, ..., h_T` 进行打分。对每个编码器位置得到一个标量。
3. 对这些分数做 softmax，得到和为 1 的注意力权重 `α_{t,1}, ..., α_{t,T}`。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`。对编码器状态的加权平均。
5. 解码器使用 `c_t` 加上上一个输出 token，生成下一个 token。

加权平均是关键。当解码器需要把法语 "Je" 翻成英语 "I" 时，它会把对应 "Je" 的编码器状态权重大；当需要翻译 "not" 时，它会把 "pas" 的权重提到高位。上下文向量会随着每一步重塑信息。

## 维度（导致错误的地方）

这是每个注意力实现第一次出错的地方。慢慢读。

| 项目 | 形状 | 说明 |
|------|------|------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 如果是 BiLSTM，`d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` 做 softmax 后 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态同形状 |

**Bahdanau（加性）得分。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 的形状为 `(d_s,)`，`h_i` 的形状为 `(d_h,)`。
- `W_a` 的形状为 `(d_attn, d_s)`。`U_a` 的形状为 `(d_attn, d_h)`。
- 它们在 tanh 内相加后的形状为 `(d_attn,)`。
- `v_α` 的形状为 `(d_attn,)`。与 `v_α` 的内积会折叠为一个标量。**这就是 `v_α` 的作用。** 它不是魔法，而是将注意力维向量投影为标量分数的向量。

**Luong（乘性）得分。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。这是一个硬性约束。如果编码器是双向的就很难用。
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中 `W` 形状为 `(d_s, d_h)`。移除了等维的约束。
- `concat`：本质上是 Bahdanau 形式。由于前两种计算更便宜，`concat` 很少使用。

**一个关于 Bahdanau / Luong 值得提醒的陷阱。** Bahdanau 使用 `s_{t-1}`（在生成当前单词之前的解码器状态）。Luong 使用 `s_t`（生成后的状态）。把它们混用会产生微妙错误的梯度，非常难调试。选定一篇论文并坚持它的约定。

```figure
attention-heatmap
```

## 实现它

### 第 1 步：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

根据上表检查你的形状。`encoder_states` 的形状为 `(T_enc, d_h)`。`projected_enc` 的形状为 `(T_enc, d_attn)`。`projected_dec` 的形状为 `(d_attn,)` 并会广播。`combined` 的形状为 `(T_enc, d_attn)`。`scores` 的形状为 `(T_enc,)`。`weights` 的形状为 `(T_enc,)`。`context` 的形状为 `(d_h,)`。可以交付了。

### 第 2 步：Luong 的 dot 和 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每个函数只有三行。这就是 Luong 论文受欢迎的原因。在大多数任务上能得到相当的准确率，但代码量少得多。

### 第 3 步：一个数值化的示例

假设有三个编码器状态（大致对应 "cat", "sat", "mat"），以及一个与第一个最对齐的解码器状态，注意力分布会集中在位置 0。如果把解码器状态移向第三个编码器状态，注意力会移动到位置 2。上下文向量跟随改变。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。然后把解码器状态移向第三个编码器状态，观察权重的变化。就是这么简单。注意力就是显式对齐。

### 第 4 步：为什么这是通向 Transformer 的桥梁

把上面的语言翻译为 Q/K/V：

- **Query** = 解码器状态 `s_{t-1}`
- **Key** = 编码器状态（用来打分的对象）
- **Value** = 编码器状态（按权重加权求和返回的内容）

在经典注意力中，keys 和 values 是相同的东西。自注意力把它们分开：你可以用一个序列去查询自身，并对 K 和 V 使用不同的学习到的投影。多头注意力用不同的投影并行运行多个注意力头。Transformer 将整个阶段堆叠多次并去掉了 RNN。

数学是相同的。形状是相同的。从 Bahdanau 注意力到缩放点积注意力的教学跳跃主要是符号上的差异。

## 使用它

PyTorch 和 TensorFlow 都直接提供注意力模块。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 Transformer 的注意力层。query 有 5 个位置，key/value 有 10 个位置，维度均为 128，8 个头。`output` 是经过上下文增强后的 queries。`weights` 是可以可视化的 5x10 对齐矩阵。

### 经典注意力仍有价值的场景

- 教学。单头、单层、基于 RNN 的版本让每个概念变得可见。
- 设备端的序列任务，当 Transformer 无法部署时。
- 阅读 2014–2017 年间的论文。不了解 Bahdanau 的约定会导致误读。
- 机器翻译中的精细对齐分析。原始注意力权重作为解释工具仍有用，即便在 Transformer 模型上，阅读它们也需要知道它们代表什么。

### 把注意力权重当作解释的陷阱

注意力权重看起来可解释。它们在位置间求和为一；可以绘图；高权重意味着“看了这个位置”。审稿人很爱它们。

但它们并不像表面看起来那么可解释。Jain 和 Wallace（2019）表明，在某些任务中，注意力分布可以被置换或替换为任意替代分布而不改变模型预测。没有做消融或反事实检验，就不要把注意力权重当作推理证据来报告。

## 交付

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. 简单。实现带掩码的 `softmax`，使编码器中的填充 token 的注意力权重为零。在一个包含可变长度序列的批次上测试。
2. 中等。将多头注意力加入 Luong `general` 形式。将 `d_h` 划分为 `n_heads` 组，对每个头分别运行注意力，再拼接输出。验证单头情况与你之前的实现一致。
3. 困难。在第 09 课的玩具复制任务上训练一个带 Bahdanau 注意力的 GRU 编码器-解码器。绘制准确率随序列长度变化的曲线。与无注意力基线比较。你应看到随着长度增长，二者差距扩大，确认注意力缓解了瓶颈。

## 关键词

| 术语 | 常说的意思 | 实际含义 |
|------|-----------|---------|
| Attention | 看哪儿 | 对值序列的加权平均，权重由 query-key 相似度计算得到。 |
| Query, Key, Value | QKV | 三个投影：Q 提问，K 是用于匹配的内容，V 是要返回的内容。 |
| Additive attention | Bahdanau | 前馈得分：`v^T tanh(W q + U k)`。 |
| Multiplicative attention | Luong dot / general | 得分为 `q^T k` 或 `q^T W k`。计算更便宜，在大多数任务上有相似精度。 |
| Alignment matrix | 漂亮的图 | 注意力权重，形状为 `(T_dec, T_enc)` 的网格。用它来观察模型关注了哪些位置。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 原论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种得分变体及比较。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 关于可解释性的注意事项。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 可运行的 PyTorch 演练。