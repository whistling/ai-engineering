# Positional Encoding — Sinusoidal, RoPE, ALiBi

> Attention 对顺序不敏感。没有位置信号时，“The cat sat on the mat”和“mat the on sat cat the”会产生相同的输出。三种算法解决了这个问题——它们对“位置”这个概念做出了不同的押注。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 7 · 02（自注意力）、Phase 7 · 03（多头注意力）  
**Time:** ~45 分钟

## 问题

缩放点积注意力（scaled dot-product attention）对顺序是盲目的。注意力矩阵 `softmax(Q K^T / √d) V` 是由成对相似度计算得到的。把输入 `X` 的行打乱，输出的行也会以相同方式被打乱。注意力内部没有任何东西关心绝对位置。

这在词袋模型中不是个 bug。但对于语言、代码、音频、视频等任何顺序携带语义的场景，这就是致命的。

解决办法是以某种方式把位置注入到嵌入中。三代解法：

1. **绝对正弦（Absolute sinusoidal）**（Vaswani 2017）。把位置的 `sin/cos` 加到嵌入上。简单、不需要学习参数，但在训练长度之外外推能力差。
2. **RoPE — Rotary Position Embeddings**（Su 2021）。按与位置成比例的角度旋转 Q 和 K 向量。在点积中直接编码相对位置。到 2026 年已占主导地位。
3. **ALiBi — Attention with Linear Biases**（Press 2022）。完全跳过位置嵌入；在注意力分数上根据距离添加每个 head 的线性惩罚。对长度外推非常好。

截至 2026 年，几乎所有前沿开源模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦目前主要是历史遗留。

## 概念

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### 绝对正弦

预先计算一个固定矩阵 `PE`，形状为 `(max_len, d_model)`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力前做 `X' = X + PE[:N]`。每个维度是不同频率的正弦波。模型通过相位模式来读取位置。在超过 `max_len` 的位置会失败：当模型只见过 0–2047 的位置时，没人告诉它位置 2048 会发生什么。

### RoPE

对 Q 和 K 向量（而不是嵌入）做旋转。对每一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

对键（keys）亦以其位置 `pos_k` 应用相同的旋转。点积 `q'_m · k'_n` 变成仅依赖 `(m - n)` 的函数。也就是说：**注意力分数只依赖相对距离**，尽管旋转是基于绝对位置的。妙极了。

扩展 RoPE：可以缩放 `base`（NTK-aware、YaRN、LongRoPE）以在不重训练的情况下外推到更长上下文。Llama 3 就通过这种方式把 8K 扩展到了 128K。

### ALiBi

跳过嵌入技巧，直接对注意力分数加偏置：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是 head 特定的斜率（例如 `1 / 2^(8·h/H)`）。靠近的 token 会被提升；远处的 token 会被惩罚。训练时没有额外成本。论文显示，长度外推性能优于正弦位置编码，并在其训练长度上与 RoPE 相当。

### 2026 年该怎么选

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | 差 | 免费 | 原始 Transformer、早期 BERT |
| Learned absolute | 无 | 很小 | GPT-2、GPT-3 |
| RoPE | 通过缩放可获得良好外推 | 免费 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 外推效果极好 | 微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 极好 | 免费 | BLOOM、MPT、Baichuan |

RoPE 胜出是因为它可以嵌入到注意力机制中而不改变架构，直接编码相对位置，并且其 `base` 超参数为长上下文微调提供了一个干净的调节旋钮。

```figure
rope-explorer
```

## 实现

### 步骤 1：正弦编码

见 `code/main.py`。四行计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前把它加到嵌入矩阵上。

### 步骤 2：对 Q、K 应用 RoPE

RoPE 就地作用于 Q 和 K。对每一对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键点：对位置为 `m` 的 Q 和位置为 `n` 的 K 应用相同的函数。它们的点积在每对坐标上都会得到一个 `cos((m-n)·θ_i)` 因子。注意力因此免费学到了相对位置。

### 步骤 3：ALiBi 斜率与偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # 将其加到 softmax 之前的注意力分数上
```

（注：代码注释已翻译）将 `bias[h]` 加到 head `h` 的 `(seq_len, seq_len)` 注意力分数矩阵上，然后做 softmax。

### 步骤 4：验证 RoPE 的相对距离不变性

选两个随机向量 `a, b`。分别按 `(pos_a, pos_b)` 旋转。再按 `(pos_a + k, pos_b + k)` 旋转。两次的点积在浮点误差范围内应当相同。这个性质就是 RoPE 的全部意义：对绝对偏移不敏感，仅依赖相对间隔。

## 使用

PyTorch 2.5+ 在 `torch.nn.functional` 中提供了 RoPE 的工具。大多数生产代码使用 `flash_attn` 或 `xformers`，RoPE 通常在注意力内核中应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware 插值。** 在将上下文从 4K 扩展到 16K+ 时，把 `base` 按 `base * (scale_factor)^(d/(d-2))` 重缩放。
- **YaRN。** 更聪明的插值方法，可在长上下文上保持注意力熵。Llama 3.1 128K 使用了它。
- **LongRoPE。** Microsoft 2024 年的方法，使用进化搜索为每维选择缩放因子。Phi-3-Long 使用了它。
- **位置插值 + 微调。** 直接按扩展因子缩小位置并用 1–5B tokens 微调。出乎意料地有效。

## 上线

见 `outputs/skill-positional-encoding-picker.md`。该技能会根据目标上下文长度、外推需求和训练预算，为新模型挑选编码策略。

## 练习

1. 简单：把 `max_len=512, d=128` 的正弦 `PE` 矩阵作热力图。确认“随着维度索引增大条纹变宽”的模式。
2. 中等：实现 NTK-aware RoPE 缩放。在序列长度 256 上训练一个小型语言模型，然后在长度 1024 上测试有无缩放时的困惑度（perplexity）。
3. 困难：在同一个注意力模块中同时实现 ALiBi 和 RoPE。用序列长度 512 的复制任务训练 4 层 Transformer。在测试时外推到 2048，比较性能下降。

## 关键词

| 术语 | 俗称 | 实际含义 |
|------|------|---------|
| 位置编码（Positional encoding） | “告诉注意力顺序” | 在嵌入或注意力上添加的任何编码位置信息的信号。 |
| 正弦（Sinusoidal） | “原始方法” | 以几何频率的 sin/cos 加到嵌入上；不能良好外推。 |
| RoPE | “Rotary embeddings” | 通过基于位置的角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | “线性偏置技巧” | 在注意力分数上加 `-m·\|i-j\|`；不需要嵌入，外推性好。 |
| base | “RoPE 的旋钮” | RoPE 中的频率缩放参数；增大可在推理时扩展上下文。 |
| NTK-aware | “RoPE 的缩放技巧” | 在扩展上下文时重缩放 `base`，以防高频维度被压缩。 |
| YaRN | “更精细的插值方法” | 保持注意力熵的按维插值+外推方案。 |
| 外推（Extrapolation） | “能在训练长度之外工作吗” | 在训练见到的 max_len 之外，位置方案是否仍能产生正确输出？ |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始的正弦位置编码。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — RoPE 缩放的最先进方法。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Microsoft 的方法，被 Phi-3-Long 等模型采用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 生产级别的各种 RoPE 缩放实现（默认、线性、动态、YaRN、LongRoPE、Llama-3）。