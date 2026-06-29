# 多头注意力

> 一个注意力头一次学习一种关系。八个头学习八种。头是廉价的。多用它们。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 7 · 02（从头实现自注意力）
**Time:** ~75 分钟

## 问题

单个自注意力头计算一个注意力矩阵。那个矩阵捕捉一种关系——通常是使训练信号上的损失最小化的那种。如果你的数据把主谓一致、共指、长程话语和句法分块都纠结在一起，单个头会把它们揉成一个 softmax 分布，从而丢失一半的信号。

2017 年 Vaswani 论文的修正方法：并行运行若干注意力函数，每个都有自己的 Q、K、V 投影，然后将输出串联。每个头在维度为 `d_model / n_heads` 的更小子空间中工作。总参数量保持不变。表示能力提高。

到 2026 年为止，多头注意力是每个 Transformer 的默认配置。争论的只是要多少头，以及键和值是否共享投影（Grouped-Query Attention、Multi-Query Attention、Multi-head Latent Attention）。

## 概念

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**拆分。** 取形状为 `(N, d_model)` 的 `X`。投影到 Q、K、V，均为形状 `(N, d_model)`。重塑为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。转置为 `(n_heads, N, d_head)`。

**并行注意力。** 在每个头内运行缩放点积注意力。每个头产生 `(N, d_head)`。各头在嵌入的不同子空间上操作，并且在注意力计算本身期间互不通信。

**串联并投影。** 把各头堆回 `(N, d_model)` 并乘以学习得到的输出矩阵 `W_o`，形状为 `(d_model, d_model)`。`W_o` 是头之间进行混合的地方。

**为什么有效。** 每个头可以专门化而不与其他头争夺表示预算。2019–2024 年的探测研究显示了不同头的角色：位置头、关注前一 token 的头、复制头、命名实体头、归纳头（支持上下文学习）。

**2026 年的变体谱系：**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA 是现代默认，因为它将 KV-cache 内存按 `N/G` 的因子缩减，同时几乎保持完整质量。MLA 更进一步：把 K/V 压缩到潜在空间，然后在计算时投影回去——增加 FLOPs，但节省更多内存。

```figure
multihead-split
```

## 实现

### 步骤 1：从我们已有的单头注意力拆分头

将 Lesson 02 中的 `SelfAttention` 包裹一对拆分/合并操作。参见 `code/main.py` 中的 numpy 实现；逻辑如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)  形状

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape 和一次 transpose。没有循环。这正是 PyTorch 在 `nn.MultiheadAttention` 下做的。

### 步骤 2：对每个头运行缩放点积注意力

每个头得到自己的 Q、K、V 切片。注意力变为一个批量化的矩阵乘法：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)  形状
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)  形状
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上 `Qh @ Kh.transpose(...)` 是一次 `bmm`。GPU 会看到一次批量矩阵乘，形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`。增加头是廉价的。

### 步骤 3：Grouped-Query Attention 变体

只有键和值的投影不同。Q 有 `n_heads` 组；K 和 V 有 `n_kv_heads < n_heads` 组，然后重复以匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)  形状
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)  形状
```

在推理时这会节省内存，因为 KV 缓存中只有 `n_kv_heads` 个副本，而不是 `n_heads`。Llama 3 70B 使用 64 个查询头和 8 个 KV 头——缓存缩小了 8 倍。

### 步骤 4：探测每个头学到了什么

对一个短句用 4 个头运行 MHA。对每个头，打印 `(N, N)` 的注意力矩阵。你会看到不同头即使在随机初始化下也选择不同结构——这部分是信号，部分是子空间中的旋转对称性。

## 使用

在 PyTorch 中，一行就能搞定：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention 在 CUDA 上会自动调度 Flash Attention。
# 对于 GQA，传入 Q 的形状为 (B, n_heads, N, d_head)，K,V 的形状为
# (B, n_kv_heads, N, d_head)。PyTorch 会处理重复操作。
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**多少个头？** 2026 年生产模型的经验法则：

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` 几乎总是落在 64 或 128。它是一个头能够“看到”多少信息的单位。低于 32 时，头开始与缩放因子 `sqrt(d_head)` 产生冲突；高于 256 时，你会失去“许多小专家”的好处。

## 部署

参见 `outputs/skill-mha-configurator.md`。该技能会根据参数预算、序列长度和部署目标，为新 Transformer 推荐头数、kv-head 数和投影策略。

## 练习

1. **简单。** 在 `code/main.py` 中把 MHA 的 `n_heads` 从 1 改为 16，同时固定 `d_model=64`。在一个合成复制任务上绘制一个小型单层模型的损失曲线。更多头是有帮助、到平台期，还是有害？
2. **中等。** 实现 MQA（所有查询头共享一个 KV 头）。测量与完整 MHA 相比参数量下降了多少。对于 N=2048，计算推理时 KV-cache 大小缩小了多少。
3. **困难。** 实现一个微型版的 Multi-head Latent Attention：把 K、V 压缩到秩为 `r` 的潜在空间，潜在表示存入 KV 缓存，注意力时再解码。在哪个 `r` 值下，缓存内存降到低于完整 MHA 的 1/8，同时质量在验证 ppl 上保持在 1 bit 以内？

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | "A single attention circuit" | One Q/K/V projection of dimension `d_head = d_model / n_heads` with its own attention matrix. |
| d_head | "Head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose around attention. |
| W_o | "Output projection" | `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "One KV head" | Multi-Query Attention: single shared K/V projection. Smallest KV cache, some quality loss. |
| GQA | "The default since Llama 2" | Grouped-Query Attention with `n_kv_heads < n_heads`; repeats to match Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V compressed to low-rank latent, decompressed at attend time. |
| Induction head | "The circuit behind in-context learning" | A pair of heads that detect previous occurrences and copy what followed them. |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — the original multi-head spec.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — the MQA paper.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — how to convert MHA to GQA after training.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA and why it beats MHA/GQA on cache memory.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — mechanistic look at what heads actually do.