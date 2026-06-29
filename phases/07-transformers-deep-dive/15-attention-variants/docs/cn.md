# Attention Variants — Sliding Window, Sparse, Differential

> Full attention is a circle. Every token sees every token, and memory pays the price. Four variants bend the shape of the circle and recover half the cost.

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 7 · 02（自注意力）、Phase 7 · 03（多头）、Phase 7 · 12（KV Cache / Flash Attention）  
**Time:** ~60 分钟

## 问题

Full attention 在序列长度上消耗 `O(N²)` 的内存和 `O(N²)` 的计算。对于一个 128K 上下文的 Llama 3 70B，每层的注意力条目是 160 亿，乘以 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 的激活内存，但不改变算术成本——每个 token 仍然要对每个其他 token 做注意力。

有三类变体直接改变注意力矩阵的拓扑结构：

1. **Sliding window attention (SWA)。** 每个 token 只关注固定窗口内的邻居，而不是整个前缀。内存和计算降为 `O(N · W)`，其中 `W` 是窗口大小。常见于 Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long。
2. **Sparse / block attention。** 只对选定的 `(i, j)` 对进行打分；其余位置被强制为零权重。代表有 Longformer、BigBird、OpenAI sparse transformer。
3. **Differential attention。** 用独立的 Q/K 投影计算两个注意力图，然后相减。能消除将权重“吸”到最前几个 token 的 attention sink 问题。微软的 DIFF Transformer（2024）。

这些方法可以共存。到 2026 年的前沿模型通常混合使用：大多数层是 SWA-1024，每五层有一层全局全注意力，还有少数差分头用来清理检索噪声。Gemma 3 的 5:1 SWA-to-global 比例已成为当前教材默认配置。

## 概念

### Sliding Window Attention (SWA)

位置 `i` 的每个 query 只关注 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）。窗口外的 token 在得分矩阵中被设为 `-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192` 且 `W = 1024`，得分矩阵期望非零行数为 1024 × 8192——约 8× 的减少。

KV cache 在 SWA 下也会缩小。每层只需保留最近 `W` 个 token 的 K 和 V。以 Gemma-3 式配置（窗口 1024，128K 上下文）为例，KV cache 可降低 128×。

质量代价：仅使用 SWA 的 Transformer 在长距离检索上表现较弱。解决办法是交错使用 SWA 层和全注意力层。Gemma 3 使用 5:1 的 SWA:global 比例。Mistral 7B 使用了因果 SWA 堆栈，信息通过重叠窗口“向前流动”——每层将有效感受野扩展 `W`，经过 `L` 层后模型可以回溯 `L × W` 个 token。

### Sparse / Block Attention

预先选择一个 `N × N` 的稀疏模式。三种典型形状：

- **Local + strided（OpenAI sparse transformer）。** 关注最近 `W` 个 token，加上在此之前每隔 `stride` 个 token 的位置。既捕捉局部也能捕捉长程，计算复杂度接近 `O(N · sqrt(N))`。
- **Longformer / BigBird。** 局部窗口 + 一小组全局 token（例如 `[CLS]`），这些全局 token 同时关注所有位置并被所有位置关注 + 随机稀疏连边。在匹配质量下经验上能带来约 2× 的上下文扩展。
- **Native Sparse Attention（DeepSeek, 2025）。** 学习哪些 `(Q, K)` 块是重要的；在内核层面跳过全零块。兼容 FlashAttention。

Sparse attention 实际上是个内核工程问题。数学上很简单（对得分矩阵做掩码）；胜利来自于从不把零条目加载到 SRAM。FlashAttention-3 和 2026 年的 FlexAttention API 让自定义稀疏模式在 PyTorch 中成为一等公民。

### Differential Attention（DIFF Transformer，2024）

常规注意力存在“attention sink”问题：softmax 强制每行和为 1，于是那些不想关注任何特定内容的查询会把权重倾倒到第一个 token（或前几个 token）上。这会窃取本该分配给真实内容的容量。

差分注意力通过计算两个注意力图并相减来修复此问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是一个可学习的标量（通常在 0.5–0.8 之间）。A1 捕捉真实内容的权重；A2 捕捉 sink。相减可以抵消 sink，将权重重新分配到相关的 token 上。

微软（2024）报告的结果：困惑度降低 5–10%，在相同训练长度下有效上下文延长 1.5–2×，对“haystack 中的针”检索更加锐利。

### 变体比较

| Variant | Compute | KV cache | Quality vs full | Production use |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | O(N) per layer | baseline | every model's default layer |
| SWA (window 1024) | O(N·W) | O(W) per layer | -0.1 ppl, good with global layers | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | mixed | similar to SWA | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | O(N) approx | mixed | matches full at 2× context | early long-context BERT |
| Native Sparse (DeepSeek-V3.2) | O(N · active fraction) | O(N) | within 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5 to -10% ppl | DIFF Transformer, early 2026 models |

```figure
gqa-kv-sharing
```

## 实现

见 `code/main.py`。我们实现了一个因果掩码比较器，能在一个小序列上并排显示 full、SWA、local+strided 和 differential attention 的掩码。

### 步骤 1：全因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

来自第 07 课的基线。下三角；对角线以上为负无穷。

### 步骤 2：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数 —— `window`。当 `window >= n` 时，你会恢复为完全相同的全因果注意力。当 `window = 1` 时，每个 token 只关注它自己。

### 步骤 3：局部 + 步幅稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集的局部窗口加上向序列起始方向每隔 `stride` 个 token 取一次。额外层数会以对数步长扩展感受野。

### 步骤 4：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力传递，用可学习的混合系数相减。在代码中我们比较单一注意力与差分注意力的 attention-sink 热力图，并观察 sink 的收敛消失。

### 步骤 5：KV cache 大小

在 `N = 131072` 时打印每层的缓存大小。SWA 和稀疏变体能下降 10–100×。差分注意力会翻倍。请在设计时有意识地支付你的内存账单。

## 使用

2026 年生产模式示例：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 在 5:1 的比例上混合 SWA（window=1024）和全局层。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 的 FlexAttention 接受一个掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译成一个自定义的 Triton 内核。对于常见模式性能接近 FlashAttention-3 的 10% 范围内，且掩码函数是一个 Python 可调用对象。

何时选择哪种方案：

- 纯全注意力 —— 对于最大 ~16K 上下文或当检索质量至关重要时使用。
- SWA + 全局混合 —— 针对超长上下文（>32K），训练与推理受内存限制时使用。2026 年在 32K 以上的默认配置。
- 稀疏块注意力 —— 需要自定义内核和自定义模式的场景。通常用于专门工作负载（检索、音频）。
- 差分注意力 —— 任何受 attention-sink 污染影响的工作负载（长上下文 RAG、needle-in-haystack）都适用。

## 部署

见 `outputs/skill-attention-variant-picker.md`。该技能会基于目标上下文长度、检索需求以及训练/推理的计算配置，为新模型挑选注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 的 SWA 将每行中最后 4 个 token 之外的位置置零（即 `-inf`）。验证 `window=n` 位于位级上复现全因果注意力。
2. **中等。** 在第 07 课的 capstone 上实现因果 SWA（`window=1024`）。在 tinyshakespeare 上训练 1,000 步。验证验证集损失相对于全注意力回退多少？峰值内存下降了多少？
3. **困难。** 在 capstone 模型中实现 Gemma-3 风格的 5:1 层混合（5 层 SWA，1 层 global）。在匹配参数下比较纯 SWA、纯 global 与混合的损失、内存和生成质量。
4. **困难。** 实现带每 head 可学习 `λ` 的差分注意力。在一个合成检索任务上训练（1 个 needle，2,000 个干扰项）。在匹配参数下，测量相对于单一注意力基线的检索准确率。

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | "Local attention" | 每个 query 只关注其最近的 `W` 个 token；KV cache 缩减到 `O(W)`。 |
| Effective receptive field | "How far back the model sees" | 在一个有 `L` 层、窗口为 `W` 的 SWA 堆栈中，可见范围最高为 `L × W` 个 token。 |
| Longformer / BigBird | "Local + global + random" | 含少量始终关注的全局 token 的稀疏模式；早期的长上下文方法。 |
| Native Sparse Attention | "DeepSeek's kernel trick" | 学习块级稀疏；在内核层面跳过全零块，同时保持质量。 |
| Differential attention | "Two maps, one subtracts" | DIFF Transformer：从第一张注意力图中减去第二张乘以可学习 `λ` 的图以抵消 attention sink。 |
| Attention sink | "Weight bleeds to token 0" | softmax 规范化强制每行和为 1；无信息的查询会把权重倾倒到位置 0。 |
| FlexAttention | "Mask-as-Python" | PyTorch 2.5+ 的 API：将任意掩码函数编译为类似 FlashAttention 的内核。 |
| Layer type mix | "5:1 SWA-to-global" | 在堆栈中交错稀疏层与全注意力层，以在更低内存下保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 经典的 sliding-window + global-token 论文。  
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — local + global + random。  
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的 local+strided 模式。  
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:global 的讨论。  
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 描述了窗口为 1024 且 5:1 混合的配置，该配置已成为教材默认。  
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 论文。  
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的学习稀疏注意力。  
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — Use It 部分所示的 mask-as-callable 模式的 API 参考。