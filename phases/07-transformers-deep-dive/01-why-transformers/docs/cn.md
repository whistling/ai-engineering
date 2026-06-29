# 为什么选择 Transformer — RNN 的问题

> RNN 逐个 token 处理。Transformer 同时处理所有 token。这个单一的架构赌注在 2017 年之后改变了深度学习中每一条扩展曲线。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 3（深度学习核心）、Phase 5 · 09（序列到序列）、Phase 5 · 10（注意力机制）  
**Time:** ~45 分钟

## 问题所在

在 2017 年之前，地球上所有最先进的序列模型 —— 语言、翻译、语音 —— 都是循环神经网络。LSTM 和 GRU 在类 ImageNet 的翻译基准上称霸了半个十年。那是当时唯一的工具。

它们有三个致命弱点。序列化计算意味着无法沿时间轴并行化：token `t+1` 需要来自 token `t` 的隐藏状态。一个 1,024 token 的序列意味着在一块能够每个周期执行 1,000,000 次浮点运算的 GPU 上需要执行 1,024 个串行步骤。训练的实际耗时会随序列长度线性增长，而这些硬件本为并行而设计。

梯度消失意味着 50 个 token 之前的信息已经被 50 次非线性压缩。门控循环单元（LSTM、GRU）缓和了这种压缩，但从未消除它。长程依赖——“我去年夏天在去京都的飞机上读的那本书是……”——经常失败。

固定宽度的隐藏状态意味着编码器在解码器看到任何内容之前，把整个源序列压进了一个向量。不管源序列是 5 个 token 还是 500 个；瓶颈形状相同。

2017 年的论文 “Attention Is All You Need” 提出了一个激进的想法：完全放弃循环。让每个位置并行地关注每个其他位置。用一次大的矩阵乘替代 1,024 次串行乘法。

到 2026 年，这一结果主导了所有模态。语言（GPT-5、Claude 4、Llama 4）、视觉（ViT、DINOv2、SAM 3）、音频（Whisper）、生物（AlphaFold 3）、机器人（RT-2）。同一个模块，不同的输入。

## 概念

![RNN 串行计算 vs Transformer 并行注意力](../assets/rnn-vs-transformer.svg)

**循环作为瓶颈。** 一个 RNN 计算 `h_t = f(h_{t-1}, x_t)`。每一步依赖于前一步。你不能在计算 `h_4` 之前计算 `h_5`。在拥有 10,000+ 并行核心的现代 GPU 上，这在长序列上浪费了 99% 的硅片资源。

**注意力作为广播。** 自注意力为每一对 `(i, j)` 同时计算 `output_i = sum_j(a_ij * v_j)`。整个 N×N 的注意力矩阵可以在一次批处理矩阵乘中填满。没有一步依赖另一部。GPU 非常喜欢这种方式。

**加速不是常数。** 这是 `O(N)` 串行深度与 `O(1)` 串行深度的差别。实际上，在匹配硬件上，N=512 时 Transformer 在每个 epoch 的训练速度上通常快 5–10×，并且随着序列长度的增加差距还会扩大，直到你遇到注意力的 `O(N²)` 内存墙（后来的 Flash Attention 解决了常数问题——见第 12 课）。

**Transformer 的代价。** 注意力的内存随 `O(N²)` 增长。对于 2K 上下文还好。对于 128K 上下文，你需要滑动窗口、RoPE 外推、Flash Attention 贴块，或线性注意力变体。循环在时间和内存上都是 `O(N)`；Transformer 用内存换时间，然后通过并行性把时间赢回来。

**归纳偏置的转变。** RNN 假设局部性和时序上的新近性。Transformer 不做这些假设——每一对都是注意候选。这就是为什么 Transformer 需要更多数据才能训练得好，但一旦拥有足够的数据就能进一步扩展。Chinchilla（2022）形式化了这一点：在足够的 token 下，参数量相等的 Transformer 总是胜过 RNN。

## 动手实践

这里不构建神经网络 —— 我们数值模拟核心瓶颈，让你在笔记本上感受差距。

### 第 1 步：测量串行深度

见 `code/main.py`。我们构建两个函数。一个把序列编码为加法链（串行，像 RNN）。一个把序列编码为并行归约（广播，像注意力）。数学相同，依赖图不同。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # 不能并行：h 依赖于前一个 h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # 每个 x 都独立
```

我们对长度最长到 100,000 的序列进行计时。RNN 版本是 O(N) 的单一 CPU 流水线。即便在纯 Python 中，注意力风格的归约在长度 ≥ 1,000 时也会胜出，因为 Python 的 `sum()` 在 C 中实现，可以在每一步避免解释器开销。

### 第 2 步：统计理论运算量

两个算法都做了 N 次加法。不同之处在于依赖深度：在开始下一步之前必须顺序完成多少操作。RNN 深度 = N。注意力深度 = 使用树形归约时为 log(N)，或使用并行扫描时为 1。决定 GPU 时间的是深度，而不是操作计数。

### 第 3 步：对长序列的实测扩展

我们打印一个计时表以展示 O(N) 的差距。在 2026 年款 Mac 笔记本上，长度低于 1,000 的序列太快而难以测量。100,000 的序列表现出清晰的线性扫描。把这放缩到具有 12 层 LSTM 等价物的 16,384 token Transformer，你就会看到为什么在 2016 年训练的实时时钟是阻碍因素。

## 何时仍选择 RNN（到 2026 年）

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

State-space models（SSM）如 Mamba 本质上是带有结构化参数化的 RNN，使其两者兼得：`O(N)` 扫描内存，通过选择性扫描实现并行训练。它们在更长上下文的扩展上以更好的方式恢复了大约 90% 的 Transformer 质量。到 2026 年，大多数前沿实验室训练混合的 SSM+transformer 模型（例如 Jamba、Samba）——循环并未死去，它是一个组件。

## 部署建议

见 `outputs/skill-architecture-picker.md`。该技能根据长度、吞吐量和训练预算约束为新的序列问题挑选架构。对于训练运行超过 1B token 的情形，它应始终拒绝推荐纯 RNN，除非明确说明权衡。

## 练习

1. **简单。** 取 `code/main.py` 中的 `rnn_style`，将标量隐藏状态替换为长度为 64 的隐藏状态向量。重新测量。串行开销随隐藏状态维度增长了多少？
2. **中等。** 在纯 Python 中实现并行前缀和（Hillis–Steele scan）。验证在长度 1024 时它产生与串行扫描相同的数值输出。计算其深度。
3. **困难。** 将注意力风格的归约移植到 PyTorch 的 GPU 上。随序列长度从 64 扫描到 65,536 时对两者计时。绘图并解释曲线形状。

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | 计算的第 t 步依赖第 t-1 步，强制沿时间轴串行执行。 |
| Serial depth | "How deep the graph is" | 依赖最长链；即使在无限硬件上也会限制实际耗时。 |
| Attention | "Let tokens look at each other" | 加权和 `sum_j a_ij v_j`，其中 `a_ij` 来自位置 i 与 j 之间的相似度评分。 |
| Context window | "How much the model sees" | 注意力层能接受的输入位置数量；二次内存代价在此处体现。 |
| Inductive bias | "Assumptions baked into the architecture" | 关于数据形态的先验；CNN 假设平移不变，RNN 假设新近性。 |
| State-space model | "RNN with algebra behind it" | 通过结构化的状态空间矩阵参数化的递归，使并行训练成为可能。 |
| Quadratic bottleneck | "Why context costs so much" | 注意力内存 = `O(N²)`；Flash Attention 隐藏了常数项，但并未改变规模。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 杀死主流 NLP 中循环的论文。  
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力的诞生，将其固定到 RNN 上。  
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — 原始 LSTM 论文，供存档查看。  
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — 面向 Transformer 的现代循环式答案。