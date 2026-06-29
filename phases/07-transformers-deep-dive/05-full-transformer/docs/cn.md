# The Full Transformer — Encoder + Decoder

> 注意力是主角。其他一切——残差、归一化、前馈、交叉注意力——都是让你能够堆叠它的脚手架。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (自注意力), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (位置编码)
**Time:** ~75 分钟

## 问题

单层注意力是一个特征提取器，而不是一个完整模型。每层一次矩阵乘法对于语言建模来说容量不够。你需要深度——而没有合适的管道，深度就会崩塌。

2017 年 Vaswani 的论文将把单层注意力变成可堆叠模块的六个设计决策打包了起来。从那以后所有的 transformer——仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5）——都继承了相同的骨架。到 2026 年这些模块被改进（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架保持不变。

本课就是这个骨架。后续课程会对它进行专门化——06 为编码器，07 为解码器，08 为编码器-解码器。

## 概念

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### 六个部分

1. **Embedding + 位置信号。** 将 token 映射到向量。位置信息通过 RoPE（现代）或正弦（经典）注入。
2. **自注意力。** 每个位置对所有其他位置进行注意力。解码器中会加掩码。
3. **前馈网络（FFN）。** 按位置的两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比为 4×。
4. **残差连接。** `x + sublayer(x)`。没有它，梯度在 ~6 层后会消失。
5. **层归一化。** `LayerNorm` 或现代的 `RMSNorm`。稳定残差流。
6. **交叉注意力（仅解码器）。** Query 来自解码器，Key/Value 来自编码器输出。

观察一个向量通过一个 block 的流动：注意力在位置间混合信息，残差将其携带前行，FFN 对其进行变换，归一化保持流的稳定。

```figure
transformer-block
```

### 编码器块（BERT、T5 编码器使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的。无掩码。所有位置都能看到所有位置。

### 解码器块（GPT、T5 解码器使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每个块有三个子层。中间的子层——交叉注意力——是唯一将信息从编码器流向解码器的地方。在纯粹的仅解码器架构（GPT）中，交叉注意力被省略，仅有带掩码的自注意力 + FFN。

### Pre-norm 与 post-norm

原始论文：`x + sublayer(LN(x))` 与 `LN(x + sublayer(x))`。post-norm 在 2019 年左右失宠——如果没有精心的 warmup，很难训练得很深。pre-norm（在子层之前做 `LN`）是 2026 年的默认：Llama、Qwen、GPT-3+、Mistral 都采用它。

### 2026 年现代化的模块

Vaswani 2017 使用 LayerNorm + ReLU。现代的堆栈替换了两者。生产环境中的模块通常长这样：

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU 使用三个矩阵，总参数量相当) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (或 MLA) |
| Bias terms | Yes | No |

RMSNorm 去掉了 LayerNorm 的均值中心化（少做一次减法），从而节省计算，并且在经验上至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 的论文中相较 ReLU/GELU FFN 在困惑度上稳定提升约 0.5 点。

### 参数量

对于一个 block，设 `d_model = d`，FFN 扩展比为 `r`：

- MHA: `4 · d²`（Q、K、V、O 投影）
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- 归一化：可忽略

在 `d = 4096, r = 2.6, layers = 32`（大致相当于 Llama 3 8B）时，总量为：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 参数/层 × 32 ≈ 7B`（加上嵌入和 head）。与已发布的计数相符。

## 构建它

### 步骤 1：构建模块

使用 Lesson 03 中的微小 `Matrix` 类（已复制到本文件以便独立运行）：

- `layer_norm(x, eps=1e-5)` — 减去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。没有均值减法。
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

完整接线见 `code/main.py`。

### 步骤 2：接线一个 2 层编码器和 2 层解码器

将它们堆叠。将编码器输出传入每一个解码器的交叉注意力。输出投影之前加一个最终的 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 步骤 3：在玩具例子上运行前向

将 6 个 token 的源序列和 5 个 token 的目标序列送入。验证输出形状为 `(5, vocab)`。不训练——本课关注的是架构，而不是损失。

### 步骤 4：替换为 RMSNorm + SwiGLU

用 RMSNorm 和 SwiGLU 替换 LayerNorm 和 ReLU-FFN。确认形状仍然匹配。这是一处函数替换完成的 2026 年现代化。

## 使用它

PyTorch/TF 的参考实现：`nn.TransformerEncoderLayer`，`nn.TransformerDecoderLayer`。但大多数 2026 年的生产代码都会自己实现 block，因为：

- Flash Attention 在 attention 内部被调用，而不是通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考实现中。
- RoPE、RMSNorm、SwiGLU 也不是 PyTorch 的默认。

HF `transformers` 有清晰的参考 block 实现，值得阅读：`modeling_llama.py` 是 2026 年的典型仅解码器 block。大约 500 行，值得细读一次。

**何时选择 编码器 / 解码器 / 编码器-解码器：**

| Need | Pick | Example |
|------|------|---------|
| 分类、嵌入、基于文本的 QA | 仅编码器 | BERT, DeBERTa, ModernBERT |
| 文本生成、聊天、代码、推理 | 仅解码器 | GPT, Llama, Claude, Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5, BART, Whisper |

仅解码器在语言建模上赢得了主流，因为它在扩展性上最干净，并且同时支持理解与生成。当输入有明确的“源序列”身份（翻译、语音识别、结构化任务）时，编码器-解码器仍然是最佳选择。

## 上线

见 `outputs/skill-transformer-block-reviewer.md`。该技能会将新的 transformer block 实现与 2026 年默认项进行对照，标记缺失的部分（pre-norm、RoPE、RMSNorm、GQA、FFN 扩展比）。

## 练习

1. **简单。** 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 下统计你实现的 `encoder_block` 的参数量。通过实现该 block 并使用 `sum(p.numel() for p in block.parameters())` 验证。
2. **中等。** 将 post-norm 切换为 pre-norm。初始化两者并在随机输入上测量 12 层堆叠后的激活范数。post-norm 的激活应该会爆炸；pre-norm 的应保持有界。
3. **困难。** 在一个玩具的复制任务上实现 4 层的编码器-解码器（将 `x` 反向复制）。训练 100 步并报告损失。替换为 RMSNorm + SwiGLU + RoPE——损失是否下降？

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | 一个 transformer 层 —— 由归一化 + 注意力 + 归一化 + FFN 组成，并被残差连接包裹。 |
| Residual | "Skip connection" | `x + f(x)` 输出；使深层堆栈中的梯度流动成为可能。 |
| Pre-norm | "Normalize before, not after" | 现代做法：`x + sublayer(LN(x))`。无需复杂的 warmup 即可训练更深的网络。 |
| RMSNorm | "LayerNorm without the mean" | 用 RMS 除法；少一次操作，但经验上稳定性相同。 |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`。在语言建模困惑度上优于 ReLU/GELU。 |
| Cross-attention | "How the decoder sees the encoder" | Query 来自解码器，K/V 来自编码器输出的 MHA。 |
| FFN expansion | "How wide the middle MLP is" | 隐层相对于 d_model 的扩展比，通常为 4（LayerNorm）或 2.6（SwiGLU）。 |
| Bias-free | "Drop the +b terms" | 现代堆栈省略线性层的偏置项；略微改善困惑度且模型更小。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始的 block 规范。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么 pre-norm 在深层中优于 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 典型的 2026 年仅解码器 block。