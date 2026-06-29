# 从零构建一个 Transformer — 毕业项目

> 十三课。一个模型。没有捷径。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** 第 7 阶段 · 第 01 到 13 课。不要跳过。  
**Time:** ~120 分钟

## 问题描述

你已经读完了所有论文。你已经实现了 attention、多头拆分、位置嵌入、编码器和解码器模块、BERT 与 GPT 损失、MoE、KV cache。现在把它们整合到真实任务上，让它们协同工作。

毕业项目：端到端训练一个小型的仅解码器（decoder-only）transformer，用于字符级语言建模任务。它读取莎士比亚文本，生成新的莎士比亚式文本。模型足够小，可以在笔记本上在不到 10 分钟内训练完。它足够正确，把数据集换大、训练时间延长后能成为真正的语言模型。

这是本课程的 “nanoGPT”。并非原创 — Karpathy 的 2023 年 nanoGPT 教程是每个学生至少实现一次的参考实现。我们保留其整体结构，并围绕课程所学进行改造。

## 概念

![Transformer-from-scratch block diagram](../assets/capstone.svg)

架构，带注释：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (因果掩码)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们交付的内容

- `GPTConfig` — 一个集中配置所有超参数的地方。  
- `MultiHeadAttention` — 因果（causal）、批处理友好，带可选的 Flash 风格路径（使用 PyTorch 的 `scaled_dot_product_attention`）。  
- `SwiGLUFFN` — 现代的前馈网络。  
- `Block` — pre-norm、带残差的 attention + FFN。  
- `GPT` — 嵌入层、堆叠 block、LM head、`generate()`。  
- 带 AdamW、余弦学习率调度（cosine LR）、梯度裁剪的训练循环。  
- 基于字节的字符分词器，适用于莎士比亚文本。

### 我们不交付的内容

- RoPE — 在第 04 课已概念性实现。这里为了简化使用了可学习的位置嵌入（learned positional embeddings）。练习会要求你替换为 RoPE。  
- 生成时的 KV cache — 每一步生成都会重新计算对前缀的 attention。更慢但更简单。练习会要求你添加 KV cache。  
- Flash Attention — PyTorch 2.0+ 会在输入匹配时自动分派；我们使用 `F.scaled_dot_product_attention`。  
- MoE — 每个 block 只用一个 FFN。你在第 11 课看过 MoE。

### 目标指标

在 Mac M2 笔记本上，配置为 4 层、4 头、d_model=128 的 GPT，在 `tinyshakespeare.txt` 上训练 2,000 步：

- 训练损失从约 4.2（随机初始化）收敛到约 1.5，耗时约 6 分钟。  
- 采样输出呈现莎士比亚风格：古体词、换行、像 "ROMEO:" 这样的专有名词出现。  
- 验证损失（保留最后 10% 文本）紧跟训练损失；在该规模/预算下没有过拟合迹象。

## 构建步骤

本课使用 PyTorch。安装 `torch`（CPU 版本即可）。参见 `code/main.py`。脚本会：

- 如果缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。  
- 基于字节的字符分词器。  
- 90/10 的训练/验证切分。  
- 在支持的硬件上使用 bf16 autocast 的训练循环。  
- 训练完成后采样输出。

### 步骤 1：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。极小的词表。适合 4 字节的 `vocab_size`。没有 BPE，也没有分词器复杂性。

### 步骤 2：模型

参见 `code/main.py`。Block 是第 05 课的教科书式实现 — pre-norm、RMSNorm、SwiGLU、因果 MHA。4/4/128 的参数量约为 ~800K。

### 步骤 3：训练循环

随机采样长度为 256 的 token 窗口构成 batch。前向。shift-by-one 的交叉熵。反向。AdamW 步进。日志记录。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 步骤 4：采样

给定提示（prompt），重复前向、对 logits 做 top-p 采样、追加 token、继续。最多生成 500 个 token。

### 步骤 5：读取输出

训练 2,000 步后示例输出：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚原文，但具有莎士比亚的风格。对于约 800K 参数、在笔记本上 6 分钟训练得到这样的结果是显著的成功。

## 使用方法

这个毕业项目是参考架构。把它用到真实场景的三个扩展方向：

1. Swap the tokenizer. 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词表规模从 65 跳到 ~50,000。模型容量需要相应扩大。  
2. Train on a bigger corpus. 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单块 A100 上训练 125M 参数的 GPT，处理 100 亿 token 大约需要 ~24 小时。  
3. Add RoPE + KV cache + Flash Attention. 下方练习会引导你逐步添加这些特性。

最终，你会得到一个 125M 参数级别的 GPT，能生成流畅的英语。虽然不是最前沿模型，但相同的代码路径（只是更大）就是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年训练研究检查点时所用的方法。

## 部署

参见 `outputs/skill-transformer-review.md`。该文档对一个从零实现的 transformer 在前 13 课中的正确性进行审查。

## 练习

1. 简单。运行 `code/main.py`。验证训练到最后一步时的验证损失是否低于 2.0。把 `max_steps` 从 2,000 改为 5,000 — 验证损失是否继续下降？  
2. 中等。用 RoPE 替换可学习的位置嵌入。在 `MultiHeadAttention` 的 Q 与 K 内部应用旋转（rotation）。训练并验证验证损失至少不变差。  
3. 中等。在采样循环中实现 KV cache。分别有/无 cache 生成 500 个 token，比较墙钟时间，应该能提升 5–20×（在笔记本上）。  
4. 困难。给模型添加第二个 head，预测下下一个 token（MTP — Multi-Token Prediction，来自 DeepSeek-V3）。联合训练。是否有助于性能？  
5. 困难。把每个 block 的单一 FFN 换成 4 个专家的 MoE。实现路由器（router）+ top-2 路由。以相同的活跃参数（active parameters）比较验证损失变化。

## 关键术语

| 术语 | 大家怎么说 | 它真正的含义 |
|------|-----------|-------------|
| nanoGPT | “Karpathy 的教程仓库” | 最小化的仅解码器 transformer 训练代码，约 300 行；规范参考实现。 |
| tinyshakespeare | “标准玩具语料” | 约 1.1 MB 文本；自 2015 年起每个字符级 LM 教程都在用。 |
| Tied embeddings | “共享输入/输出矩阵” | LM head 权重 = token embedding 矩阵的转置；节省参数并提升质量。 |
| bf16 autocast | “训练精度技巧” | 前向/反向在 bf16 下运行，优化器状态保留为 fp32；自 2021 年以来的标准做法。 |
| Gradient clipping | “阻止梯度爆炸” | 将全局梯度范数上限设为 1.0；防止训练失控。 |
| Cosine LR schedule | “自 2020 年代的默认” | 学习率线性升温（warmup），然后按余弦曲线衰减至峰值的 10%。 |
| MFU | “模型 FLOP 利用率” | 实际达到的 FLOPs / 理论峰值；2026 年，稠密模型 40%、MoE 30% 是优秀水平。 |
| Val loss | “保留集上的损失” | 在模型从未见过的数据上的交叉熵；用于检测过拟合。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 经典的带注释实现。