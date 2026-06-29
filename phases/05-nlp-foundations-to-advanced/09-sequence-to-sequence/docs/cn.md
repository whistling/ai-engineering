# Sequence-to-Sequence Models

> 两个假装是翻译器的 RNN。它们遇到的瓶颈正是注意力存在的原因。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 5 · 08（用于文本的 CNNs + RNNs），Phase 3 · 11（PyTorch 介绍）  
**Time:** ~75 分钟

## 问题

分类把可变长度序列映射到单个标签。翻译把可变长度序列映射到另一个可变长度序列。输入和输出可能属于不同的词表、不同的语言，并且长度不一定相等。

seq2seq 架构（Sutskever, Vinyals, Le, 2014）用一个刻意简单的配方解决了这个问题。两个 RNN。一个阅读源句子并生成一个固定大小的上下文向量。另一个从该向量初始化并逐步生成目标句子的标记。和在第08课写的代码相似，只是拼接方式不同。

研究它有两个原因。首先，上下文向量瓶颈是 NLP 教学中最有价值的失败示例。它直接推动了注意力和 Transformer 的发展。其次，训练配方（teacher forcing、scheduled sampling、推理时的 beam search）仍然适用于包括大型语言模型在内的所有现代生成系统。

## 概念

**Encoder（编码器）。** 一个读取源句子的 RNN。它的最终隐藏状态就是**上下文向量**——对整个输入的固定大小摘要。理论上除了源信息之外不丢失任何东西。

**Decoder（解码器）。** 另一个从上下文向量初始化的 RNN。在每一步它把上一步生成的标记作为输入，并对目标词表输出一个分布。对下一个标记做采样或 argmax。将其反馈回去。重复，直到产生 `<EOS>` 标记或达到最大长度。

**训练：** 在每个解码器步骤上计算交叉熵损失，并对序列求和。通过时间反向传播贯穿两个网络。

**Teacher forcing（教师强制）。** 在训练期间，解码器在时间步 `t` 的输入是位置 `t-1` 的真实标记，而不是解码器自己上一步的预测。这样可稳定训练；没有它，早期错误会连锁放大，模型几乎无法学习。推理时你必须使用模型自己的预测，因此训练/推理之间总存在分布差距。这个差距叫做 **exposure bias（暴露偏差）**。

**瓶颈。** 编码器从源句子学到的一切必须被压缩到那个单一的上下文向量中。长句子细节会丢失。罕见词会被模糊化。重排序（例如 chat noir 与 black cat）必须被记住而不是计算。

注意力（第10课）通过让解码器查看每一个编码器隐藏状态而不是仅最后一个来修复这个问题。这就是全部要点。

```figure
lstm-gates
```

## 构建它

### 第 1 步：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状为 `[batch, seq_len, hidden_dim]` — 每个输入位置对应一个隐藏状态。`hidden` 的形状为 `[1, batch, hidden_dim]` — 最后一步的隐藏状态。第08课里说过“对 outputs 做池化用于分类”。在这里我们保留最后的隐藏状态作为上下文向量，忽略每步输出。

### 第 2 步：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器按步调用。输入：一批单个标记和当前隐藏状态。输出：下一个标记的词表 logits 和更新后的隐藏状态。

### 第 3 步：带教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

有两个值得命名的旋钮。`ignore_index=0` 在 padding 标记上跳过损失。`teacher_forcing_ratio` 是在每一步使用真实标记而非模型预测的概率。训练开始时设为 1.0（完全教师强制），并在训练中逐渐退火到大约 0.5，以缩小暴露偏差。

### 第 4 步：推理循环（贪婪）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪婪解码在每一步选择概率最高的标记。它可能偏离：一旦你确认了一个标记，就无法撤回。**Beam search（束搜索）** 在每一步保留 top-`k` 个部分序列，最后选择得分最高的完整序列。束宽 3-5 是常用的选择。

### 第 5 步：展示瓶颈

在一个玩具复制任务上训练模型：源 `[a, b, c, d, e]`，目标 `[a, b, c, d, e]`。增加序列长度，观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损地记住 40 个标记的输入。编码器在每一步都有信息，但解码器只看最后一个状态。注意力直接修复了这个问题。

## 使用它

PyTorch 提供了 `nn.Transformer` 和 基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库提供了完整的编码器-解码器模型（BART、T5、mBART、NLLB），在数十亿标记上训练。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代的编码器-解码器放弃了 RNN，转而使用 Transformer。高层形态（编码器、解码器、按标记生成）与 2014 年的 seq2seq 论文完全相同。每个模块内部的机制不同。

### 什么时候仍然考虑基于 RNN 的 seq2seq

几乎不会用于新项目。特定例外：

- 流式翻译，需要以有界内存逐标记消费输入。
- 设备端文本生成，当 Transformer 的内存开销不可接受时。
- 教学用途。理解编码器-解码器瓶颈是理解 Transformer 胜出的最快路径。

### 暴露偏差及其缓解方法

- **Scheduled sampling（计划采样）。** 在训练中退火 teacher forcing 比率，使模型学会从自己的错误中恢复。
- **Minimum risk training（最小风险训练）。** 用句子级别的 BLEU 分数替代逐标记的交叉熵进行训练。更接近实际目标。
- **Reinforcement learning fine-tuning（强化学习微调）。** 用一个指标对序列生成器进行奖励。现代 LLM 的 RLHF 即采用此类思路。

这三种方法仍然适用于基于 Transformer 的生成。

## 发布它

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: 为给定任务设计一个 sequence-to-sequence 流水线。
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. 简单。实现玩具复制任务。训练一个 GRU seq2seq，使输入和输出相等。测量长度为 5、10、20 时的准确率。复现瓶颈现象。
2. 中等。添加束搜索解码，束宽为 3。在一个小的平行语料上对比贪婪解码与束搜索的 BLEU。记录束搜索在哪些地方胜出（通常是末尾标记），以及在哪些地方没有差别。
3. 困难。对 `facebook/bart-base` 在一个 10k 对的释义数据集上进行微调。对比微调模型的 beam-4 输出与基础模型在保留集上的表现。报告 BLEU 并挑选 10 个定性示例。

## 术语表

| 术语 | 人们如何称呼 | 实际含义 |
|------|---------------|---------|
| Encoder | 输入 RNN / 编码器 | 读取源序列。产生每步隐藏状态和最终的上下文向量。 |
| Decoder | 输出 RNN / 解码器 | 从上下文向量初始化。逐个生成目标标记。 |
| Context vector | 摘要 / 上下文向量 | 编码器的最终隐藏状态。固定大小。是注意力要解决的瓶颈。 |
| Teacher forcing | 使用真实标记 | 在训练时输入真实的上一步标记。稳定学习过程。 |
| Exposure bias | 训练/测试差 | 在真实标记上训练的模型没有练习从自身错误中恢复。 |
| Beam search | 更好的解码 | 在每一步保留 top-k 的部分序列，而不是贪婪地一次确定一个标记。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始的 seq2seq 论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入了 GRU 和编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力论文。读完本课后立刻阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可复现的 seq2seq + 注意力 教程与代码。