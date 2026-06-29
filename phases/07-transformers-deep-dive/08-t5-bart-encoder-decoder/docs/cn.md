# T5, BART — 编码器-解码器模型

> 编码器负责理解。解码器负责生成。把它们组合在一起，就得到一个为输入 → 输出任务设计的模型：翻译、摘要、改写、转录。

**Type:** 学习
**Languages:** Python
**Prerequisites:** Phase 7 · 05（完整 Transformer）、Phase 7 · 06（BERT）、Phase 7 · 07（GPT）
**Time:** ~45 分钟

## 问题

仅解码的 GPT 和仅编码的 BERT 各自为不同目标简化了 2017 年的 Transformer 架构。但很多任务天然属于输入-输出范式：

- 翻译：English → French。
- 摘要：5,000 令牌的文章 → 200 令牌的摘要。
- 语音识别：音频令牌 → 文本令牌。
- 结构化抽取：散文 → JSON。

对于这些任务，编码器-解码器是最匹配的。编码器对源序列产出一个密集表示。解码器生成输出，在每一步对该表示进行交叉注意力。训练时在输出端做一个位移一位（shift-by-one）。损失与 GPT 相同，只是以编码器输出为条件。

两篇论文定义了现代套路：

1. **T5**（Raffel 等，2019）。“Text-to-Text Transfer Transformer”。把每个 NLP 任务重新表述为文本输入-文本输出。单一架构、单一词表、单一损失。预训练采用遮蔽 span 预测（在输入中破坏 span，在输出中解码它们）。
2. **BART**（Lewis 等，2019）。“Bidirectional and Auto-Regressive Transformer”。去噪自编码器：以多种方式破坏输入（打乱、掩码、删除、旋转），让解码器重构原文。

到 2026 年，编码器-解码器格式仍在输入结构重要的场景中广泛使用：

- Whisper（语音 → 文本）。
- Google 的翻译栈。
- 一些具有独特“上下文-编辑”结构的代码补全/修复模型。
- 用于结构化推理任务的 Flan-T5 及其变体。

虽然仅解码模型占据了聚光灯，编码器-解码器并未消失。

## 概念

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### 前向流程

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键点：编码器对每个输入只运行一次。解码器以自回归方式运行，但在每一步都对相同的编码器输出做交叉注意力。对编码器输出进行缓存对于长输入是一个免费的加速手段。

### T5 预训练 — span corruption（跨度损坏）

随机选取输入中的 spans（平均长度 3 个令牌，总计约 15%）。用唯一的哨兵标记替换每个 span：`<extra_id_0>`、`<extra_id_1>` 等。解码器只输出被破坏的 spans，并以哨兵作为前缀：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列的信号更便宜。在 T5 论文的消融实验中与 MLM（BERT）和 prefix-LM（UniLM）具有竞争力。

### BART 预训练 — 多噪声去噪

BART 采用五种噪声函数：

1. 令牌掩码。
2. 令牌删除。
3. 文本填空（掩盖一个 span，解码器插入正确长度）。
4. 句子置换。
5. 文档旋转。

将文本填空与句子置换结合产生了最佳下游效果。解码器始终重构原始文本。BART 的输出是完整序列，而不是仅仅被破坏的 spans——因此预训练计算量高于 T5。

### 推理

与 GPT 相同的自回归生成。可使用贪心、束搜索、top-p 采样等。对于翻译和摘要，束搜索（宽度 4–5）是标准做法，因为输出分布通常比对话更窄。

### 在 2026 年何时选择哪种变体

| Task | Encoder-decoder? | Why |
|------|------------------|-----|
| Translation | 是，通常是 | 源序列清晰；输出分布确定；束搜索有效 |
| Speech-to-text | 是（Whisper） | 输入模态不同于输出；编码器负责构建音频特征 |
| Chat / reasoning | 否，仅解码 | 没有持续不变的“输入”——会话本身就是序列 |
| Code completion | 通常否 | 具有长上下文时仅解码更有优势；像 Qwen 2.5 Coder 这样的代码模型是仅解码的 |
| Summarization | 两者都可 | BART、PEGASUS 战胜了早期的仅解码基线；现代仅解码 LLM 与之相当 |
| Structured extraction | 两者都可 | T5 很干净，因为“文本 → 文本”可以承载任何输出格式 |

自 ~2022 年以来的趋势：仅解码模型接管了许多原本属于编码器-解码器的任务，原因是 (a) 指令微调后的仅解码 LLM 通过提示词能泛化为任意任务，(b) 单一架构比双栈更容易扩展，(c) RLHF 通常基于解码器。编码器-解码器在输入模态不同（语音、图像）或需要束搜索质量保障的场景中仍占有一席之地。

## 实现

见 `code/main.py`。我们实现了 T5 风格的 span corruption，用于一个玩具语料 —— 这是本课中最实用的一部分，因为自 T5 以来它出现在每个编码器-解码器预训练配方中。

### 步骤 1：跨度损坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式遵循 T5 约定：`<sent0> span0 <sent1> span1 ...`。被破坏的输入在 span 位置将未改变的令牌与哨兵标记交错插入。

（注：代码块内的实现保持原样；如有注释，请将注释翻译为中文以便开发者理解。）

### 步骤 2：验证可逆性（round-trip）

给定被破坏的输入和目标，重构原始句子。如果你的损坏过程是可逆的，前向过程就是定义良好的。这是一个健全性检查——真实训练不会做这个，但测试开销低并能捕捉跨越计数的 off-by-one 错误。

### 步骤 3：BART 噪声生成

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 使用示例

HuggingFace 参考代码：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的技巧：任务名称放入输入文本。相同的模型处理几十种任务，因为每个任务都是文本输入-文本输出。到 2026 年，这一模式已被指令微调的仅解码模型所泛化，但 T5 是最早把它系统化的。

## 部署（Ship It）

见 `outputs/skill-seq2seq-picker.md`。该技能根据输入-输出结构、延迟和质量目标，为新任务在编码器-解码器与仅解码之间做出选择。

## 练习

1. **简单。** 运行 `code/main.py`，对一个 30 令牌的句子应用 span corruption，验证将非哨兵的源令牌与解码出的目标 spans 串联能否重现原句。
2. **中等。** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` 令牌替换随机 spans，解码器必须推断出正确的 span 长度和内容。展示一个示例。
3. **困难。** 在一个小型 English → pig-Latin 语料（200 对）上微调 `flan-t5-small`，在留出的 50 对测试集上测量 BLEU。与在相同数据和相同计算下微调 `Llama-3.2-1B` 的结果比较。

## 关键词

| 术语 | 大家怎么说 | 实际意义 |
|------|-----------|---------|
| 编码器-解码器 | “Seq2seq transformer” | 两个堆栈：用于输入的双向编码器，以及用于输出带交叉注意力的因果解码器。 |
| 交叉注意力 | “Where source talks to target” | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一通路。 |
| 跨度损坏 | “T5's pretraining trick” | 用哨兵标记替换随机 spans；解码器输出这些 spans。 |
| 去噪目标 | “BART's game” | 对输入应用噪声函数，训练解码器重构干净序列。 |
| 哨兵标记 | “The `<extra_id_N>` placeholder” | 在源中标注被破坏的 spans，并在目标中重新标注的特殊令牌。 |
| Flan | “Instruction-tuned T5” | 在 >1,800 个任务上对 T5 进行微调；使编码器-解码器在指令遵从上具有竞争力。 |
| 束搜索 | “Decoding strategy” | 在每一步保留 top-k 的部分序列；翻译/摘要的标准解码方法。 |
| 教师强制 | “Training-time input” | 训练时将真实的前一个输出令牌提供给解码器，而不是采样得到的令牌。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年的典型编码器-解码器系统。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。