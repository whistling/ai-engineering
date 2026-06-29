# 命名实体识别

> 把名字抽取出来。听起来很简单，直到你遇到模糊边界、嵌套实体和领域术语。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (词嵌入)
**Time:** ~75 分钟

## 问题描述

"Apple sued Google over its iPhone search deal in the US." 五个实体：Apple (ORG)、Google (ORG)、iPhone (PRODUCT)、search deal（可能）、US (GPE)。一个好的 NER 系统能全部正确抽取并标注类型。差的系统可能漏掉 iPhone，把公司 Apple 识别为水果，或者把 "US" 标注为 PERSON。

NER 是所有结构化抽取流水线背后的主力。简历解析、合规日志扫描、病历匿名化、搜索查询理解、聊天机器人应答的落地、法律合同抽取。你常常看不到它，但又始终依赖它。

本课带你从经典路径（基于规则、HMM、CRF）走到现代方法（BiLSTM-CRF，然后是 transformer）。每一步都在解决前一阶段的具体局限性。这种递进模式就是本课的要点。

## 概念

**BIO 标注**（或 BILOU）把实体抽取转成序列标注问题。给每个 token 标注 `B-TYPE`（实体开始）、`I-TYPE`（实体内部）或 `O`（不属于任何实体）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多 token 实体是串联的：`New B-GPE`、`York I-GPE`、`City I-GPE`。能理解 BIO 的模型就能抽取任意跨度。

架构演进：

- 基于规则。正则 + 地名/实体词表查询。对已知实体精度高，但对新实体覆盖为零。
- HMM。隐马尔可夫模型。基于标签的发射概率和标签间的转移概率。用 Viterbi 解码。基于标注数据训练。
- CRF。条件随机场。比 HMM 判别式，可以混合任意特征（词形、大小写、邻词等）。直到 2026 年，CRF 仍是低资源部署的经典生产主力。
- BiLSTM-CRF。用神经网络学习特征替代手工特征。LSTM 双向读句子，CRF 层在输出端保证标注一致性。
- 基于 Transformer。微调 BERT 并接一个 token-classification 头。精度最高，但计算量也最大。

```figure
ner-bio-tagging
```

## 实现

### 第 1 步：BIO 标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 第 2 步：手工特征

对于经典（非神经）NER，特征是关键。有用的特征示例：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式对专有名词信号很强。

### 第 3 步：简单的基于规则 + 词表基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产级的词表会有从 Wikipedia、DBpedia 抓取的数百万条目。覆盖面很好，但消歧（例如 Apple 公司 vs 苹果水果）非常差。这就是统计模型胜出的原因。

### 第 4 步：CRF（草图，不是完整实现）

从零实现完整 CRF 在 50 行内并不能说明概率论基础。建议使用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 和 L2 正则化。`all_possible_transitions=True` 让模型学习到非法序列（例如 `I-ORG` 在 `O` 之后）不太可能出现，这就是 CRF 在不显式写约束的情况下强制 BIO 一致性的方式。

### 第 5 步：BiLSTM-CRF 的增益

特征由模型学习而不是人工设计。输入：token 嵌入（GloVe 或 fastText）。LSTM 双向读取，拼接的隐状态通过 CRF 输出层。CRF 仍然保证标签序列一致性；LSTM 用学习到的特征替代手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF 层可以使用 `torchcrf.CRF`（pip install pytorch-crf）。相比手工特征的 CRF，神经方法的提升是可测量的，但如果没有数万级别的标注句子，提升往往不如预期那么大。

## 使用示例

spaCy 提供了开箱即用的生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标为 `ORG` 而不是 `PRODUCT` —— spaCy 的小模型在产品实体覆盖上比较弱。`en_core_web_lg`（大模型）表现更好，`en_core_web_trf`（transformer 模型）更佳。

Hugging Face 基于 BERT 的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 会把连续的 B-X、I-X token 合并成一个 span。没有该策略时，你会得到 token 级别的标签，需要自己合并。

### 基于 LLM 的 NER（2026 年选项）

零样本和少样本的 LLM NER 在许多领域已与微调模型持平，当标注数据稀缺时效果会显著更好。

- 零样本提示（Zero-shot prompting）。给模型一组实体类型和示例 schema，要求输出 JSON。开箱即可用；在新领域上精度中等。
- ZeroTuneBio 风格的提示。把任务分解为候选抽取 → 含义解释 → 判断 → 复核。多阶段提示（而非一次性）能显著提升生物医学 NER 的精度。同样模式在法律、金融、科研领域也有效。
- 结合 RAG 的动态提示。为每次推理检索最相似的带标注示例，从小的带注释种子集中构建少样本提示。在 2026 年的基准中，这能使 GPT-4 在生物医学 NER 上的 F1 比静态提示提升约 11–12%。
- 按实体类型分解。对于长文档，一次性抽取所有类型随着长度增长会丢失召回率。对每个实体类型单独跑一次抽取。推理成本更高，但精度大幅提升。这是临床笔记和法律合同中的常用模式。

2026 年的生产建议：在收集训练数据前，先跑 LLM 零样本基线。很多情况下 F1 已足够好，根本不需要微调。

### 经典 NER 仍占优势的场景

即便有 LLM，可经典方法仍有优势，当：

- 延迟预算低于 50ms。
- 你有数千条标注样本并需要 98%+ 的 F1。
- 领域有稳定本体，预训练的 CRF 或 BiLSTM 能很好迁移。
- 法规要求必须在本地部署、不可使用生成式模型。

### 损坏（失败）的场景

- 域转移（Domain shift）。在 CoNLL 上训练的 NER 在法律合同上可能不如一个词表。要在目标域上微调。
- 嵌套实体。像 "Bank of America Tower" 同时是 ORG 和 FACILITY。标准 BIO 无法表示重叠跨度。需要嵌套 NER（多轮或基于跨度的模型）。
- 长实体。像 "United States Federal Deposit Insurance Corporation." 这样的长串有时被 token 级模型断开。使用 `aggregation_strategy` 或后处理。
- 稀疏类型。医学 NER 中像 DRUG_BRAND、ADVERSE_EVENT、DOSE 这类标签很稀疏。通用模型无从下手。Scispacy 和 BioBERT 是起点。

## 上线部署

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. 简单。实现 `bio_to_spans`（`spans_to_bio` 的逆函数），并在 10 个句子上验证往返一致性。
2. 中等。用上面给出的 sklearn-crfsuite CRF 在 CoNLL-2003 英语 NER 数据集上训练。用 `seqeval` 报告每类实体的 F1。典型结果：约 ~84 F1。
3. 困难。微调 `distilbert-base-cased` 在一个领域特定的 NER 数据集（医学、法律或金融）上。与 spaCy 小模型比较。记录数据泄漏检查并写出让你惊讶的发现。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| NER | Extract names | 用类型（PERSON、ORG、GPE、DATE 等）标注 token 跨度。 |
| BIO | Tagging scheme | `B-X` 表示开始，`I-X` 表示继续，`O` 表示外部。 |
| BILOU | Better BIO | 增加 `L-X`（最后）和 `U-X`（单元）以获得更清晰的边界。 |
| CRF | Structured classifier | 对标签间的转移建模，而不仅仅是发射。强制有效序列。 |
| Nested NER | Overlapping entities | 一个跨度是另一个跨度的子串并表示不同实体。BIO 无法表达这一点。 |
| Entity-level F1 | Proper NER metric | 预测的实体跨度必须与真实跨度完全匹配。基于 token 的 F1 会夸大准确率。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 论文，经典之作。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入了成为标准的 token-classification 模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 上每个属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的评估库。务必使用它。