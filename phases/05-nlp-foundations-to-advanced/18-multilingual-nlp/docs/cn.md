# 多语言 NLP

> 一个模型，覆盖 100+ 语言，对大多数语言没有任何训练数据。跨语言迁移是 2020 年代的实用奇迹。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (机器翻译)  
**Time:** ~45 分钟

## 问题

英语有数十亿条带标注的样本。乌尔都语有数千条。迈蒂利语几乎没有。任何面向全球受众的实用 NLP 系统都必须在那些没有任务特定训练数据的长尾语言上工作。

多语种模型通过在许多语言上同时训练一个模型来解决这个问题。共享表示使模型能将从高资源语言学到的能力迁移到低资源语言。在英文情感分析上微调模型，然后直接在乌尔都语上推理就能产生令人惊讶的好结果。这就是零样本跨语言迁移（zero-shot cross-lingual transfer），它已经重塑了 NLP 的全球部署方式。

本课介绍其权衡、典型模型，以及一个让刚接触多语种工作的团队犯错的关键决策：选择用于迁移的源语言。

## 概念

![通过共享多语种嵌入空间进行跨语言迁移](../assets/multilingual.svg)

**共享词表。** 多语种模型使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece 分词器。词表是共享的：同一个子词单元在相关语言中表示相同语素。英文和意大利语中的 `anti-` 会得到相同的 token。

**共享表示。** 在多语言上以掩码语言建模预训练的 Transformer 会学到：不同语言中语义相似的句子会产生相似的隐藏态。mBERT、XLM-R 和 NLLB 都表现出这种特性。英语的 "cat"、法语的 "chat" 和西班牙语的 "gato" 的嵌入会聚在一起，整句嵌入也同理。

**零样本迁移（Zero-shot transfer）。** 在一种语言（通常是英语）上微调模型。在推理时，将模型运行在它支持的任何其他语言上。无需目标语言标注。对于语系上相关的语言，效果很强；对于远缘语言，效果较弱。

**少样本微调（Few-shot fine-tuning）。** 在目标语言中加入 100-500 条标注样本。分类任务的准确率通常能提升到英文基线的 95-98%。这是多语种 NLP 中性价比最高的手段。

## 模型

| Model | Year | Coverage | Notes |
|-------|------|----------|-------|
| mBERT | 2018 | 104 languages | 在 Wikipedia 上训练。第一个实用的多语种语言模型。对低资源语言表现较弱。 |
| XLM-R | 2019 | 100 languages | 在 CommonCrawl 上训练（远大于 Wikipedia）。奠定了跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 languages | XLM-R，词表扩展到 1M token（ vs 250k）。对低资源语言更友好。 |
| mT5 | 2020 | 101 languages | 将 T5 架构用于多语种生成任务。 |
| NLLB-200 | 2022 | 200 languages | Meta 的翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 languages + 13 programming | 开源的 176B 多语种 LLM。 |
| Aya-23 | 2024 | 23 languages | Cohere 的多语种 LLM。对阿拉伯语、印地语、斯瓦希里语表现强劲。 |

按用例选择。分类任务用 XLM-R-base 作为稳妥默认通常良好。生成任务根据是翻译还是开放生成分别选择 mT5 或 NLLB。类 LLM 的工作则配合 Aya-23 或 Claude 并使用明确的多语种提示词（prompting）。

## 源语言决策（2026 年研究）

大多数团队默认选择英语作为微调源语言。最新研究（2026）表明这经常不是最佳选择。

语言相似性比原始语料规模更能预测迁移质量。对于斯拉夫语系的目标语言，德语或俄语常常胜过英语。对于印地语系的目标语言，印地语往往胜过英语。基于世界语言结构地图（World Atlas of Language Structures）特征的 **qWALS** 相似性度量（2026）对此量化。**LANGRANK**（Lin 等，ACL 2019）是一个较早的方法，它结合语言学相似性、语料规模和基因学关系对候选源语言进行排序。

实用规则：如果目标语言有一个在类型学上接近的高资源亲缘语言，先尝试在该语言上微调，再与在英语上微调的结果比较。

## 构建

### 步骤 1：零样本跨语种分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，相同的 API。XLM-R 在 NLI 数据上训练，使用蕴含（entailment）技巧能够良好地迁移到分类任务。

### 步骤 2：多语种嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译句子在嵌入空间中会靠得更近。不同的英文句子会更远。这正是跨语种检索、聚类和相似性测量得以工作的原因。

### 步骤 3：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 条目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全的默认值。学习率太高会破坏多语种对齐，模型会退化为仅适应英语的模型。

## 实际有效的评估

- **按语言划分的在保留集上的准确率。** 不要只看汇总指标。汇总会掩盖长尾问题。  
- **与单语基线比较。** 对于有足够数据的语言，从头训练的单语模型有时会优于多语种模型。测试一下。  
- **实体级测试。** 目标语言中的命名实体。多语种模型在远离拉丁字母的文字脚本上的分词常常较弱。  
- **跨语种一致性。** 两种语言中意思相同的文本应产生相同的预测，度量它们之间的差距。

## 使用

2026 年技术栈：

| Task | Recommended |
|-----|-------------|
| Classification, 100 languages | XLM-R-base (~270M) 微调 |
| Zero-shot text classification | `joeddav/xlm-roberta-large-xnli` |
| Multilingual sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Translation, 200 languages | `facebook/nllb-200-distilled-600M`（参见 lesson 11） |
| Generative multilingual | Claude、GPT-4、Aya-23、mT5-XXL |
| Low-resource language NLP | XLM-V 或在相关高资源语言上进行领域特定微调 |

如果性能重要，请始终预算目标语言的微调。零样本只是一个起点，不是最终答案。

### 分词税（低资源语言会遇到的问题）

多语种模型在所有语言之间共享一个分词器。该词表是在以英语、法语、西班牙语、中文、德语为主的语料上训练的。对于任何不在主导集合中的语言，三个成本会悄然叠加：

- **分词膨胀成本（Fertility tax）。** 低资源语言的文本分词后每词所占的 token 数远高于英语。一个印地语句子可能需要相当于英文 3-5 倍的 token。这 3-5 倍会吞噬你的上下文窗口、训练效率和延迟。  
- **变体恢复成本（Variant recovery tax）。** 每一个拼写错误、变音符号差异、Unicode 归一化不匹配或大小写变体都会成为在嵌入空间中冷启动的不同序列。模型无法学会母语者视为显然的正字法对应关系。  
- **容量溢出成本（Capacity spillover tax）。** 成本 1 与 2 会消耗上下文位置、层深和嵌入维度。留给实际推理的容量系统性地比高资源语言从同一模型中得到的要少。

实际症状：模型在印地语上训练看起来正常，损失曲线良好，评估困惑度也合理，但生产输出会细微地出错。形态学在句中崩塌，罕见词形无法恢复。你不能通过简单扩大数据规模来解决一个损坏的分词器（You cannot data-scale your way out of a broken tokenizer）。

缓解措施：为目标语言选择覆盖率好的分词器（XLM-V 的 1M token 词表是直接的修复）；在训练前在保留的目标文本上验证分词膨胀率；对真正的长尾脚本使用字节级回退（SentencePiece 的 `byte_fallback=True`，或 GPT-2 风格的字节级 BPE），以确保没有 OOV。

## 部署

Save as `outputs/skill-multilingual-picker.md`:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. 简单：对英、法、印地语和阿拉伯语每种语言运行 10 条句子的零样本分类管道。报告每种语言的准确率。你应该看到法语强、印地语尚可、阿拉伯语差异较大。  
2. 中等：使用 `paraphrase-multilingual-MiniLM-L12-v2` 构建一个混合语言小语料库的跨语种检索器。用英文查询，检索任意语言的文档。测量 recall@5。  
3. 困难：比较以英语为源和以印地语为源对印地语分类任务的微调效果。对两种方案都用 500 条目标语言样本进行少样本微调。报告哪个源语言产生了更好的印地语准确率以及差距。这个练习是 LANGRANK 论点的缩影。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Multilingual model | One model, many languages | 跨语言共享词表和参数的模型。 |
| Cross-lingual transfer | Train on one language, run on another | 在源语言上微调，在目标语言上评估而无需目标语言标签。 |
| Zero-shot | No target-language labels | 无目标语言标签的迁移（不在目标语言上微调）。 |
| Few-shot | Small target labels | 少量目标语言标注；通常为 100-500 条用于微调。 |
| mBERT | First multilingual LM | 在 104 种语言的 Wikipedia 上预训练的 BERT。 |
| XLM-R | Standard cross-lingual baseline | 在 100 种语言的 CommonCrawl 上预训练的 RoBERTa。 |
| NLLB | Meta's 200-language MT | No Language Left Behind；包含许多低资源语言的翻译模型。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文。  
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 启动跨语言迁移研究方向的分析论文。  
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。  
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的多语种 LLM。  
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — 关于 qWALS / LANGRANK 的源语言选择论文（2026）。