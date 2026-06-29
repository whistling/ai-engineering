# 自然语言推断 — 文本蕴含

> “t 蕴含 h” 意味着阅读 t 的人会得出 h 为真的结论。NLI 的任务是预测 蕴含 / 矛盾 / 中立。表面上枯燥，但在生产环境中承载着重要功能。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 5 · 05 (情感分析), Phase 5 · 13 (问答)  
**Time:** ~60 分钟

## 问题

你构建了一个摘要器。它生成了一个摘要。你如何确认摘要没有包含幻觉（hallucination）？

你构建了一个聊天机器人。它回答了“是”。你如何确认该答案得到了检索到的片段支持？

你需要对 10,000 篇新闻文章进行主题分类，但没有训练标签。你能复用一个模型吗？

这三类问题都可以归约为自然语言推断。NLI 问的是：给定前提 `t` 和假设 `h`，`h` 是否被 `t` 蕴含、被矛盾，还是中立（无关）？

- 幻觉检测：`t` = 源文档，`h` = 摘要断言。如果不是蕴含 = 幻觉。
- 有依据的问答（Grounded QA）：`t` = 检索到的片段，`h` = 生成的答案。如果不是蕴含 = 捏造。
- 零样本分类：`t` = 文档，`h` = 口语化标签（“This is about sports”）。蕴含 = 预测标签。

一个任务，三种生产用途。这就是为什么每个 RAG 评估框架在底层都会配备一个 NLI 模型。

## 概念

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**三种标签。**

- **蕴含（Entailment）。** `t` → `h`。例如 “The cat is on the mat” 蕴含 “There is a cat.”
- **矛盾（Contradiction）。** `t` → ¬`h`。例如 “The cat is on the mat” 与 “There is no cat.” 矛盾。
- **中立（Neutral）。** 无法推断任一方向。例如 “The cat is on the mat” 对 “The cat is hungry” 保持中立。

**非逻辑上的严格蕴含。** NLI 是“自然语言”推断——衡量典型人类读者会如何推断，而不是严密的形式逻辑。比如在 NLI 中，“John walked his dog” 蕴含 “John has a dog”，而在严格的一阶逻辑里，只有当你把“有狗”作为公理化的前提时才会成立。

**数据集。**

- **SNLI**（2015）。57 万对人工标注样本，前提来自图像标题。领域较窄。
- **MultiNLI**（2017）。43.3 万对，覆盖 10 个体裁。到 2026 年仍是标准训练语料。
- **ANLI**（2019）。对抗性 NLI。人类专门编写旨在打破现有模型的样例。更难。
- **DocNLI、ConTRoL**（2020–21）。文档级前提。测试多跳与长距离推断能力。

**架构。** 使用 transformer 编码器（如 BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 表示用于 3 类 Softmax。用 MNLI 训练，在相应验证集上评估，能在同分布对上达到 90%+ 的准确率。

**基于 NLI 的零样本。** 给定文档和候选标签，把每个标签变为一个假设（例如 “This text is about sports”）。计算每个标签的蕴含概率，选择最大值。这是 Hugging Face `zero-shot-classification` pipeline 的内部机制。

## 构建

### 第 1 步：运行预训练的 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # 返回所有标签；替代已弃用的 return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

在生产环境中，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是开源默认选择。DeBERTa-v3 在排行榜上表现优异。

### 第 2 步：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认的模板是 "This example is about {label}."。可通过 `hypothesis_template` 自定义。无需训练数据，也无需微调，开箱即可工作。

### 第 3 步：RAG 的忠实性检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实性检查的核心。把生成的答案拆分成原子断言（atomic claims）。对每个断言使用 NLI 与检索到的上下文比对。报告被蕴含的断言占比。

### 第 4 步：手工实现的 NLI 分类器（概念性）

参见 `code/main.py` 的标准库玩具实现：前提和假设通过词汇重叠与否定检测进行比较。无法与 transformer 模型竞争——但它展示了任务的形状：两个文本输入，输出三分类，损失 = 对 `{entail, contradict, neutral}` 的交叉熵。

## 陷阱

- **仅靠假设的捷径（Hypothesis-only shortcuts）。** 模型仅凭假设就能在 SNLI 上达到约 60% 的准确率，因为 “not”、“nobody”、“never” 等词与 contradiction 强相关。假设-only 基线是检测标签泄露的强有力手段。
- **词汇重叠启发式。** 子序列启发式（“every subsequence is entailed”）能过 SNLI，但会在 HANS/ANLI 上失败。使用对抗性基准。
- **文档长度导致性能下降。** 单句 NLI 模型在文档级前提上会下降 20+ F1。对长上下文请使用在 DocNLI 上训练的模型。
- **零样本模板敏感性。** “This example is about {label}” 与 “{label}” 或 “The topic is {label}” 的差异可能导致准确率波动超过 10 个点。需要调优模板。
- **领域不匹配。** MNLI 在通用英语上训练。法律、医学、科学文本需要领域特定的 NLI 模型（例如 SciNLI、MedNLI）。

## 使用场景

2026 年技术栈：

| Use case | Model |
|---------|-------|
| General-purpose NLI | `microsoft/deberta-v3-large-mnli` |
| Fast / edge | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot classification (lightweight) | `facebook/bart-large-mnli` |
| Document-level NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Multilingual | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Hallucination detection in RAG | NLI layer inside RAGAS / DeepEval |

2026 年的元模式：NLI 是文本理解的万能胶。当你需要判定 “A 是否支持 B？” 或 “A 是否与 B 矛盾？”——在发起另一次 LLM 调用之前，先试试 NLI。

## 上线指南（Ship It）

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: 为分类 / 忠实性 / 零样本任务挑选 NLI 模型、标签模板与评估设置。
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

给定一个用例（忠实性检查、零样本分类、文档级推断），输出：

1. 模型。命名的 NLI checkpoint。说明为什么根据领域、长度、语言选择该模型。
2. 模板（如果是零样本）。口语化模式。示例。
3. 阈值。用于决策规则的蕴含阈值。基于校准的理由。
4. 评估。在保留标注集上的准确率、仅假设基线、对抗子集表现。

拒绝在没有 100 个样例的标注性检查（sanity check）的情况下上线零样本分类。拒绝在文档级前提上使用句子级 NLI 模型。标记任何声称 NLI 可以完全解决幻觉的问题 —— 它能减少幻觉，但不能彻底消除。
```

## 练习

1. 简单（Easy）。在 20 个手工构造的（前提，假设，标签）三元组上运行 `facebook/bart-large-mnli`，覆盖三种类别。测量准确率。加入对抗性的“子序列启发式”陷阱（如 “I did not eat the cake” vs “I ate the cake”）并观察是否被突破。
2. 中等（Medium）。在 100 条 AG News 标题上比较三种零样本模板 `"This text is about {label}"`、`"The topic is {label}"` 和 `"{label}"` 的效果。报告准确率波动。
3. 困难（Hard）。构建一个 RAG 忠实性检查器：原子断言分解 + 对每个断言的 NLI 检查。在 50 个带有黄金上下文的 RAG 生成答案上评估。测量相对于人工标注的假阳性与假阴性率。

## 术语

| Term | 大众说法 | 实际含义 |
|------|---------|---------|
| NLI | Natural Language Inference | 前提-假设关系的三分类（蕴含 / 矛盾 / 中立）。 |
| RTE | Recognizing Textual Entailment | NLI 的旧称；任务相同。 |
| Entailment | “t implies h” | 在典型读者看来，给定 t 可以推断出 h 为真。 |
| Contradiction | “t rules out h” | 在典型读者看来，给定 t 可以判定 h 为假。 |
| Neutral | “undecided” | 从 t 无法向任一方向推断 h。 |
| Zero-shot classification | NLI 作为分类器 | 将标签口语化为假设，选取最大蕴含概率。 |
| Faithfulness | 答案是否被支持？ | 在（检索到的上下文，生成的答案）上运行 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI 基准。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — 将 NLI 用作分类器的基准工作。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 到 2026 年依然是 NLI 的主力工作。