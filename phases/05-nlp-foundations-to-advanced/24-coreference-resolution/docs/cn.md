# Coreference Resolution

> “She called him. He did not answer. The doctor was at lunch.” 三个指称涉及两个人，但没有人被点名。共指解析（coreference resolution）就是弄清谁是谁。

**Type:** 学习
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 分钟

## 问题描述

从一篇约 300 字的文章中抽取对 Apple Inc. 的每一次提及。当文章直接写着 “Apple” 时很容易；但当它写着 “the company”（该公司）、“they”（他们）、“Cupertino's technology giant”（库比蒂诺的科技巨头）、或 “Jobs's firm”（乔布斯的公司）时就很难。如果不把这些提及解析成指向同一实体，你的 NER 管道会漏掉 60–80% 的提及。

共指解析将指向同一现实世界实体的所有表达链接为一个簇（cluster）。它是表层 NLP（NER、句法分析）和下游语义（信息抽取、问答、摘要、知识图谱）之间的粘合剂。

为什么在 2026 年仍然重要：

- 摘要： “The CEO announced...” vs “Tim Cook announced...” — 摘要应该点名 CEO。
- 问答： “Who did she call?” 需要解析 “she”。
- 信息抽取：如果知识图谱出现 “PER1 founded Apple” 和 “Jobs founded Apple” 作为两个独立条目就是错误的。
- 跨文档信息抽取：将关于同一事件的多篇文章中的提及合并是跨文档共指问题。

## 概念

![Coreference clustering: mentions → entities](../assets/coref.svg)

**任务。** 输入：一篇文档。输出：一组提及（span）的聚类，其中每个聚类指向一个实体。

提及类型（Mention types）

- **命名实体。** “Tim Cook”
- **名词性提及（Nominal）。** “the CEO”，“the company”
- **代词性提及（Pronominal）。** “he”，“she”，“they”，“it”
- **同位语（Appositive）。** “Tim Cook, Apple's CEO,”

架构（Architectures）

1. **基于规则（Hobbs, 1978）。** 基于句法树的代词消解，使用语法规则。作为基线效果很好。在代词解析上令人惊讶地难以被击败。
2. **提及对分类器（Mention-pair classifier）。** 对每对提及 (m_i, m_j) 预测是否共指。通过传递闭包聚类。2016 年前的常规方法。
3. **提及排序（Mention-ranking）。** 对每个提及，对候选先行项进行排序（包括“没有先行项”）。选择排名最高的。
4. **基于 span 的端到端模型（Lee et al., 2017）。** Transformer 编码器。枚举所有长度上限内的候选 span。预测提及得分。预测每个 span 的先行项概率。贪心聚类。现代默认方法。
5. **生成式（2024+）。** 提问 LLM：“列出本文中每个代词及其先行项。” 在简单情况表现很好，在长文档和罕见指称上存在困难。

评估指标。五个标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有单一指标能全面衡量聚类质量。通常报告前三个的平均值作为 CoNLL F1。2026 年在 CoNLL-2012 上的 SOTA：约 83 F1。

已知难点

- 定指描述指向数页前引入的实体。
- 桥接照应（bridging anaphora）——“the wheels”（车轮）指向之前提到的一辆车。
- 中文、日文等语言中的零照应（zero anaphora）。
- 前照（cataphora）：代词在先，指称在后：“When **she** walked in, Mary smiled.”（当她走进来时，玛丽微笑了。）

## 构建方法

### 第 1 步：预训练神经共指（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在较长的文档上，你可能得到类似结果：
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### 第 2 步：基于规则的代词解析器（教学用途）

查看 `code/main.py`，有一个仅使用标准库的实现：

1. 抽取提及：命名实体（大写短语）、代词（词典查找）、定指描述（“the X”）。
2. 对每个代词，查看前 K 个提及并按分数排序，评分依据包括：
   - 性别/数一致性（启发式）
   - 新近性（越近越好）
   - 句法角色（主语优先）
3. 链接得分最高的先行项。

在与神经模型比较时竞争力不足。但它展示了搜索空间以及端到端模型必须做出的决策。

### 第 3 步：使用 LLM 做共指

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

要注意的两个失败模式。首先，LLM 会过度合并（把指向两个不同人的 “him” 和 “her” 误认为同一人）。其次，LLM 在长文档中会默默丢弃提及。始终用 span-offset 校验输出。

### 第 4 步：评估

标准的 conll-2012 脚本计算 MUC、B³、CEAF-φ4 并报告平均值。用于内部评估时，从你的标注测试集开始，先做基于 span 的精确率和召回率，然后再加入提及链接的 F1。

## 陷阱

- 单例爆炸（Singleton explosion）。一些系统会把每个提及都报告为独立簇。B³ 比较宽松，MUC 会惩罚这种情况。始终同时查看三种指标。
- 长上下文中的代词。文档超过 2000 个 token 时性能下降约 15 F1。要谨慎分块。
- 性别假设。硬编码的性别规则会在非二元指称、机构、动物上失效。使用学习到的模型或中性评分。
- LLM 在长文档上的漂移。一次 API 调用无法可靠地对 50+ 段落进行聚类。使用滑动窗口并合并结果。

## 使用建议

2026 年技术栈：

| 情景 | 选择 |
|------|------|
| English, single document | `en_coreference_web_trf` (spaCy-experimental) 或 AllenNLP 神经共指 |
| Multilingual | 在 OntoNotes 或 Multilingual CoNLL 上训练的 SpanBERT / XLM-R |
| Cross-document event coref | 专门的端到端模型（2025–26 SOTA） |
| Quick LLM baseline | 使用 GPT-4o / Claude 并配结构化输出的共指提示 |
| Production dialog systems | 规则回退 + 神经主流程 + 对关键槽位的人工复核 |

2026 年的常见集成模式：先运行 NER，再运行共指，将共指簇合并到 NER 实体中。下游任务看到的是每个簇对应一个实体，而不是每个提及对应一个实体。

## 上线（Ship It）

将以下内容保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: 选择共指方法、评估计划和集成策略。
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

给定一个用例（单文档 / 多文档、领域、语言），输出：

1. 方法。Rule-based / neural span-based / LLM-prompted / hybrid。用一句话说明理由。
2. 模型。若为神经方法，给出命名的 checkpoint。
3. 集成。操作顺序：tokenize → NER → coref → 下游任务。
4. 评估。在保留集上报告 CoNLL F1（MUC + B³ + CEAF-φ4 平均）+ 对 20 篇文档的人工簇审查。

对于超过 2000 个 token 的文档，拒绝仅用 LLM 的共指（除非采用滑动窗口并合并）。拒绝任何在未提供提及级精确率/召回报告的情况下运行共指的流水线。在人口学多样的文本中部署基于性别启发式的系统需要打上标记（flag）。
```

## 练习

1. 简单：在 5 个手工设计的段落上运行 `code/main.py` 中的基于规则的解析器。对照真值测量提及链接准确率。
2. 中等：在一篇新闻文章上使用预训练的神经共指模型。将聚类与自己手工标注的结果比较。分析失败的场景。
3. 困难：构建一个共指增强的 NER 管道：先 NER，然后通过共指簇合并。对 100 篇文章测量实体覆盖率相对于仅 NER 的提升。

## 关键词

| 术语 | 常用说法 | 实际含义 |
|------|---------|---------|
| Mention | A reference | 指称：文本中指向实体的一个 span（名字、代词、名词短语）。 |
| Antecedent | What "it" refers to | 先行项：一个后出现的提及所共指的早期提及。 |
| Cluster | The entity's mentions | 簇：所有指向同一现实世界实体的提及集合。 |
| Anaphora | Backward reference | 回指（向后引用）：后面的提及指向前面的（“he” → “John”）。 |
| Cataphora | Forward reference | 前照（向前引用）：前面的提及指向后面的（“When he arrived, John...”）。 |
| Bridging | Implicit reference | 桥接：隐含指称（“I bought a car. The wheels were bad.” → 指的是那辆车的车轮）。 |
| CoNLL F1 | The number on leaderboards | CoNLL F1：MUC、B³、CEAF-φ4 三项 F1 的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 权威教科书章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 基于 span 的端到端方法。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 提升共指的预训练方法。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 基准任务。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典论文。