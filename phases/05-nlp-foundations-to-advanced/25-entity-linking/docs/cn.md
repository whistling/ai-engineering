# 实体链接与消歧

> NER 发现了 "Paris."。实体链接决定：是巴黎（法国）？巴黎·希尔顿？德克萨斯州的巴黎？特洛伊王子帕里斯？没有链接，你的知识图谱会保持模糊不清。

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (共指消解)  
**Time:** ~60 分钟

## 问题描述

一句话写着："Jordan beat the press." 你的 NER 将 "Jordan" 标注为 PERSON。很好。但到底是哪位 Jordan？

- Michael Jordan（篮球）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利的机器学习教授 — 是的，在 ML 论文中这种混淆是真实存在的）？
- Jordan（国家）？
- Jordan（希伯来人名）？

实体链接（Entity Linking，EL）将每个提及解析为知识库中的唯一条目：Wikidata、Wikipedia、DBpedia，或你的领域 KB。有两个子任务：

1. 候选生成（Candidate generation）。给定 "Jordan"，哪些 KB 条目是合理的候选？
2. 消歧（Disambiguation）。给定上下文，哪个候选是正确的？

这两步都是可学习的，并且都有基准测试。端到端流程在过去十年很稳定——变化的是消歧器的质量。

## 概念

![实体链接流程：提及 → 候选 → 消歧后的实体](../assets/entity-linking.svg)

候选生成。给定提及的表面形式（"Jordan"），在别名索引中查找候选。维基百科的别名词典覆盖大多数命名实体：“JFK” → John F. Kennedy、Jacqueline Kennedy、JFK 机场、电影《JFK》。典型索引每个提及返回 10–30 个候选。

消歧：三种方法。

1. 先验 + 上下文（Milne & Witten, 2008）。`P(entity | mention) × context-similarity(entity, text)`。效果好、速度快、不需训练。
2. 基于嵌入（Embedding-based，ESS / REL / Blink）。对提及 + 上下文进行编码；对每个候选的描述进行编码；取余弦相似度最大的。2020–2024 年间的默认选择。
3. 生成式（GENRE, 2021；基于 LLM 的方法，2023+）。逐字（或逐 token）解码实体的规范名称。通过限制解码到有效实体名的 trie 来保证输出是有效的 KB id。

端到端 vs 流水线。现代模型（ELQ、BLINK、ExtEnD、GENRE）可以在一次前向中运行 NER + 候选生成 + 消歧。流水线系统在生产中仍占主导，因为可以替换组件。

### 两项度量

- 提及召回率（mention recall，候选生成）。在金标提及中，正确的 KB 条目出现在候选列表中的比例。是整个流水线的下限。
- 消歧准确率 / F1。给定正确候选时，top-1 有多常是正确的。

始终报告两者。一套在候选召回为 80% 时消歧 99% 的系统，其流水线整体只达 80%。

## 实现

### 第一步：从维基百科重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

维基百科别名数据约有 ~1800 万对 (alias, entity)。可从 Wikidata dumps 下载。存为倒排索引。

### 第二步：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠只是个玩具示例。用嵌入上的余弦相似度代替会更好（参见 `code/main.py` 中的 transformer 版本 step-2）。

### 第三步：基于嵌入（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

在索引构建时，对每个 KB 实体只做一次嵌入。在查询时，对提及 + 上下文做一次嵌入，与候选池做点积，取最大者。

### 第四步：生成式实体链接（概念）

GENRE 按字符逐步解码实体的维基百科标题。受限解码（见第 20 课）确保只输出有效标题。紧密结合 KB 支持的 trie。现代后继者有 REL-GEN 与基于 LLM 的结构化输出 EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

配合白名单（概述中的 `choice`），这是 2026 年最容易上手的 EL 管道之一。

### 第五步：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准 EL 基准：1,393 篇路透社文章、34k 提及、维基百科实体。报告 in-KB 准确率（`P@1`）和 out-of-KB 的 NIL 检测率。

## 陷阱

- NIL 处理。部分提及不在 KB 中（新兴实体、鲜为人知的人物）。系统必须预测 NIL，而不是胡乱猜测错误实体。单独度量。
- 提及边界错误。上游 NER 漏掉部分跨度（例如将 "Bank of America" 只标注为 "Bank"）。EL 召回下降。
- 流行度偏差。训练系统倾向于过度预测高频实体。在 ML 论文中提到的 "Michael I. Jordan" 经常被链接到篮球的 Jordan。
- 跨语言实体链接。将中文文本中的提及映射到英文维基百科实体。需要多语种编码器或翻译步骤。
- KB 陈旧。新的公司、事件、人物可能不在去年或更早的维基百科 dump 中。生产管线需要刷新机制。

## 使用建议

| 场景 | 选择 |
|------|------|
| 通用英语 + Wikipedia | BLINK 或 REL |
| 跨语言，KB = Wikipedia | mGENRE |
| 面向 LLM、每天提及量少 | 使用 Claude/GPT-4 的提示 + 候选列表 + 受限 JSON 输出 |
| 特定领域 KB（医疗、法律） | 定制 BERT 加上 KB 感知检索并在领域 AIDA 风格数据上微调 |
| 极低延迟场景 | 仅用精确匹配的先验（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

到 2026 年可上线的生产模式：NER → 共指消解 → 对每个提及做 EL → 将簇内提及折叠为一个规范实体。输出：文档中每个实体一个 KB id，而不是每个提及一个。

## 上线交付

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习

1. 简单题。实现 `code/main.py` 中的先验 + 上下文消歧器，对 10 个有歧义的提及（Paris、Jordan、Apple）进行测试。手工标注正确实体。计算准确率。
2. 中等题。用 sentence transformer 对 50 个歧义提及做编码。为每个候选嵌入描述。比较基于嵌入的消歧与 Jaccard 上下文重叠的效果。
3. 困难题。构建一个 1k 条目的领域 KB（例如公司员工 + 产品）。实现从 NER 到 EL 的端到端系统。在 100 条保留句子上测精确率和召回率。

## 术语表

| 术语 | 常说的叫法 | 实际含义 |
|------|-----------|---------|
| 实体链接（EL） | Link to Wikipedia | 将一个提及映射到唯一的 KB 条目。 |
| 候选生成 | Who could it be? | 返回该提及的可行 KB 候选列表。 |
| 消歧 | Pick the right one | 使用上下文为候选打分，选择赢家。 |
| 别名索引 | The lookup table | 从表面形式映射到候选实体。 |
| NIL | Not in KB | 明确预测没有匹配的 KB 条目。 |
| KB | Knowledge base | Wikidata、Wikipedia、DBpedia 或你的领域 KB。 |
| AIDA-CoNLL | The benchmark | 1,393 篇路透社文章带有金标实体链接。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 奠基性的先验 + 上下文方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于嵌入的实用方法。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 带受限解码的生成式实体检索。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 基准论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开放的生产级堆栈。