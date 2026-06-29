# 关系抽取与知识图谱构建

> NER 识别出实体。实体链接将其锚定。关系抽取发现它们之间的边。知识图谱是节点、边及其来源的总和。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 5 · 06 (NER)，Phase 5 · 25 (实体链接)  
**Time:** ~60 分钟

## 问题描述

分析员读到：“Tim Cook became CEO of Apple in 2011.” 四条事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（Relation Extraction, RE）把自由文本转换为结构化三元组 `(subject, relation, object)`。在语料上汇总便得到知识图谱。聚合并查询它们，就可以为 RAG、分析或合规审计提供推理基底。

2026 年的问题：大型模型对抽取关系很热心——太热心了。它们会虚构文本并不支持的三元组。没有来源证据，你无法区分真实三元组与貌似合理的谬误。2026 年的答案是 AEVS 风格的锚定并验证流水线。

## 概念

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系可以来自闭合本体（Wikidata 属性、FIBO、UMLS）或开放集合（OpenIE 风格，任意短语均可）。

**三种抽取方法。**

1. **基于规则 / 模式。** Hearst 模式： "X such as Y" → `(Y, isA, X)`。加上手工正则。脆弱但精确，可解释。
2. **监督分类器。** 给定句子中的两个实体提及，预测固定集合中的关系类型。在 TACRED、ACE、KBP 上训练。2015–2022 年的标准做法。
3. **生成式 LLM。** 提示模型输出三元组。开箱即可工作。需要来源证据，否则会幻觉产生看起来合理的垃圾。

**AEVS（Anchor-Extraction-Verification-Supplement，锚定-抽取-验证-补充，2026）。** 当前的幻觉缓解框架：

- **Anchor（锚定）。** 标识每个实体跨度和关系短语跨度并记录精确位置。
- **Extract（抽取）。** 生成与锚定跨度关联的三元组。
- **Verify（验证）。** 将每个三元组元素匹配回源文本；拒绝任何不被支持的项。
- **Supplement（补充）。** 做一次覆盖检查，确保没有被锚定的跨度被遗漏。

幻觉显著下降。需要更多计算但可审计。

**开放式 vs 闭合集合的权衡。**

- **闭合本体。** 固定属性列表（例如 Wikidata 的 11,000+ 属性）。可预测、可查询、不易被发明。
- **Open IE。** 任意动词短语都可成为关系。召回高，精确低，查询混乱。

生产环境的知识图谱通常混合使用：用 Open IE 做发现，然后将关系规一化到闭合本体，再合并入主图谱。

## 实现

### 步骤 1：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

参见 `code/main.py` 中的完整玩具抽取器。Hearst 模式在特定领域流水线中仍然在用，因为它们易于调试。

### 步骤 2：监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 seq2seq 的关系抽取器：文本进，三元组出，已经使用 Wikidata 属性 id。使用远监督数据微调。是开源权重的基线方案。

### 步骤 3：带锚定的 LLM 提示抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

对每个返回的跨度在源文本中做校验。拒绝任何满足 `text[start:end] != triple_entity` 的结果。这就是 AEVS 中“验证”步骤的最小实现。

### 步骤 4：规一化到闭合本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

（注：上方注释已保留为原始英文注释，代码块内的注释在需要时可翻译以便开发者阅读。）

规一化通常占据 60–80% 的工程工作量。务必在预算中预留时间。

### 步骤 5：构建小图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个 RAG-over-KG 系统的原子操作。按需使用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强的图存储来扩展。

## 陷阱

- **在 RE 之前做共指消解。** “He founded Apple”——RE 需要知道 “he” 指谁。先做共指（参考 lesson 24）。
- **实体规一化。** “Apple Inc” 与 “Apple” 必须解析为相同节点。先做实体链接（参考 lesson 25）。
- **虚构三元组。** LLM 会产生文本不支持的三元组。强制执行跨度验证。
- **关系规一化漂移。** Open IE 关系不一致（“was born in”、“came from”、“is a native of”）。规约到规范 id，否则图无法查询。
- **时间错误。** “Tim Cook is CEO of Apple”——现在成立，但在 2005 年不成立。许多关系是有时间范围的。使用限定词（Wikidata 的 `P580` 起始时间、`P582` 结束时间）。
- **领域不匹配。** REBEL 在维基百科上训练。法律、医学、科学文本通常需要领域微调的 RE 模型。

## 使用建议

2026 年的技术栈：

| Situation | Pick |
|-----------|------|
| Fast production, general domain | REBEL or LlamaPred with Wikidata canonicalization |
| Domain-specific (biomed, legal) | SciREX-style domain fine-tune + custom ontology |
| LLM-prompted, audited output | AEVS pipeline: anchor → extract → verify → supplement |
| High-volume news IE | Pattern-based + supervised hybrid |
| Building a KG from scratch | Open IE + manual canonicalization pass |
| Temporal KG | Extract with qualifiers (start/end time, point in time) |

集成模式：NER → 共指消解 → 实体链接 → 关系抽取 → 本体映射 → 图加载。每个阶段都是潜在的质量门控点。

## 部署示例（保存为 outputs/skill-re-designer.md）

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习

1. 简单：在 `code/main.py` 上对 5 条新闻句子运行模式抽取器。人工检验精确率。
2. 中等：对相同句子使用 REBEL（或一个小型 LLM）。比较三元组。哪个抽取器精确率更高？召回率更高？
3. 困难：构建 AEVS 流水线：用 LLM 抽取 + 将跨度在源文中验证。对 50 条维基风格句子，在验证步骤前后测量幻觉率。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Triple | Subject-relation-object | 三元组 `(s, r, o)`，是知识图谱的原子单元。 |
| Open IE | Extract anything | 开放式信息抽取（Open IE）：开放词汇的关系短语，召回高、精确低。 |
| Closed ontology | Fixed schema | 闭合本体：有界的关系类型集合（Wikidata、UMLS、FIBO）。 |
| Canonicalization | Normalize everything | 规一化：将表面名称 / 关系映射到规范 id。 |
| AEVS | Grounded extraction | AEVS：锚定-抽取-验证-补充流水线（2026 年提出的落地抽取框架）。 |
| Provenance | Source-of-truth link | 源证据：每个三元组包含文档 id + 字符跨度以指向其来源。 |
| Distant supervision | Cheap labels | 远监督：把文本与现有 KG 对齐以生成训练数据（廉价标注）。 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远监督关系抽取的开创论文。  
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — 基于 seq2seq 的关系抽取工作（REBEL）。  
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合信息抽取方法（DyGIE++）。  
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 年的幻觉缓解设计（AEVS 框架）。  
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — Wikidata 的规范图查询入门。