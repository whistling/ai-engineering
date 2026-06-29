# 问答系统

> 三种系统塑造了现代问答（QA）：抽取式在文段中定位片段；检索增强则将答案落地到文档；生成式直接生成答案。每个现代 AI 助手通常是三者的混合体。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 5 · 11（机器翻译），Phase 5 · 10（注意力机制）  
**Time:** ~75 分钟

## 问题

用户输入 “When did the first iPhone launch?” 并期望得到 “June 29, 2007.”。不要得到 “Apple's history is long and varied.”，也不要只返回孤立的 “2007”。需要直接、落地且正确的答案。

在过去十年里，三种架构主导了 QA 领域。

- **抽取式问答（Extractive QA）。** 给定一个已知包含答案的问题和段落，找到答案在段落中的起止索引。SQuAD 是典型基准。
- **开放域问答（Open-domain QA）。** 没有给定段落。先检索相关段落，然后抽取或生成答案。这是当今所有 RAG 管道的基石。
- **生成式 / 闭卷问答（Generative / Closed-book QA）。** 大型语言模型直接从参数记忆中回答。没有检索步骤。推理最快，但事实可靠性最低。

截至 2026 年的趋势是混合：检索出若干最相关的段落，然后通过提示让生成式模型在这些段落基础上给出答案。这就是 RAG，本课程第 14 课深入讲检索部分。本课构建 QA 的回答侧。

## 概念

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**抽取式。** 将问题和段落一起用 Transformer（BERT 系列）编码。训练两个头分别预测答案的起始和结束 token 索引。损失是对合法位置的交叉熵。输出是段落中的一个片段。按构造永不产生幻觉，但也按构造无法处理段落中不存在答案的问题。

**检索增强（RAG）。** 两阶段。首先，检索器从语料库中找到 top-k 段落。第二，阅读器（抽取式或生成式）使用这些段落生成答案。检索器-阅读器的拆分使得两者可以独立训练和评估。现代 RAG 通常在两者之间加入一个重排序器（reranker）。

**生成式。** 解码器型 LLM（GPT、Claude、Llama）直接从学习到的权重回答。没有检索步骤。对于常识类知识表现优异，但在罕见或近期事实上灾难性失准。幻觉率与预训练数据中事实出现频率成反比。

## 构建它

### 步骤 1：使用预训练模型做抽取式 QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，包含不可回答的问题。默认情况下，`question-answering` pipeline 会返回得分最高的片段，即便模型的空答案（null）得分更高 —— 它不会自动返回空答案。若需显式的“无答案”行为，请在调用 pipeline 时传入 `handle_impossible_answer=True`：此时仅当空答案得分超过所有片段得分时才返回空答案。无论如何，总要检查返回的 `score` 字段。

### 步骤 2：一个检索增强管道（示意）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段管道。密集检索器（Sentence-BERT）按语义相似性找到相关段落。抽取式阅读器（RoBERTa-SQuAD）从合并的 top 段落中抽出答案片段。适用于小规模语料。若语料规模达百万文档，请使用 FAISS 或向量数据库。

### 步骤 3：生成式 RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示模版很重要。明确告诉模型以上下文为依据并在上下文不足时返回 "I don't know."，相比简单提示能将幻觉率降低约 40–60%。更复杂的模版会加入引用、置信度分数和结构化抽取。

### 步骤 4：反映真实世界的评估

SQuAD 使用 **Exact Match (EM)** 和 **token-level F1**。EM 是严格匹配，先归一化（小写、去标点、去冠词）后判断预测是否完全相同——否则得 0 分。F1 基于预测与参考答案之间的 token 重叠，给予部分奖励。两者都对同义改写惩罚较多：例如 “June 29, 2007” vs “June 29th, 2007” 通常 EM 得 0（序数词破坏归一化），但仍能从重叠 token 中获得较高 F1。

用于生产环境的 QA 指标包括：

- **答案准确率。** 由 LLM 或人工评判，因为自动指标无法完全捕捉语义等价性。
- **引用准确率。** 所引用的段落是否真正支持答案？可以通过将生成的引用字符串与检索到的段落做字符串匹配来自动检查。
- **拒绝校准。** 当答案不在检索到的段落中时，系统是否正确地说 “我不知道”？衡量错误的置信率。
- **检索召回率。** 在评估阅读器之前，先衡量检索器是否将正确段落放进 top-k。若检索器漏掉了正确段落，阅读器无法补救。

### RAGAS：2026 年的生产评估框架

`RAGAS` 专为 RAG 系统设计，并在 2026 年成为默认上船方案。它在无需金标准参考答案的情况下打分四个维度：

- **可信度（Faithfulness）。** 答案中的每个断言是否来源于检索到的上下文？通过基于 NLI 的蕴含检测来衡量。这是你的主要幻觉度量。
- **答案相关性（Answer relevance）。** 答案是否确实回答了问题？通过从答案生成假设问题并与真实问题比较来衡量。
- **上下文精确度（Context precision）。** 在检索到的块中，有多少比例实际上是相关的？低精确度 = 提示中有噪音。
- **上下文召回率（Context recall）。** 检索集合是否包含完成回答所需的全部信息？低召回率 = 阅读器无法成功。

无参考评分让你能在生产流量上评估模型而无需策划金标准答案。对于开放性问题，可在其上叠加 LLM 作为裁判。

安装：`pip install ragas`。将你的检索器 + 阅读器插入，针对每个查询得到四个标量。对回归进行告警。

## 使用它

2026 年技术栈。

| 用例 | 推荐 |
|---------|-------------|
| 给定段落，定位答案片段 | `deepset/roberta-base-squad2` |
| 在固定语料上，不能接受闭卷 | RAG：密集检索器 + LLM 阅读器 |
| 实时文档存取 | RAG，混合检索（BM25 + dense）+ 重排序器（见第 14 课） |
| 会话式问答（后续追问） | 带对话历史的 LLM + 每轮 RAG |
| 高度事实性、受监管领域 | 在权威语料上使用抽取式；绝不单独依赖生成式 |

抽取式 QA 在 2026 年不再流行，因为结合 LLM 的 RAG 能覆盖更多场景。但在需要逐字引用的场景仍会部署：法律检索、合规审核、审计工具等。

## 部署（Ship It）

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. 简单：在 10 个维基百科段落上搭建上述 SQuAD 抽取式管道。人工写 10 个问题。衡量答案正确率。如果段落和问题都干净，你应该看到 7–9 个正确。
2. 中等：加入拒绝分类器。当最高检索分低于阈值（例如余弦 0.3）时，返回 “我不知道” 而不是调用阅读器。在验证集上调优阈值。
3. 困难：在 10,000 文档语料上构建 RAG 管道。实现混合检索（BM25 + dense）并用 RRF 融合（见第 14 课）。衡量有/无混合检索时的答案准确率。记录哪些题型受益最多。

## 关键词

| 术语 | 常说法 | 实际含义 |
|------|--------|---------|
| 抽取式问答 (Extractive QA) | Find the answer span | 在给定段落中预测答案的起止索引。 |
| 开放域问答 (Open-domain QA) | QA over a corpus | 没有给定段落；必须先检索再回答。 |
| RAG | Retrieve then generate | Retrieval-augmented generation（检索增强生成）。检索器 + 阅读器管道。 |
| SQuAD | Canonical benchmark | Stanford Question Answering Dataset。使用 EM + F1 指标。 |
| 幻觉 / 编造答案 (Hallucination) | Made-up answer | 阅读器输出未被检索到的上下文支持的答案。 |
| 拒绝校准 (Refusal calibration) | Know when to shut up | 系统在无法回答时正确地说 “我不知道”。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文。  
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，问答中经典的密集检索器。  
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名并系统化 RAG 的论文。  
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — RAG 的综合调查。