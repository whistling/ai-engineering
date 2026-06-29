# 信息检索与搜索

> BM25 精确但脆弱。向量检索（dense）覆盖面广但会漏掉关键词。混合是 2026 年的默认。其他一切都是调优。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)  
**Time:** ~75 分钟

## 问题

用户输入“what happens if someone lies to get money” 并期望找到真正覆盖该行为的法规条款：“Section 420 IPC”。关键词搜索完全找不到（没有共享词汇）。如果嵌入没有在法律文本上训练，语义检索也会漏掉。真实的搜索必须同时处理两类情况。

IR 是每个 RAG 系统、每个搜索栏、每个文档站点模糊查找背后的管道。到 2026 年可在生产环境中工作的架构不是单一方法，而是一系列互补的方法链条，每一步捕捉前一步的失败。

本课构建每个组成部分并说明它们各自补救哪些失败。

## 概念

![混合检索：BM25 + 向量 + RRF + cross-encoder 重排序](../assets/retrieval.svg)

四层。按需选择。

1. **稀疏检索 (BM25)。** 快速，对精确匹配很精确，但语义能力差。运行于倒排索引上。对百万级文档的每次查询延迟小于 10ms。能帮你找到法规引用、产品编码、错误信息、命名实体等精确匹配项。
2. **向量检索（Dense retrieval）。** 将查询和文档编码为向量，进行最近邻搜索。能捕捉释义和语义相似性。会漏掉那些只差一个字符的精确关键词匹配。使用 FAISS 或向量数据库时每次查询大约 50–200ms。
3. **融合（Fusion）。** 合并稀疏和向量的排序列表。Reciprocal Rank Fusion (RRF) 是默认的便捷方案，因为它忽略原始分数（分数尺度不同）而只使用排名位置。当你知道某一信号在你的领域占主导时，也可以使用加权融合。
4. **Cross-encoder 重排序。** 对融合结果的前 30 条进行交叉编码器（query + document 一起输入，评分每对）重排序。保留前 5。Cross-encoder 对每对的计算比 bi-encoder 慢，但准确得多。因为只在 top-30 上运行，所以可以摊薄代价。

三路检索（BM25 + 向量 + learned-sparse，如 SPLADE）在 2026 年的基准上优于两路检索，但需要支持 learned-sparse 索引的基础设施。对大多数团队而言，两路加 cross-encoder 重排序是性价比最高的方案。

## 构建

### 第 1 步：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

有两个值得了解的参数。`k1=1.5` 控制词频饱和；越高意味着对词重复的权重越大。`b=0.75` 控制长度归一化；0 忽略文档长度，1 则完全归一化。这些默认值来自 Robertson 在原始论文中的建议，通常不需要调整。

### 第 2 步：使用 bi-encoder 的向量检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入做 L2 归一化，使点积等价于余弦相似度。`all-MiniLM-L6-v2` 是 384 维、速度快并且对大多数英文检索足够强的模型。多语种任务可使用 `paraphrase-multilingual-MiniLM-L12-v2`。若追求最高准确度，使用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第 3 步：Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

常数 `k=60` 来自原始 RRF 论文。更高的 `k` 会平滑排名差异的贡献；更低的 `k` 会让顶位排名占主导。60 是已发表的默认值，通常无需调优。

### 第 4 步：混合搜索 + 重排序

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三阶段合成。BM25 找到词汇匹配，向量找到语义匹配。RRF 在不需要分数校准的情况下合并两个排名。Cross-encoder 使用 query-document 对对 top-30 重新评分，捕捉 bi-encoder 漏掉的细粒度相关性。保留前 5。

### 第 5 步：评估

| Metric | Meaning |
|--------|---------|
| Recall@k | 在正确文档存在的查询中，多少比例的查询在 top-k 内找到该文档？ |
| MRR (Mean Reciprocal Rank) | 第一个相关文档的 1/排名 的平均值。 |
| nDCG@k | 考虑相关性分级，而不仅仅是二值的相关/不相关。 |

对于 RAG 来说，检索器的 **Recall@k** 是最重要的指标。如果检索集合里没有正确的段落，Reader 无法答对。

调试提示：对于失败的查询，对比稀疏和向量的排名。如果其中一个能找到正确文档而另一个找不到，说明存在词汇不匹配（修复：补上缺失的那半边）或语义歧义（修复：使用更好的嵌入或一个 reranker）。

## 使用

2026 年的栈：

| Scale | Stack |
|-------|-------|
| 1k-100k 文档 | 内存中 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。无需单独数据库。 |
| 100k-10M 文档 | 密集向量用 FAISS 或 pgvector，BM25 用 Elasticsearch / OpenSearch。并行运行。 |
| 10M+ 文档 | 使用支持混合检索的 Qdrant / Weaviate / Vespa / Milvus。对 top-30 做 cross-encoder 重排序。 |
| 最佳质量前沿 | 三路（BM25 + 向量 + SPLADE）+ ColBERT late-interaction 重排序 |

无论选什么，别忘了预算评估工作。先基准检索的 recall，再去基准端到端 RAG 的准确率。Reader 不能修复检索器遗漏的内容。

### 2026 年生产 RAG 的血泪经验

- **80% 的 RAG 失败归因于摄取和切分，而不是模型本身。** 团队会花几周时间换 LLM、调提示词，而检索却每三次查询就悄悄返回错误上下文。先修正切分。
- **切分策略比切分大小更重要。** 固定大小切分会破坏表格、代码和嵌套标题。以句子为单位是默认；对技术文档和产品手册，语义或基于 LLM 的切分能带来收益。
- **父文档模式。** 为了精确检索，检索小的“子”切片。当同一父节的多个子片段同时出现时，替换为父块以保留上下文。这通常能在不重训模型的情况下提升答案质量。
- **k_rerank=3 通常是最优的。** 超过该数量的每一个额外切片都会增加 token 成本和生成延迟，而不必然提升答案质量。如果 k=8 比 k=3 好，说明 reranker 表现不佳。
- **HyDE / 查询扩展。** 从查询生成一个假想答案并对其做嵌入检索。能桥接短问题与长文档之间的措辞差距。无需训练即可免费提升精度。
- **上下文预算低于 8K 个 token。** 若一致命中该限制，说明 reranker 阈值设置过松。
- **所有内容都要版本化。** 提示词、切分规则、嵌入模型、reranker。任何漂移都会悄然破坏答案质量。在 CI 中用忠实度、上下文精确率和未回答问题率作为门禁，阻止回归出现在用户面前。
- **三路检索（BM25 + 向量 + learned-sparse，如 SPLADE）在 2026 基准上优于两路检索，** 尤其对同时含有专有名词和语义的查询效果明显。基础设施支持 SPLADE 索引时就可以上线。

适当的检索设计能将幻觉减少 70–90%（2026 年行业测量）。大多数 RAG 性能提升来自更好的检索，而不是模型微调。

## 上线交付

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. 简单：在 500 篇文档的语料上实现上面的 `hybrid_search`。测试 20 个查询。比较 BM25-only、dense-only 与 hybrid 在 top-5 的 recall。
2. 中等：添加 MRR 计算。对于每个已知正确文档的测试查询，找出该文档在 BM25、dense 与 hybrid 排名中的位置。报告每种方法的 MRR。
3. 困难：使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域上微调一个密集编码器。用 500 对查询-文档对构建训练集。比较微调前后 recall。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BM25 | Keyword search | Okapi BM25。按词频、IDF 和长度对文档评分。 |
| Dense retrieval | Vector search | 将查询和文档编码为向量，查找最近邻。 |
| Bi-encoder | Embedding model | 独立编码查询和文档的模型。查询时速度快。 |
| Cross-encoder | Reranker model | 同时编码查询与文档的模型。慢但准确。 |
| RRF | Rank fusion | 通过求和 `1/(k + rank)` 的方式合并两个排名。 |
| Recall@k | Retrieval metric | 在 top-k 内包含相关文档的查询比例。 |

## 深入阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 的权威论文综述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，典型的 bi-encoder 方法。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 能收窄稀疏与稠密差距的 learned-sparse 检索器。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — late-interaction 检索。