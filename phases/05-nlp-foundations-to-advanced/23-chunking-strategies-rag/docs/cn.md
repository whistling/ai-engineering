# RAG 的分块策略

> 分块配置对检索质量的影响与嵌入模型的选择一样重要（Vectara NAACL 2025）。分块做错了，再多的重排也救不了你。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 5 · 14 (信息检索), Phase 5 · 22 (嵌入模型)  
**Time:** ~60 分钟

## 问题

你把一份 50 页的合同放进 RAG 系统。用户问：“终止条款是什么？”检索器返回了封面页。为什么？因为模型在训练时使用的是 512 令牌的块，终止条款位于 20 页之后，跨页被拆分，且与查询没有局部关键词关联。

解决方法不是“买更好的嵌入模型”，而是分块。大小多少合适？重叠多少？在哪里拆分？是否保留周边上下文？

2026 年 2 月的基准测试展示了令人惊讶的结果：

- Vectara 的 2026 研究：递归 512 令牌分块在准确率上击败语义分块，69% → 54%。
- 在 Natural Questions 上，SPLADE + Mistral-8B：重叠并没有带来可测量的好处。
- 上下文断崖（context cliff）：在约 2,500 令牌的上下文长度附近，响应质量急剧下降。

“显而易见”的答案（语义分块、20% 重叠、1000 令牌）通常是错的。本课旨在建立对六种策略的直觉，并告诉你在何时使用哪一种。

## 概念

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**固定分块（Fixed chunking）。** 每 N 个字符或令牌切分。最简单的基线。会在句子中间切断。压缩效果好，但连贯性差。

**递归（Recursive）。** LangChain 的 `RecursiveCharacterTextSplitter`。优先按 `\n\n` 切分，再按 `\n`、`.`、空格。回退机制良好。2026 年的默认选择。

**语义分块（Semantic）。** 对每个句子做嵌入。计算相邻句子之间的余弦相似度。当相似度低于阈值时切分。能保持主题连贯。较慢；有时会产生只有 40 令牌的微小片段，反而损害检索效果。

**句子分块（Sentence）。** 按句子边界切分。每块一个句子或 N 个句子的窗口。通常在代价更小的情况下匹配语义分块在 ~5k 令牌以内的效果。

**父文档（Parent-document）。** 同时为检索存小的子块（child chunks）和更大的父块（parent chunk）。按子块检索，返回父块。退化表现更优雅：即使子块不理想，仍能返回合理的父块上下文。

**后期分块（Late chunking，2024）。** 先在令牌级别对整篇文档做嵌入，然后将令牌嵌入汇聚成块嵌入。保留跨块上下文。适用于长上下文嵌入器（如 BGE-M3、Jina v3）。计算开销更高。

**上下文检索（Contextual retrieval，Anthropic，2024）。** 在每个块前加上由 LLM 生成的关于该块在文档中位置的摘要（“该块是终止条款的 3.2 节……”）。Anthropic 的基准显示检索提升 35–50%。构建索引开销较大。

### 胜过所有默认设置的规则

将分块大小与查询类型匹配：

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

来自 NVIDIA 的 2026 基准。分块应当既足够大以包含答案及局部上下文，又足够小以使检索器的 top-K 更聚焦于答案而非噪声上下文。

## 构建步骤

### 步骤 1：固定与递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 步骤 2：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域上调节 `threshold`。阈值太高会产生碎片；太低会形成一个巨大的块。

### 步骤 3：父-子文档（parent-document）

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞见：去重父文档。多个子块可能映射到同一个父块；返回所有会浪费上下文。

### 步骤 4：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引这些带上下文的块。查询时，额外的上下文信号能提升检索效果，但索引成本较高。

### 步骤 5：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终进行基准测试。对你语料最优的策略可能与任何博客文章都不同。

## 陷阱

- **仅在事实型查询上评估分块。** 多跳查询会显示完全不同的优胜策略。使用按查询类型分层的评估集。
- **语义分块没有最小大小约束。** 会产生 40 令牌的碎片，损害检索。始终强制 `min_tokens`。
- **把重叠当作迷信（cargo cult）。** 2026 年的研究发现重叠常常没有任何收益，却会使索引成本翻倍。测量，不要假设。
- **不强制最小/最大值。** 5 令牌或 5000 令牌的块都会破坏检索。要做截断与钳位（clamp）。
- **跨文档分块。** 绝不要让一个块跨越两个文档。始终先按文档分块，再视需要合并。

## 使用场景

2026 年推荐栈：

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| Heavy cross-reference (contracts, papers) | Late chunking or contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances (tweets, reviews) | One document = one chunk |

从 `Recursive`、512 开始。在 50 条查询的评估集上测量 recall@5。再据此调优。

## 发布（Ship It）

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. 简单：用 fixed(512, 0)、recursive(512, 0) 和 recursive(512, 100) 对一份 20 页文档做分块。比较分块数量和边界质量。
2. 中等：在 5 篇文档上构建 30 条查询的评估集。比较 recursive、semantic 和 parent-document 的 recall@5。哪一个胜出？是否与博客文章结论一致？
3. 困难：实现上下文检索。衡量相对于基线 recursive 的 MRR 提升。报告索引成本（LLM 调用次数）与精度增益的对比。

## 关键术语

| 术语 | 常说法 | 实际含义 |
|------|--------|---------|
| 分块（Chunk） | 文档的一部分 | 用作嵌入、索引与检索的子文档单元。 |
| 重叠（Overlap） | 安全边界 | 相邻块共享的 N 个令牌；在 2026 年基准中常常无效。 |
| 语义分块（Semantic chunking） | 智能切分 | 在相邻句子嵌入相似度下降处切分。 |
| 父文档（Parent-document） | 两级检索 | 检索小的子块，返回更大的父块。 |
| 后期分块（Late chunking） | 嵌入后分块 | 在令牌级别为整篇文档做嵌入，再汇聚成块向量。 |
| 上下文检索（Contextual retrieval） | Anthropic 的技巧 | 在索引前为每个块加上由 LLM 生成的摘要前缀。 |
| 上下文断崖（Context cliff） | 2500 令牌的墙 | 在 RAG 中约 2.5k 令牌处观测到的质量下跌（2026 年 1 月）。 |

## 延伸阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产环境中的默认做法。  
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — 分块的重要性与嵌入选择不相上下。  
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 关于后期分块的文章。  
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 通过 LLM 生成的上下文前缀获得 35–50% 的检索提升。  
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型的分块大小指南。