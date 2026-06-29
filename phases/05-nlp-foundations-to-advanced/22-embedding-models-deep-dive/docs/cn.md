# 嵌入模型 — 2026 年深度解析

> Word2Vec 给你每个单词一个向量。现代嵌入模型给你每个段落一个向量、跨语言支持，并提供稀疏、稠密和多向量视图，尺寸可按索引需求调整。选错了，你的 RAG 就会检索到错误的内容。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (信息检索)  
**Time:** ~60 分钟

## 问题

你的 RAG 系统有 40% 的时间检索到错误段落。罪魁祸首很少是向量数据库或提示词，往往是嵌入模型。

在 2026 年选择嵌入意味着要在五个维度上权衡：

1. **稠密 vs 稀疏 vs 多向量。** 每段一个向量，还是每个 token 一个向量，或是稀疏加权词袋。
2. **语言覆盖。** 单语英文模型在纯英文任务上仍然领先。多语模型在混合语料上表现更好。
3. **上下文长度。** 512 token 与 8,192 或 32,768 —— 实际有效容量常常只有标称最大值的 60–70%。
4. **维度预算。** 3,072 个浮点数（全精度）≈ 每向量 12 KB。100M 向量时，存储费用约为 $1,300/月。Matryoshka 截断可以把这个成本削减 4×。
5. **开源权重 vs 托管 API。** 开源权重让你掌控栈和数据；托管意味着用可用的最新模型换取对堆栈的可控性的放弃。

本课将把权衡点点名，让你基于证据而不是上季度流行什么来做选择。

## 概念

![密集、稀疏和多向量嵌入](../assets/embedding-modes.svg)

**稠密嵌入。** 每段一个向量（通常 384–3,072 维）。用余弦相似度对段落按语义接近性排序。示例：OpenAI 的 `text-embedding-3-large`、BGE-M3 的稠密模式、Voyage-3。默认选择。

**稀疏嵌入。** 类似 SPLADE。一个 transformer 预测每个词表 token 的权重，然后将大部分置零。得到的就是 |vocab| 维的稀疏向量。它捕获词汇匹配（类似 BM25），但使用学习到的词项权重。在关键词密集的查询上表现强劲。

**多向量（后期交互）。** ColBERTv2、Jina-ColBERT。每个 token 一个向量。用 MaxSim 打分：对每个查询 token 找到最相似的文档 token，然后累加得分。存储和打分更昂贵，但在长查询和领域特定语料上获胜。

**BGE-M3：三合一。** 单个模型同时输出稠密、稀疏和多向量表示。各自可独立查询；分数通过加权和融合。想要从单个 checkpoint 获得灵活性时，2026 年的默认选择。

**Matryoshka 表示学习。** 训练使向量的前 N 维构成有用的独立嵌入。将 1,536 维向量截断到 256 维通常只损失 ~1% 的准确度，但节省 6× 存储。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 支持此特性。

### MTEB 排行榜只讲部分事实

Massive Text Embedding Benchmark（MTEB）——发布时包括 56 个任务、8 个任务类型（2022），在 MTEB v2 扩展到 100+ 任务。2026 年初，Gemini Embedding 2 在检索上领先（67.71 MTEB-R）。Cohere embed-v4 在通用任务上领先（65.2 MTEB）。BGE-M3 在开源多语模型中领先（63.0）。排行榜有参考价值，但不充分——务必在你的领域上做基准。

### 三层模式

| 用例 | 模式 |
|------|------|
| 快速初筛 | 稠密双编码器（BGE-M3、text-3-small） |
| 提升召回 | 稀疏（SPLADE、BGE-M3 sparse）+ RRF 融合 |
| Top-50 精确率 | 多向量（ColBERTv2）或 cross-encoder 重排序器 |

大多数生产堆栈都会同时使用这三类方法。

## 实建

### 步骤 1：基线 — 用 Sentence-BERT 做稠密嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使点积等同于余弦相似度。务必设置它。

### 步骤 2：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后需重新归一化。Nomic v1.5、OpenAI text-3 与 Voyage-4 的训练使得前几个截断级别损失很小。非 Matryoshka 模型（原始 Sentence-BERT）在截断时会急剧退化。

### 步骤 3：BGE-M3 的多功能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

一次推理，生成三类索引。分数融合示例：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调节权重。

### 步骤 4：在自定义任务上做 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在具有代表性的小样本子集上运行候选模型。不要只信排行榜 —— 你的领域更重要。

### 步骤 5：从零开始手写余弦

见 `code/main.py`。平均哈希技巧的嵌入（仅标准库示例）。虽然无法与 transformer 嵌入竞争，但能展示流程：分词 → 向量化 → 归一化 → 点积。

## 陷阱

- **查询和文档使用同一模型。** 有些模型（如 Voyage、Jina-ColBERT）使用非对称编码 —— 查询与文档走不同路径。务必查看模型卡。
- **忘记前缀。** `bge-*` 系列模型需要在查询前加上 `Represent this sentence for searching relevant passages: ` 这类前缀。忘记会导致 3–5 个点的召回损失。
- **过度截断 Matryoshka。** 1,536 → 256 通常安全；1,536 → 64 则不安全。务必在评估集上验证。
- **上下文被截断。** 大多数模型会在输入超过最大长度时静默截断。长文档需要切分（见第 23 课）。
- **忽视延迟尾部。** MTEB 分数不反映 p99 延迟。一个 600M 模型可能在分数上比 335M 模型高 2 个点，但每次查询成本可能高 3×。

## 使用建议

2026 年堆栈：

| 情况 | 选择 |
|------|------|
| 仅英文、要求快速、API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重、英文 | `BAAI/bge-large-en-v1.5` |
| 开源权重、多语 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（137M 参数，MoE） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词密集的查询 | 添加 SPLADE 稀疏表示，与稠密表示做 RRF 融合 |

2026 年常见模式：以 BGE-M3 或 text-3-large 起步，在你的领域用 MTEB 评估；如果某个领域专用模型领先超过 3 个点则替换。

## 上线交付

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

（注：上面代码块为示例配置文件，保留原样）

## 练习

1. 简单：用 `bge-small-en-v1.5` 编码 100 条句子，使用完整维度（384），然后使用 Matryoshka 截断到 128。对 10 个查询测量 MRR 下降。
2. 中等：在你领域的 500 条段落上比较 BGE-M3 的稠密、稀疏和 ColBERT 表现。哪种在 recall@10 上领先？RRF 融合是否胜过最好的单一模式？
3. 困难：在你的前两项领域任务上，对三款候选模型运行 MTEB。报告 MTEB 得分、100 查询批次的 p99 延迟，以及每 1M 查询的费用。选择帕累托最优模型。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|---------|---------|
| Dense embedding | 向量 | 每段固定大小向量。用余弦相似度排序。 |
| Sparse embedding | 学习到的 BM25 | 每个词表 token 的权重；大部分为零；端到端训练得到。 |
| Multi-vector | ColBERT 风格 | 每个 token 一个向量；MaxSim 打分；索引更大但召回更好。 |
| Matryoshka | 俄罗斯套娃技巧 | 向量的前 N 维本身就是有效的更小嵌入。 |
| MTEB | 基准 | Massive Text Embedding Benchmark —— 发布时 56 个任务，v2 超过 100 个任务。 |
| BEIR | 检索基准 | 18 个零样本检索任务；常用于跨领域鲁棒性评估。 |
| Asymmetric encoding | 查询 ≠ 文档路径 | 查询与文档使用不同的投影路径。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — 双编码器论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 三模统一模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 生产级后期交互。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — 实时排行榜。