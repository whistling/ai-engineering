# 主题建模 — LDA 与 BERTopic

> LDA：文档是主题的混合，主题是单词的分布。BERTopic：文档在嵌入空间中聚类，聚类即主题。目标相同，但分解方式不同。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)  
**Time:** ~45 分钟

## 问题

你有 10,000 条客户支持工单、50,000 篇新闻文章或 200,000 条推文。你需要在不通读的情况下了解集合的主题。你没有标注类别，甚至不知道存在多少类别。

主题建模在无监督下回答这个问题。给它一个语料库，返回一小组连贯的主题，并为每个文档给出这些主题的分布。

两类算法占主导地位。LDA（2003）将每个文档视为潜在主题的混合，每个主题是单词上的分布。推断是贝叶斯的。当你需要混合成员主题分配和可解释的词级概率分布时，LDA 仍然在生产环境中使用。

BERTopic（2020）用 BERT 对文档进行编码，用 UMAP 降维，用 HDBSCAN 聚类，并通过基于类别的 TF-IDF 提取主题词。它在短文本、社交媒体以及语义相似性比词汇重叠更重要的场景中表现更好。一个文档分配一个主题，这对于长文本内容是一个限制。

本课旨在建立对两者的直观理解，并指出在给定语料上应选择哪种方法。

## 概念

![LDA 混合模型 vs BERTopic 聚类](../assets/topic-modeling.svg)

**LDA 的生成故事。** 每个主题是单词的分布。每个文档是主题的混合。要在文档中生成一个单词，先从文档的主题混合中采样一个主题，然后从该主题的单词分布中采样一个单词。推断则是反过来：给定观测到的单词，推断每个文档的主题分布和每个主题的单词分布。常用的算法有折叠式 Gibbs 抽样或变分贝叶斯。

LDA 的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行和为 1（文档的主题混合）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行和为 1（主题的单词分布）。

**BERTopic 流水线。**

1. 使用句子变换器（例如 `all-MiniLM-L6-v2`）对每个文档进行编码，得到 384 维向量。
2. 用 UMAP 将维度降到 ~5 维。BERT 嵌入维度太高，不适合直接聚类。
3. 用 HDBSCAN 聚类。基于密度，会产生不同大小的簇并存在一个“离群”标签。
4. 对每个簇，基于该簇的文档计算基于类别的 TF-IDF，以提取主题词。

输出是每个文档一个主题（外加 -1 的离群标签）。可选地，HDBSCAN 的概率向量可用于软成员关系。

## 构建

### 第 1 步：通过 scikit-learn 实现 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：去掉了停用词，`min_df` 和 `max_df` 用于过滤稀有和普遍出现的词。使用 CountVectorizer（而不是 TfidfVectorizer）因为 LDA 期望的是原始计数。

### 第 2 步：BERTopic（生产）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

对 `Topic != -1` 的筛选会去掉 BERTopic 的离群桶（HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小簇大小；BERTopic 库默认是 10。本例为本课程规模显式设置为 15。对于超过 10,000 文档的语料，建议增大到 50 或 100。

### 第 3 步：评估

两种方法都会输出主题词。问题是这些词是否连贯。

- **主题一致性（c_v）。** 结合了顶词对在滑动窗口上下文中的 NPMI（归一化点互信息），将分数聚合为主题向量，再通过余弦相似度比较。值越高越好。使用 `gensim.models.CoherenceModel`，参数 `coherence="c_v"`。
- **主题多样性。** 所有主题顶词中唯一词的比例。值越高越好（表示主题不重叠）。
- **定性检查。** 阅读每个主题的顶词。它们是否能命名一个真实的事物？人工判断仍是最后的防线。

## 何时选择哪种方法

| Situation | Pick |
|-----------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 长文档且具有主题混合 | LDA |
| 无 GPU / 计算资源有限 | LDA 或 NMF |
| 需要文档级的多主题分布 | LDA |
| 与 LLM 集成以自动标签主题 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 追求最高语义连贯性 | BERTopic |

实际最大的考虑是文档长度。BERT 嵌入会被截断；LDA 的计数方法在任意长度上都有效。对于超过嵌入模型上下文窗口的文档，需采用切分+聚合或使用 LDA。

## 使用栈（2026）

- **BERTopic。** 短文本及任何语义重要的场景的默认选择。
- **`gensim.models.LdaModel`。** 经典的 LDA，适合生产，成熟且经受过考验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 便于实验的 LDA。
- **NMF。** 非负矩阵分解。对短文本是 LDA 的快速替代，质量相当。
- **Top2Vec。** 设计类似于 BERTopic。社区较小但在某些基准上效果良好。
- **FASTopic。** 更新、更快，在非常大规模语料上优于 BERTopic。
- **基于 LLM 的标签化。** 先运行任何聚类算法，然后用大模型为每个簇命名。

## 部署建议（Ship It）

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习

1. 简单：在 20 Newsgroups 数据集上用 5 个主题拟合 LDA。打印每个主题的前 10 个词。手动为每个主题命名。算法找到了真实的类别吗？
2. 中等：在相同的 20 Newsgroups 子集上拟合 BERTopic。比较找到的主题数量、顶词和相对于 LDA 的定性连贯性。哪个更清晰地反映了真实类别？
3. 困难：对你的语料计算 LDA 和 BERTopic 的 c_v 一致性。分别用 5、10、20、50 个主题运行。绘制一致性随主题数变化的曲线。报告哪种方法在不同主题数下更稳定。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Topic | A thing the corpus is about | 在语料上关于某事物（LDA 中是词的概率分布；BERTopic 中是相似文档的簇） |
| Mixed membership | Doc is multiple topics | LDA 给每个文档分配一个关于所有主题的分布 |
| UMAP | Dimensionality reduction | 保留局部结构的流形学习；用于 BERTopic 的降维 |
| HDBSCAN | Density clustering | 找到可变大小的簇；为离群点生成“噪声”标签（-1） |
| c_v coherence | Topic quality metric | 在滑动窗口内顶词的平均点互信息（用于衡量主题质量） |

## 进一步阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 等一致性度量的论文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 生产参考。包含优秀示例。