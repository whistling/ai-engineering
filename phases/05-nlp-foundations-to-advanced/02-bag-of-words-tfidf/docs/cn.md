# 词袋、TF-IDF 与文本表示

> 先计数，后思考。到 2026 年，在定义明确的任务上，TF-IDF 仍然胜过嵌入。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 5 · 01（文本处理）, Phase 2 · 02（从零实现线性回归）
**Time:** ~75 分钟

## 问题

模型需要数字。你有字符串。

每个 NLP 流水线都要回答同一个问题：如何把可变长度的标记流转换成分类器可以消费的固定大小向量。该领域最先落地的答案是最简单可行的：统计词频。做成向量。

这个向量承载了比任何嵌入模型更多的生产级 NLP 应用。垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25 出现之前）、第一波情感分析、学术 NLP 基准的第一个十年。到 2026 年，实践者在窄分类任务上仍然首先尝试它。它速度快、可解释，并且在以词的存在与否为关键信号的任务上，往往难以区分与一个 4 亿参数嵌入模型的差异。

本课从头构建词袋，然后是 TF-IDF。之后展示用 scikit-learn 三行代码完成相同工作。最后指出让你转向嵌入的失败模式。

## 概念

**词袋（Bag of Words, BoW）** 丢弃顺序。对每个文档，统计词表中每个词出现的次数。向量长度等于词表大小。位置 `i` 是词 `i` 的计数。

**TF-IDF** 对 BoW 重新加权。出现在所有文档中的词没有信息量，因此把权重压低。跨语料库罕见但在单个文档中频繁出现的词是有信号的，因此把权重放大。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是文档中的词频，`df` 是文档频率（含该词的文档数量），`N` 是文档总数。`log` 保证对普遍词的权重有界。

关键属性：两者都产生稀疏向量且轴可解释。你可以查看训练好分类器的系数，读出哪些词把文档推向某个类别。你无法对 768 维的 BERT 嵌入做同样的直观解释。

```figure
bow-tfidf
```

## 实现

### 步骤 1：构建词表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何基于词的分词器都可以；本课的 `code/main.py` 使用简化的小写变体）。输出：`{word: index}` 字典。稳定的插入顺序意味着词索引 0 是在第一个文档中看到的第一个词。约定各异；scikit-learn 会按字母排序。

### 步骤 2：词袋

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行代表文档。列代表词表索引。条目 `[i][j]` 表示“词 `j` 在文档 `i` 中出现了多少次”。文档 1 中 `cat` 出现两次，因为确实如此。文档 0 中 `ran` 出现 0 次，因为没有。

### 步骤 3：词频与文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个值得说明的平滑技巧。使用 `(n+1)/(d+1)` 避免 `log(x/0)`。末尾的 `+1` 保证即使一个词出现在所有文档中，其 IDF 也为 1（而不是 0），这与 scikit-learn 的默认行为一致。其他实现使用原始的 `log(N/df)`。两者都可行；平滑版本更友好。

### 步骤 4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三篇文档，五个词表词（`the`, `cat`, `sat`, `dog`, `ran`）。`the` 出现在所有三篇文档中，所以它的 IDF 很低。`dog` 只在一篇出现，因此 IDF 很高。向量是稀疏的（大多数项很小），判别性词会脱颖而出。

### 步骤 5：L2 归一化行向量

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

不做归一化时，较长的文档会得到更大的向量并主导相似度计算。L2 归一化把每个文档放到单位超球面上。此时行之间的余弦相似度就是点积。

## 使用

scikit-learn 提供了生产级实现。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 在一次调用中完成分词、词表和 BoW。`TfidfVectorizer` 在此基础上添加了 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。对于 10 万篇文档，稠密表示无法放入内存；在分类器需要稠密前保持稀疏。

会改变一切的重要参数：

| Arg | Effect |
|-----|--------|
| `ngram_range=(1, 2)` | 包含二元语法（bigrams）。通常能提升分类效果。 |
| `min_df=2` | 丢弃少于 2 篇文档中出现的词。在噪声数据上裁剪词表。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词。近似停用词移除而不需要硬编码列表。 |
| `stop_words="english"` | scikit-learn 内置的停用词表。依赖任务 —— 情感分析不应删除否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 代替原始 `tf`。在某个词在同一文档中重复很多次时有帮助。 |

### 在哪些场景下 TF-IDF 仍然胜出（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。词的存在即关键信号，语义细微差别并不重要。
- 低数据情形（数百个标注样本）。TF-IDF 加上逻辑回归没有预训练成本。
- 对延迟有严格要求的场景。TF-IDF 加线性模型能在微秒级返回结果。通过 transformer 计算文档嵌入需要 10–100ms。
- 需要解释性的时候。检查分类器系数，正向最高的词通常就是预测原因。

### TF-IDF 的失效场景

语义盲区失效。考虑下面两条文档：

- “The movie was not good at all.”
- “The movie was excellent.”

一条是负评，一条是正评。它们的 TF-IDF 重叠刚好是 `{the, movie, was}`。词袋分类器必须记住 `not` 在 `good` 附近会翻转标签。它可以在足够数据上学到，但永远不如理解句法的模型自然。

另一个失败是推断时的词表外（OOV）词。一个在 IMDb 评论上训练的 BoW 模型对从未出现过的 `Zoomer-approved` 毫无头绪。子词嵌入（课 04）可以处理这个问题。TF-IDF 无法。

### 混合：TF-IDF 加权嵌入

到 2026 年，对于中等数据量的分类任务，务实的默认是：用 TF-IDF 权重对词嵌入做注意力加权。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你通过嵌入获得语义能力，通过 TF-IDF 获得对稀有词的强调。分类器在池化向量上训练。在情感、主题和意图分类中，这通常优于单独使用任一方法，尤其是在标注样本数低于约 5 万时。

## 部署

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: 给定一个文本分类任务，推荐 BoW、TF-IDF、嵌入，或混合方法。
phase: 5
lesson: 02
---

你需要推荐一种文本向量化策略。给定任务描述，输出：

1. 表示方法（BoW、TF-IDF、transformer 嵌入，或混合）。用一句话说明理由。
2. 具体的 vectorizer 配置。指明库名称。引用参数（`ngram_range`、`min_df`、`max_df`、`sublinear_tf`、`stop_words`）。
3. 在上线前需要测试的一个失败模式。

当用户标注样本少于 500 个时，除非用户提供证据表明 TF-IDF 基线存在语义性失败，否则拒绝推荐嵌入。在情感分析中拒绝删除停用词（否定词携带信号）。标注类别不平衡应被标记出来，表示这需要比仅仅改向量化器更多的措施。

示例输入: "将 3 万条客户支持工单分为 12 类。每条工单多为 2-3 句。仅英文。需要可解释性以供审计。"

示例输出：

- 表示方法：TF-IDF。3 万样本并非少样本；可解释性需求排除了稠密嵌入。
- 配置：`TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`。保留停用词，因为类别关键词有时是停用词（例如 "not working" 与 "working"）。
- 上线前需测试：验证 `min_df=3` 不会丢弃稀有类别的关键词。按类别过滤 `get_feature_names_out` 并人工检查。
```

## 练习

1. **简单。** 在 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，词表不相交的文档得分为 0.0。
2. **中等。** 为 `bag_of_words` 添加 n-gram 支持。参数 `n` 生成对 n-gram 的计数。测试在 `["the", "cat", "sat"]` 上 `n=2` 会生成 `["the cat", "cat sat"]` 的二元组计数。
3. **困难。** 使用 GloVe 100d 向量（下载一次并缓存）构建上面提到的 TF-IDF 加权嵌入混合模型。在 20 Newsgroups 数据集上，将其与纯 TF-IDF 以及纯均值池化嵌入做分类准确率比较。报告各自适合的场景与优劣。

## 术语

| 术语 | 常说法 | 实际含义 |
|------|--------|---------|
| BoW | 词频向量 | 一个文档中词表词的计数。丢弃顺序。 |
| TF | 词频（Term Frequency） | 单个文档中某词的计数，或按文档长度归一化后的计数。 |
| DF | 文档频率（Document Frequency） | 至少出现一次的文档数量。 |
| IDF | 逆文档频率（Inverse Document Frequency） | 平滑后的 `log(N / df)`。压低在所有文档中都出现的词的权重。 |
| 稀疏向量 | 大多数元素为零 | 词表通常有 1 万到 10 万个词；任一文档中大多数词均不出现。 |
| 余弦相似度 | 向量夹角 | L2 归一化向量的点积。1 表示相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 规范的 API 参考，并对每个参数有说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 使 TF-IDF 成为默认方法的一篇论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 一篇 2026 年的文章，讨论旧方法何时胜出及原因。