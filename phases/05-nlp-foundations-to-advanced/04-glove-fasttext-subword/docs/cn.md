# GloVe, FastText, 和 子词嵌入

> Word2Vec 为每个词训练一个嵌入。GloVe 对共现矩阵做了分解。FastText 嵌入了片段。BPE 把这一切与 Transformer 连接起来。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 5 · 03（从零实现 Word2Vec）  
**Time:** ~45 分钟

## 问题

Word2Vec 留下了两个未解问题。

首先，有另一条并行研究路线直接对共现矩阵做分解（LSA、HAL），而不是进行在线的 skip-gram 更新。Word2Vec 的迭代方法真的是根本更好，还是两种方法在处理计数时的差异导致了不同？**GloVe** 给出了答案：用精心设计的损失函数对矩阵分解进行训练，其效果可与或优于 Word2Vec，而且训练成本更低。

其次，这两种方法都没有处理未见过词的方案。`Zoomer-approved`、`dogecoin`、上周才出现的专有名词、以及稀有词根的所有词形变化。**FastText** 通过嵌入字符 n-gram 解决了这个问题：一个词等于其部分（包括词素）的和，因此即使是词表外词也能得到合理的向量表示。

第三，一旦 Transformer 出现，问题又发生了转移。词级词汇表通常上线在约一百万条目；真实语言比这更开放。**字节对编码（BPE）** 及其变体通过学习高频子词单元的词表解决了这个问题，从而覆盖所有情况。每一个现代 LLM 的分词器都是一种子词分词器。

本课将介绍三者的原理与实现，并解释在何种情形下选择哪一种。

## 概念

**GloVe（Global Vectors）。** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 表示词 j 在词 i 的上下文中出现的频率。训练向量使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失做加权以避免高频对主导优化。就这样。

**FastText。** 一个词由其字符 n-gram 加上整个词本身的向量之和表示。`where` 变为 `<wh, whe, her, ere, re>, <where>`。词向量是这些组成向量的求和。以 Word2Vec 的方式训练。好处是：未见词（如 `whereupon`）可以由已知的 n-gram 组成并得到向量。

**BPE（Byte-Pair Encoding）。** 从单个字节（或字符）组成的词表开始。统计语料中所有相邻对的出现次数。合并出现频率最高的一对为一个新 token。重复进行 k 次。结果是一个包含 `k + 256`（若以字节为单位）个 token 的词表，高频序列（如 `ing`、`tion`、`the`）成为单个 token，罕见词则被分解为熟悉的片段。任何句子都能被分词为某种已知 token 序列。

## 实现

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两点值得说明。加权函数 `f(x) = (x/x_max)^alpha` 会对极高频的词对（例如 `(the, and)`）进行降权，以免它们主导损失。最终嵌入是 `W`（中心）与 `W_tilde`（上下文）表的和。将两者相加是一个发表过的技巧，通常优于仅使用其中一张表。

### FastText：感知子词的嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由其 n-gram 集合表示（通常为 3 到 6 个字符）。词嵌入是其 n-gram 嵌入的求和。在 skip-gram 训练中，把这个和放到原先 Word2Vec 使用单一向量的位置即可。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未见词，只要它的一些 n-gram 已知，就仍然能得到向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere`、`<where` 等 n-gram，因此两个词会在向量空间上相互靠近。

### BPE：学习子词词表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一轮会合并最常见的相邻对。经过足够多的轮次，高频子串（如 `low`、`est`、`tion`）会成为单个 token，而罕见词会被干净地拆分成熟悉的部分。

真实的 GPT / BERT / T5 分词器会学习 3 万到 10 万次合并。结果是：任何文本都会被分词为已知 ID 的有界长度序列，永远不会出现 OOV。

## 使用

在实际中，你很少会自己训练这些模型。通常直接加载预训练检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

对于 Transformer 时代的 BPE 风格子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀表示词边界（GPT-2 的一个约定）。每一个现代分词器都是 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 何时选择哪种方法

| Situation | Pick |
|-----------|------|
| 预训练的一般用途词向量，且不需要 OOV 容错 | GloVe 300d |
| 预训练的一般用途词向量，必须处理拼写错误 / 新词 / 形态丰富的语言 | FastText |
| 任何要进入 Transformer（训练或推理）的内容 | 使用模型自带的分词器。切勿替换。 |
| 从头训练自己的语言模型 | 先在你的语料上训练一个 BPE 或 SentencePiece 分词器 |
| 用线性模型做生产环境文本分类 | 仍然使用 TF-IDF。参见 Lesson 02。 |

## 发布

保存为 `outputs/skill-embeddings-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1. 简单。运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard 重叠率。你应该能看到大量共享片段（`pla`、`lay`、`play`），这就是 FastText 在形态变体间迁移性能良好的原因。
2. 中等。扩展 `learn_bpe` 以跟踪词表增长。绘制合并次数与每个语料字符对应的 token 数（tokens-per-corpus-character）之间的关系曲线。你会看到最初快速压缩，趋近于每个 token 大约 ~2–3 个字符的水平。
3. 困难。在莎士比亚全集上训练一个 1k 合并的 BPE。比较常见词与罕见专有名词的分词效果。衡量合并前后平均每词的 token 数。写出让你惊讶的发现。

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Co-occurrence matrix | Word-word frequency table | `X[i][j]` = 词 j 在词 i 的窗口内出现的频率（共现矩阵）。 |
| Subword | Piece of a word | 一个词的片段：字符 n-gram（FastText）或学习得到的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | Byte-pair encoding | 反复合并最频繁的相邻对直到词表达到目标大小。 |
| OOV | Out of vocabulary | 模型从未见过的词。Word2Vec/GloVe 无法处理，FastText 和 BPE 能处理。 |
| Byte-level BPE | BPE on raw bytes | GPT-2 的方案。词表以 256 个字节开始，所以永远不会出现 OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，七页，仍然是损失推导方面最好的资料。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 将 BPE 引入现代 NLP 的工作。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — 关于 BPE、WordPiece 和 SentencePiece 在实践中实际差异的说明。