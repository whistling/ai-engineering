# 词嵌入 — 从头实现 Word2Vec

> 一个词由它的“同伴”决定。基于这个想法训练一个浅层网络，几何结构自然显现。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 分钟

## 问题

TF-IDF 知道 `dog` 和 `puppy` 是不同的词，但它不知道它们几乎是同义的。用 `dog` 训练的分类器不能泛化到有关 `puppy` 的评论。你可以通过列出同义词来掩盖这个问题，但这会在罕见术语、领域行话以及你未预见到的每种语言上失败。

你希望有一种表示，使得 `dog` 和 `puppy` 在空间中彼此靠近；使得 `king - man + woman` 落在 `queen` 附近；使得一个在 `dog` 上训练的模型能“免费”把一些信号转移到 `puppy`。

Word2Vec 给了我们这样的空间。两层神经网络，万亿级别的训练语料，2013 年发表。架构几乎令人尴尬地简单，但结果塑造了十年的 NLP 发展。

## 概念

**分布式假设**（Firth, 1957）：“可以通过一个词所处的环境来认识这个词。”如果两个词出现在相似的上下文中，它们很可能意思相近。

Word2Vec 有两种变体，都利用了这个思想。

- **Skip-gram。** 给定中心词，预测周围的词。窗口大小为 2 时 `cat -> (the, sat, on)`。
- **CBOW（continuous bag of words）。** 给定周围词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练更慢但对罕见词更友好，因此成为默认选择。

网络有一层隐藏层且不使用非线性。输入是词表上的 one-hot 向量。输出是词表上的 softmax。训练后，你丢弃输出层。隐藏层的权重就是嵌入。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

关键技巧：对 10 万词做 softmax 是昂贵到无法接受的。Word2Vec 使用 **负采样** 将其转为二分类任务。预测“这个上下文词是否出现在这个中心词附近，是或否”。对于每个训练对，从词表中采样少量的负样本（未共现的词），而不是对整个词表计算 softmax。

```figure
word-vector-arithmetic
```

## 实现它

### 步骤 1：从语料构建训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内的每个 (center, context) 对都是一个正样本训练例。

### 步骤 2：嵌入表

两个矩阵。`W` 是中心词的嵌入表（训练后保留）。`W'` 是上下文词的表（常被丢弃，有时会与 `W` 平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小幅随机初始化。词表大小 10k、维度 100 是现实可用的；教学演示可以用 50 个词汇、16 维来观察几何效果。

### 步骤 3：负采样目标

对于每个正样本 `(center, context)`，从词表中采样 `k` 个随机词作为负样本。训练目标是使得点积 `W[center] · W'[context]` 在正样本上很大，在负样本上很小。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

魔法公式：对正样本使用逻辑损失（希望 sigmoid 接近 1），对负样本使用逻辑损失（希望 sigmoid 接近 0）。梯度会流向两个嵌入表。完整推导见原始论文；如果想要记牢可以拿铅笔和纸推导一遍。

### 步骤 4：在玩具语料上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大语料、足够的 epoch 后，共享上下文的词会得到相似的中心嵌入。在玩具语料上你能隐约看到效果；在数十亿标记上，你会明显看到。

### 步骤 5：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300 维 Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。这并不是因为模型知道什么是皇室，而是向量 `(king - man)` 捕捉到类似“皇室”的概念，把它加到 `woman` 会落在“女性皇室”区域附近。

## 使用它

手写 Word2Vec 有教学意义。生产环境通常使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

实际工作中，你几乎不会自己从头训练 Word2Vec。通常直接下载预训练向量。

- **GloVe** — 斯坦福的共现矩阵分解方法。提供 50d、100d、200d、300d 的检查点，覆盖面广。Lesson 04 专门介绍 GloVe。
- **fastText** — Facebook 对 Word2Vec 的扩展，嵌入字符 n-gram。通过子词组合处理 OOV。Lesson 04。
- **Pretrained Word2Vec on Google News** — 300d、3M 词汇表，2013 年发布，至今仍被频繁下载。

### 在 2026 年 Word2Vec 仍然有用的场景

- 轻量级、领域特定的检索。在笔记本上用一小时训练医学摘要，得到通用模型捕捉不到的专业向量。
- 类比式特征工程。比如 `gender_vector = mean(man - woman pairs)`，从其他词中减去它以得到性别中性轴。在公平性研究中仍在使用。
- 可解释性。100 维足够小，可以用 PCA 或 t-SNE 可视化并真实看到簇的形成。
- 任何需要在无 GPU 的设备上运行的推理场景。Word2Vec 的查找仅是一次矩阵行的抓取。

### Word2Vec 的局限

多义性问题（polysemy wall）。`bank` 只有一个向量，无法区分 `river bank` 和 `financial bank`。`table`（电子表格 vs 家具）同理。下游分类器无法从静态向量中辨别词义。

上下文嵌入（ELMo、BERT 以及此后所有 Transformer）通过基于上下文为词的每次出现生成不同的向量来解决了这个问题。这是从 Word2Vec 到 BERT 的飞跃：从静态到上下文化。Phase 7 覆盖 Transformer 部分。

另一个失败是词表外问题（OOV）。如果训练语料中没有见过 `Zoomer-approved`，Word2Vec 无法回退。fastText 用子词组合修复了这一点（见 Lesson 04）。

## 上线部署

将以下内容保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习

1. **简单。** 在一个小语料上（20 条关于猫和狗的句子）运行训练循环。训练 200 个 epoch 后，验证 `nearest(vocab, W, W[vocab["cat"]])` 是否在前 3 名中返回 `dog`。如果没有，增加 epoch 或调整词表。
2. **中等。** 添加高频词的下采样。频率高于 `10^-5` 的词以与其频率成比例的概率从训练对中丢弃。测量对罕见词相似性的影响。
3. **困难。** 在 20 Newsgroups 语料上训练模型。计算两个偏差轴：`he - she` 和 `doctor - nurse`。将职业词投影到这两个轴上，报告哪些职业的偏差差距最大。这是公平性研究者常用的一类探针。

## 关键术语

| 术语 | 常说的含义 | 实际意思 |
|------|-----------|--------|
| Word embedding | Word as a vector | 一个从上下文中学习到的致密、低维（通常 100–300 维）表示。 |
| Skip-gram | Word2Vec trick | 从中心词预测上下文词。比 CBOW 慢，但对罕见词更好。 |
| Negative sampling | Training shortcut | 用与 `k` 个随机词的二分类替代对全词表的 softmax。 |
| Static embedding | One vector per word | 每个词只有一个向量，与上下文无关。对多义词无能为力。 |
| Contextual embedding | Context-sensitive vector | 基于周围词为每次出现生成不同向量。Transformer 等模型产生的就是这种向量。 |
| OOV | Out of vocabulary | 训练时未见过的词。Word2Vec 无法为其生成向量。 |

## 扩展阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 负采样论文，短小且易读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 如果原论文的数学推导感觉晦涩，这是对梯度推导最清晰的解释。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 实用的生产级训练设置。