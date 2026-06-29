# Text Generation Before Transformers — N-gram Language Models

> 如果一个词是令人惊讶的，那么模型就不好。困惑度（perplexity）把惊讶度量化为一个数字。平滑确保它是有限的。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (文本处理), Phase 2 · 14 (朴素贝叶斯)
**Time:** ~45 分钟

## 问题

在 Transformer、循环神经网络（RNN）和词嵌入出现之前，语言模型通过计数某个词在前面 `n-1` 个词之后出现的频率来预测下一个词。比如统计 "the cat" → "sat" 出现 47 次，"the cat" → "jumped" 出现 12 次，"the cat" → "refrigerator" 出现 0 次。归一化后得到一个概率分布。

这就是 n-gram 语言模型。它驱动了从 1980 年到 2015 年的每一个语音识别器、拼写检查器和基于短语的机器翻译系统。当你需要低成本的设备端语言建模时，它仍然很有用。

有趣的问题是如何处理未见过的 n-gram。基于原始计数的模型会把任何未在训练中看到的 n-gram 赋予零概率，这是灾难性的，因为句子很长，几乎每个长句都会包含至少一个未见序列。五十年的平滑研究解决了这个问题。Kneser-Ney 平滑是最终成果，现代深度学习继承了它的经验传统。

## 概念

![N-gram 模型：计数、平滑、生成](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（三元组通常取 3，四元组取 4 等）。通过计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何训练中未见过的 n-gram 都被赋予概率零。对 Brown 语料库的 2007 年研究发现，即便是 4-gram 模型，也有 30% 的保留集 4-grams 在训练中未见。没有平滑你无法在任何真实文本上进行评估。

**平滑方法，按复杂度排序：**

1. **Laplace（加一）。** 在每个计数上加 1。简单，但在稀有事件上表现糟糕。
2. **Good-Turing。** 根据频率的频率（frequency-of-frequencies）把概率质量从高频事件重新分配给未见事件。
3. **插值（Interpolation）。** 用可调权重将 n-gram、(n-1)-gram 等估计组合起来。
4. **回退（Backoff）。** 如果某个 n-gram 的计数为零，就回退到 (n-1)-gram。Katz 回退对其进行归一化。
5. **绝对折扣（Absolute discounting）。** 从所有计数中减去固定折扣 `D`，并将质量重新分配给未见事件。
6. **Kneser-Ney。** 绝对折扣加上对低阶模型的巧妙选择：使用*续现概率*（一个词出现于多少不同上下文中）而不是原始频率。

Kneser-Ney 的洞见很深刻。"San Francisco" 是个常见的二元组。单词 "Francisco" 的 unigram 主要出现在 "San" 之后。朴素的绝对折扣会给予 "Francisco" 很高的 unigram 概率（因为计数高）。Kneser-Ney 注意到 "Francisco" 只出现在很少的上下文中，从而降低它的续现概率。结果：以 "Francisco" 结尾的新二元组会得到恰当的低概率。

**评估：困惑度（perplexity）。** 在保留测试集上，每词平均负对数似然的指数。越低越好。困惑度为 100 意味着模型的迷惑程度相当于在 100 个词中均匀选择。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 实现

### 步骤 1：三元组计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分词后的句子列表。输出是 n-gram 计数和上下文计数。`<s>` 和 `</s>` 是句子边界标记。

### 步骤 2：Laplace 平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

对每个计数加 1。可以平滑但会过度分配概率质量给未见事件，从而损害已见但罕见事件的概率。

### 步骤 3：Kneser-Ney（bigram，插值）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

这里有三部分移动的要点。`continuation_prob` 捕捉“这个词出现在多少不同的上下文中？”（Kneser-Ney 的创新）。`lambda_prev` 是折扣释放出的概率质量，用于回退权重。最终概率是折扣后的主项加上加权的续现项。

### 步骤 4：用采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。每个 seed 通常会产生不同输出。若想要类似束搜索的输出，可以在每一步选择最大概率项（贪心）并加入一个小的随机性（温度）调节。

### 步骤 5：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。在 Brown 语料库上，调优良好的 4-gram KN 模型困惑度大约为 140。相同测试集上的 Transformer 语言模型困惑度在 15–30。差距约为 10 倍，这也是该领域转向深度模型的原因之一。

## 使用场景

- 传统自然语言处理教学：最清晰地展示平滑、MLE（极大似然估计）和困惑度概念。
- KenLM：生产级 n-gram 库。在对延迟敏感的语音和 MT 系统中用作重评分器（rescorer）。
- 设备端自动补全：键盘中的三元组模型，仍在使用。
- 基线：在宣布你的神经 LM 表现良好之前，总是先计算一个 n-gram LM 的困惑度基线。如果你的 Transformer 没有明显优于 KN，很可能有问题。

## 发布（Ship It）

保存为 `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. 简单。用 1,000 句莎士比亚语料训练一个三元组 LM。生成 20 句。它们在局部上看起来合理但在全局上会不连贯。这是经典演示。
2. 中等。对你的 KN 模型在保留集上实现困惑度评估。与 Laplace 比较。你应该看到 KN 的困惑度降低约 30–50%。
3. 困难。构建一个三元组拼写校正器：给定一个拼写错误的词及其上下文，生成候选更正并按 LM 下的上下文概率排序。在 Birkbeck 拼写语料（公开）上评估。

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| N-gram | Word sequence | 连续 `n` 个标记的序列。 |
| Smoothing | Avoiding zeros | 重新分配概率质量，让未见事件得到非零概率。 |
| Perplexity | LM quality metric | 在保留数据上的 `exp(-平均对数概率)`。越低越好。 |
| Backoff | Fallback to shorter context | 如果三元组计数为零，使用二元组。Katz 回退对此进行了形式化。 |
| Kneser-Ney | Best smoothing for n-grams | 绝对折扣 + 将低阶模型用续现概率替代原始频率。 |
| Continuation probability | KN-specific | 基于一个词出现过多少不同上下文来衡量 `P(w)`，而不是基于原始计数。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — 关于 n-gram 语言模型和平滑的权威教材。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确立 Kneser-Ney 为最佳 n-gram 平滑方法的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 原始 Kneser-Ney 论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 高速的生产级 n-gram LM，直到 2026 年仍在延迟敏感的应用中使用。