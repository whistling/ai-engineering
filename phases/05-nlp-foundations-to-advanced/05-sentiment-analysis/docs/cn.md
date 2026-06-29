# 情感分析

> 经典的 NLP 任务。大多数关于经典文本分类需要了解的内容都会在这里出现。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** 阶段 5 · 02 (BoW + TF-IDF), 阶段 2 · 14 (Naive Bayes)  
**Time:** ~75 分钟

## 问题描述

“The food was not great.” 是正面还是负面？

情感看起来很简单。评论者喜欢或不喜欢某物，对句子打标签。但之所以成为经典任务，是因为每个看似简单的例子背后都隐藏着难点。否定会翻转含义。讽刺会颠倒含义。尽管包含两个负面编码词，"Not bad at all" 实际上是正面的。表情符号往往比周围文本携带更多信息。领域词汇很重要（音乐评论中 `tight` 与时尚评论中 `tight` 的含义不同）。

情感是经典 NLP 的试验田。如果你理解了为什么每个天真的基线都有特定的失败模式，你就能理解为什么会发明更复杂的模型。本课从头构建 Naive Bayes 基线，加入逻辑回归，并指出那些使生产级情感成为合规问题的陷阱。

## 概念

经典情感分析是两步配方：

1. 表示（Represent）。把文本变为特征向量。BoW、TF-IDF 或 n-gram。
2. 分类（Classify）。在有标签的示例上拟合线性模型（Naive Bayes、logistic regression、SVM）。

Naive Bayes 是能工作的最“愚蠢”的模型。假设在给定标签的条件下，每个特征相互独立。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。推理时把概率相乘。这个“天真”的独立性假设荒谬地错误，然而结果却出乎意料地强。原因是：在稀疏文本特征和适量数据下，分类器更关心每个词更偏向哪一边，而不是偏向的程度。

Logistic regression 修复了独立性假设。它为每个特征学习一个权重，包括负权重。`not good` 作为一个二元组特征会得到一个负权重。Naive Bayes 对于从未见过的二元组无法做到这一点。

```figure
sentiment-logits
```

## 构建过程

### 第 1 步：一个真实的小型数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

故意很小。真实工作会使用成千上万的示例（IMDb、SST-2、Yelp polarity）。数学是相同的。

### 第 2 步：从零实现多项式 Naive Bayes

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加性平滑（alpha=1.0）即拉普拉斯平滑。没有它，在某类中未见过的词概率为零，log 会爆炸。实践中常用 `alpha=0.01`。教学默认用 `alpha=1.0`。

### 第 3 步：从零实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 正则化在这里很重要。文本特征是稀疏的；没有 L2 模型会记住训练样本。默认从 `0.01` 开始调参。

### 第 4 步：处理否定（失败模式）

考虑 `not good` 和 `not bad`。BoW 分类器看到的是 `{not, good}` 和 `{not, bad}`，并从训练中学到哪个更常见。二元组分类器看到 `not_good` 和 `not_bad`，把它们作为不同特征来学习。通常这已经足够。

在没有二元组的情况下，一个更粗糙但有效的修复方法是：否定作用域（negation scoping）。给紧随否定词之后直到下一个标点的 token 加上 `NOT_` 前缀。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同的特征。分类器可以给它们相反的权重。三行的预处理，在情感基准上能带来可测量的准确率提升。

### 第 5 步：重要的评估指标

如果类别不平衡，仅报告准确率具有误导性。真实情感语料通常 70%-80% 倾向正面或 70%-80% 倾向负面；一个恒定预测多数类的分类器可以得到 80% 的准确率，但毫无价值。请报告以下所有指标：

- 每类的精确率（precision）和召回率（recall）。每类一对。对它们做宏平均以得到一个考虑类别平衡的单一数字。
- 宏平均 F1（Macro-F1，不平衡数据的主要指标）。取每类 F1 的均值，等权重。在类别不平衡时用它替代准确率。
- 加权 F1（Weighted-F1，可选）。与宏平均相同但按类别频率加权。当类别不平衡本身具有业务含义时一并报告。
- 混淆矩阵。原始计数。在相信任何标量指标之前总要查看；它能揭示模型混淆了哪对类别。
- 每类错误样本。每类抽取 5 个错误预测并阅读它们。没有什么比阅读实际错误更有价值。

对于严重不平衡的数据（> 95-5），请报告 AUROC 和 AUPRC 而不是准确率。AUPRC 对少数类更敏感，这通常是你关心的（垃圾邮件、欺诈、罕见情感）。

常见错误：在不平衡数据上报告 micro-F1 而不是 macro-F1，会得到看起来很高的数字，因为它被多数类主导。宏平均 F1 强迫你看到少数类的表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使用方法

scikit-learn 用六行代码就能正确完成。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

注意三点。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 添加二元组，使 `not_good` 成为特征。`sublinear_tf=True` 对重复词做衰减。这三个参数是 SST-2 上将基线从 75% 提升到 85% 的差别所在。

### 何时使用 Transformer

- 讽刺检测。经典模型在这上面失败——就是这样。
- 长篇评论中情感在文档中间发生转变。
- 基于方面的情感（aspect-based sentiment）。例如 “Camera was great but battery was terrible.” 你需要把情感归因到具体方面。只有 Transformer 或结构化输出模型能够胜任。
- 非英语的低资源语言。多语 BERT 可以为你提供零样本基线。

如果你需要以上任一项，请跳到阶段 7（transformers 深入）。否则，基于 TF-IDF 加二元组并做否定处理的 Naive Bayes 或 logistic regression 是你在 2026 年的生产基线。

### 可复现性陷阱（又来了）

重新训练情感模型是常规操作。重新评估它们并非如此。论文中报告的准确率使用了特定的划分、特定的预处理和特定的分词器。如果你在没有使用完全相同流水线的情况下把新模型与基线比较，你会得到误导性的差异。总是用你的流水线重新生成基线，而不是直接使用论文的数字。

## 部署（Ship It）

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 练习

1. 简单题。把 `apply_negation` 作为 scikit-learn 流水线中的预处理步骤，并在一个小型情感数据集上衡量 F1 的变化。
2. 中等题。实现带类权重的逻辑回归（传 `class_weight="balanced"` 到 scikit-learn，或自己从梯度上推导）。在合成的 90-10 类不平衡上衡量效果。
3. 困难题。通过在情感模型残差上训练第二个分类器构建讽刺检测器。记录你的实验设置。当准确率低于随机水平时要警告读者（两类讽刺的随机水平约为 50%，且大多数首次尝试会落在那附近）。

## 术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|---------|
| Polarity | Positive or negative | 二元标签；有时扩展为中性或细粒度（5 星） |
| Aspect-based sentiment | Per-aspect polarity | 将情感归因到文本中提到的具体实体或属性 |
| Negation scoping | Reversing nearby tokens | 否定作用域：在 "not" 之后直到标点，对后续 token 加上 `NOT_` 前缀 |
| Laplace smoothing | Adding 1 to counts | 拉普拉斯平滑：计数加一，防止 Naive Bayes 中出现零概率特征 |
| L2 regularization | Shrinking weights | L2 正则化：在损失上加上 `lambda * sum(w^2)`，对稀疏文本特征至关重要 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 奠基综述。较长，但前四节涵盖了经典方法的全部内容。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 该论文表明在短文本上二元组 + Naive Bayes 很难被打败。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 的参考文档以及你会调的所有参数。