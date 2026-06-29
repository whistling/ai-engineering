# 朴素贝叶斯

> “朴素”的假设是错误的，但它仍然有效。这就是它的美妙之处。

**Type:** 构建
**Language:** Python
**Prerequisites:** 第2阶段，课程 01-07（分类，贝叶斯定理）
**Time:** ~75 分钟

## 学习目标

- 从零实现带拉普拉斯平滑的 Multinomial Naive Bayes（多项式朴素贝叶斯）用于文本分类
- 解释为何朴素的独立性假设在数学上是错误的但在实践中能产生正确的类别排名
- 比较 Multinomial、Bernoulli 和 Gaussian 朴素贝叶斯变体并为给定特征类型选择合适的变体
- 在高维稀疏数据上将朴素贝叶斯与逻辑回归比较，并解释其中的偏差-方差权衡

## 问题描述

你需要对文本做分类。将邮件分类为垃圾邮件或非垃圾邮件。将客户评论分类为正面或负面。将支持工单分类到不同类别。你有成千上万的特征（每个词一个特征）但训练数据有限。

大多数分类器在这种情况下会崩溃。逻辑回归需要足够样本来可靠地估计成千上万的权重。决策树一次只按一个词拆分，容易严重过拟合。在 10,000 维空间中，KNN 毫无意义，因为每个点与其他点的距离几乎相同。

朴素贝叶斯可以处理这些情形。它做了一个数学上错误的假设（在已知类别的条件下，每个特征彼此独立），但在文本分类上仍然优于“更聪明”的模型，尤其是在训练集较小时。它只需对数据进行一次扫描即可训练。它能扩展到数百万个特征。它会产生概率估计（尽管由于独立性假设，概率通常校准不好）。

理解为什么一个错误的假设能产生良好的预测，可以教会你机器学习中的一个基础：最好的模型不是最“正确”的那个，而是对你数据在偏差-方差权衡上最合适的那个。

## 概念

### 贝叶斯定理（快速回顾）

贝叶斯定理翻转条件概率：

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们想要的是 `P(class | features)` —— 在已知文档中的词时该文档属于某个类别的概率。我们可以从以下几项计算得到：
- `P(features | class)` —— 在该类别的文档中看到这些词的可能性（似然）
- `P(class)` —— 类别的先验概率（例如垃圾邮件一般有多常见？）
- `P(features)` —— 证据，对所有类别相同，因此在比较时可以忽略

具有最高 `P(class | features)` 的类别获胜。

### 朴素独立性假设

精确地计算 `P(features | class)` 需要估计所有特征的联合概率。若词汇表包含 10,000 个词，你需要估计 2^10,000 个可能组合的分布——不可能。

朴素假设：在已知类别的条件下，每个特征相互独立。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

你不再估计一个不可能的联合分布，而是估计 n 个简单的逐特征分布。每个分布只需要一个计数。

显然这个假设是错误的。“machine”和“learning”在任何文档中都不是独立的。但分类器不需要正确的概率估计。它需要正确的排名——哪个类别的概率最高。独立性假设引入系统性的误差，但这些误差对所有类别的影响类似，因此排名仍然往往是正确的。

### 为什么它仍然有效

三点原因：

1. 排名优先于校准。分类只需要将排名最高的类别预测正确。即使 P(spam) = 0.99999（而真实概率为 0.7），分类器仍然会正确选择垃圾邮件。我们不需要精确的概率，只需要正确的赢家。

2. 高偏差，低方差。独立性假设是一种强先验。它强烈约束模型，防止过拟合。在训练数据有限时，稍有偏差但稳定的模型会胜过理论上更正确但极不稳定的模型。这就是偏差-方差权衡的体现。

3. 特征冗余会相互抵消。相关特征提供冗余证据。分类器会对这种证据重复计数，但它对正确类别也是重复计数。如果“machine”和“learning”总是一起出现，它们都会为“科技”类别提供证据。朴素贝叶斯把它们计为两次，但都是为正确的类别计两次。

第四个更实用的原因：朴素贝叶斯非常快。训练只需一次数据计数。预测是一次矩阵乘法。你可以在几秒钟内在一百万个文档上训练好模型。这种速度意味着你可以更快地迭代，尝试更多特征集，并比更慢的模型做更多实验。

### 数学逐步推导

让我们通过一个具体例子来跟踪。假设我们有两个类别：spam（垃圾）和 not-spam（非垃圾）。词汇表有三个词：“free”、“money”、“meeting”。

训练数据：
- 垃圾邮件中 “free” 出现 80 次，“money” 出现 60 次，“meeting” 出现 10 次（总词数 150）
- 非垃圾邮件中 “free” 出现 5 次，“money” 出现 10 次，“meeting” 出现 100 次（总词数 115）
- 40% 的邮件是垃圾邮件，60% 不是

使用拉普拉斯平滑（alpha=1）：

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含：“free”（出现 2 次）、“money”（出现 1 次）、“meeting”（出现 0 次）。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

垃圾邮件以很大优势获胜。“free” 出现两次是垃圾邮件的强烈证据。注意在 Multinomial NB 中，未出现的词对两者的对数和没影响（0 * log(P)）。是 Bernoulli NB 明确建模词的缺失。

### 三种变体

朴素贝叶斯有三种常见变体。每种变体对 `P(feature | class)` 的建模方式不同。

#### Multinomial Naive Bayes

将每个特征建模为计数。最适合用于以词频或 TF-IDF 为特征的文本数据。

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` 是拉普拉斯平滑（下面会解释）。这个变体是文本分类的主力。

#### Gaussian Naive Bayes

将每个特征建模为正态分布。最适合连续特征。

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别对每个特征都有自己的均值和方差。当特征在每个类别内确实呈钟形分布时，这个方法效果很好。

#### Bernoulli Naive Bayes

将每个特征建模为二元（存在或不存在）。最适合短文本或二进制特征向量。

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

与 Multinomial 不同，Bernoulli 明确惩罚词的缺失。如果 “free” 通常出现在垃圾邮件中但在这封邮件缺失，Bernoulli 会将其计为反对垃圾邮件的证据。

### 何时使用每个变体

| Variant | Feature Type | Best For | Example |
|---------|--------------|----------|---------|
| Multinomial | 计数或频率 | 文本分类，词袋模型 | 邮件垃圾分类、主题分类 |
| Gaussian | 连续值 | 具有近似正态分布的表格数据 | 鸢尾花分类、传感器数据 |
| Bernoulli | 二元 (0/1) | 短文本、二元特征向量 | 短信垃圾分类、存在/缺失特征 |

### 拉普拉斯平滑

当测试数据中的某个词在某个类的训练集中从未出现时，会发生什么？

无平滑时：`P(word | class) = 0/N = 0`。一个零值乘进整个乘积会使 `P(class | features) = 0`，不顾其他所有证据。一个未见过的词会毁掉整个预测，不管其他证据有多强。

拉普拉斯平滑会给每个特征计数加上一个小的常数 `alpha`（通常为 1）：

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

当 alpha=1 时，每个词至少有一个极小的概率。像 “discombobulate” 这样的词出现在测试邮件中不再会完全杀掉垃圾邮件的概率。平滑可以从贝叶斯角度解释为在词分布上放置一个均匀的 Dirichlet 先验。

更高的 alpha 意味着更强的平滑（使分布更均匀）。更低的 alpha 意味着模型更信任数据。alpha 是一个需要调优的超参数。

alpha 的效果：

| Alpha | Effect | When to use |
|-------|--------|-------------|
| 0.001 | 几乎无平滑，更信任数据 | 训练集非常大且不期望出现未见特征 |
| 0.1   | 轻度平滑 | 大训练集 |
| 1.0   | 标准拉普拉斯平滑 | 默认起点 |
| 10.0  | 强平滑，使分布更平坦 | 训练集非常小且预计会有许多未见特征 |

### 对数空间计算

相乘数百个小于 1 的概率会导致浮点下溢。乘积在浮点中变为零，即便真实值是一个很小的正数。

解决方法：在对数空间中工作。不要相乘概率，而是相加它们的对数：

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这把预测转化为点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是朴素贝叶斯预测如此快速的原因——它与单层线性模型做的操作相同。

### 朴素贝叶斯 vs 逻辑回归

对于文本，两者都是线性分类器。区别在于建模对象不同。

| Aspect | Naive Bayes | Logistic Regression |
|--------|-------------|---------------------|
| Type | 生成式（建模 P(X\|Y)） | 判别式（建模 P(Y\|X)） |
| Training | 计数频率 | 优化损失函数 |
| Small data | 更好（强先验有帮助） | 更差（样本不足难以估计权重） |
| Large data | 更差（错误假设造成损害） | 更好（决策边界更灵活） |
| Features | 假设独立 | 能处理相关性 |
| Speed | 单次遍历，非常快 | 迭代优化 |
| Calibration | 概率校准差 | 概率更好 |

经验法则：先用朴素贝叶斯。如果有足够数据且 NB 达到瓶颈，再切换到逻辑回归。

### 分类流水线

```mermaid
flowchart LR
    A[原始文本] --> B[分词]
    B --> C[构建词汇表]
    C --> D[统计词频]
    D --> E[应用平滑]
    E --> F[计算对数概率]
    F --> G[预测：argmax 给定词的类别概率]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

在实际工作中，我们在对数空间操作以避免浮点下溢。与其相乘许多小概率，不如相加它们的对数：

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## 自行实现

`code/naive_bayes.py` 中的代码从零实现了 MultinomialNB 和 GaussianNB。

### MultinomialNB

从零实现的要点：

1. **fit(X, y)**：对每个类别，统计每个特征的频率。添加拉普拉斯平滑。计算对数概率。存储类别先验（类别频率的对数）。

2. **predict_log_proba(X)**：对每个样本，计算 log P(class) + 各特征的 log P(feature_i | class) 之和。这个操作可以写成矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回对数概率最高的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键见解：训练完成后，预测仅仅是矩阵乘法加上一个偏置项。这就是朴素贝叶斯如此快速的原因。

### GaussianNB

对于连续特征，我们为每个类别的每个特征估计均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测时对每个特征使用高斯概率密度函数，并在对数空间相加（相乘在原空间）。

### 示例：文本分类

代码生成了模拟的词袋数据，模拟两类（科技文章 vs 体育文章）。每类有不同的词频分布。MultinomialNB 使用词频对它们进行分类。

合成数据的构造如下：创建 200 个“词”（特征列）。词 0-39 在科技文章中频率高，在体育中低。词 80-119 在体育中频率高，在科技中低。词 40-79 在两类中频率中等。这创造了一个现实的场景：一些词是强类别指示器，而其他是噪声。

### 示例：连续特征

代码生成类似 Iris 的数据（3 类，4 个特征，高斯簇）。GaussianNB 使用每类的均值和方差进行分类。每个类别有不同的中心（均值向量）和不同的扩散（方差），模拟在现实世界中测量值在类别间系统性差异的情形。

代码还演示了：
- **平滑比较：** 对 MultinomialNB 使用不同的 alpha 值训练，展示平滑强度对准确性的影响。
- **训练集规模实验：** 随着训练数据从 20 增加到 1600，NB 的准确率如何提高。NB 在非常少样本时就能达到不错的准确率——这是其主要优势。
- **混淆矩阵：** 每类的精确率、召回率和 F1 分数，展示 NB 在何处出错。

### 预测速度

朴素贝叶斯的预测是一次矩阵乘法。对于有 n 个样本、d 个特征和 k 个类别：
- MultinomialNB：一次矩阵乘法 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB：n * k 次高斯 PDF 计算，每次遍历 d 个特征 = O(n * d * k)

两者在每个维度上都是线性的。与之对比，KNN 需要对所有训练点计算距离，或带 RBF 核的 SVM 需要对所有支持向量评估核函数。NB 在预测时快了几个数量级。

## 使用方法

使用 sklearn，两种变体都是一行搞定：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

用于文本分类的 sklearn 示例：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中的代码将从零实现与 sklearn 的实现对比，以验证正确性。

### TF-IDF 与朴素贝叶斯

原始词频会给每次出现的词相同的权重。但像 “the” 和 “is” 这样的常见词在每个类别中都会频繁出现——它们并不携带判别信息。TF-IDF（词频-逆文档频率）会降低常见词的权重，提高罕见但具有判别性的词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值是非负的，因此可与 MultinomialNB 一起使用。TF-IDF + MultinomialNB 的组合是文本分类中最强的基线之一。在训练样本少于 10,000 的数据集中，它经常胜过更复杂的模型。

### 短文本使用 BernoulliNB

对于短文本（推文、短信、聊天消息），BernoulliNB 有时会优于 MultinomialNB。短文本的词频很低，MultinomialNB 依赖的频率信息噪声较大。BernoulliNB 只关心存在/不存在，这在短文本中更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer 的 `binary=True` 标志将所有计数转换为 0/1。若不设置，BernoulliNB 仍然能工作，但它会看到并非为其设计的计数值。

### 校准 NB 的概率

NB 的概率校准较差。当 NB 说 P(spam) = 0.95 时，真实概率可能是 0.7。如果你需要可靠的概率估计（例如为了设定阈值或与其它模型组合），使用 sklearn 的 CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

这会使用交叉验证在 NB 的原始分数上拟合一个逻辑回归。得到的概率会更接近真实类别频率。

### 常见陷阱

1. 负值特征。MultinomialNB 要求非负特征。如果你有负值（如某些设置下的 TF-IDF 或标准化后的特征），请使用 GaussianNB，或将特征平移为正值。

2. 零方差特征。GaussianNB 会除以方差。如果某个类别的某个特征方差为零（所有值相同），概率计算会出问题。代码在所有方差上加了一个很小的平滑项（1e-9）以避免这种情况。

3. 类别不平衡。如果 99% 的邮件不是垃圾邮件，先验 P(not-spam) = 0.99 会非常强，压倒似然证据。你可以手动设置类别先验，或在 sklearn 中使用 class_prior 参数。

4. 特征缩放。MultinomialNB 不需要缩放（它基于计数）。GaussianNB 也不需要缩放（它估计每特征的统计量）。这是相对于逻辑回归和 SVM 的一个优势，后者对特征尺度敏感。

## 部署

本课产出：
- `outputs/skill-naive-bayes-chooser.md` —— 一个用于选择合适 NB 变体的决策技能
- `code/naive_bayes.py` —— 从零实现的 MultinomialNB 和 GaussianNB，并与 sklearn 进行比较

### 朴素贝叶斯失败的情况

当独立性假设导致错误的排名（而不仅仅是错误的概率）时，NB 会失败。常见情形包括：

1. 强烈的特征交互。如果类别依赖于两个特征的组合而非任一单独特征（XOR 型模式），NB 将完全无法捕捉。每个特征单独都不提供证据，NB 无法以非线性方式将它们组合。

2. 高度相关但相反的证据。如果特征 A 表示“垃圾”，而特征 B 表示“非垃圾”，但 A 和 B 完全相关（它们在现实中总是同时出现并一致），NB 会看到冲突的证据，而实际上不存在冲突。

3. 非常大的训练集。数据足够多时，判别式模型如逻辑回归可以学习到真实的决策边界并超越 NB。此前帮助小数据的独立性假设现在反而是桎梏。

在实践中，这些失败模式在文本分类中较为罕见。文本特征多而单个弱，独立性假设的误差往往能相互抵消。对于表格数据且特征较少且高度相关的情况，优先考虑逻辑回归或基于树的模型。

## 练习

1. 平滑实验。对文本数据使用 alpha 值 0.01、0.1、1.0、10.0 和 100.0 训练 MultinomialNB。绘制准确率随 alpha 的变化曲线。在哪个 alpha 值处性能达到峰值？为什么过高的 alpha 会降低性能？

2. 特征独立性测试。取一个真实文本数据集。选择两个明显相关的词（例如 “machine” 和 “learning”）。计算 P(word1 | class) * P(word2 | class) 并与 P(word1 AND word2 | class) 比较。独立性假设错误得有多离谱？这是否影响分类准确率？

3. 实现 Bernoulli。扩展代码实现一个 BernoulliNB 类。将词袋转换为二值（存在/不存在）并在文本数据上与 MultinomialNB 比较准确率。什么时候 Bernoulli 胜出？

4. NB vs 逻辑回归。在文本数据上训练二者。从 100 个训练样本开始增加到 10,000 个。绘制两者随训练集大小的准确率曲线。逻辑回归在何时超过朴素贝叶斯？

5. 垃圾邮件过滤器。构建一个完整的垃圾邮件分类器：对原始邮件文本进行分词，构建词汇表，创建词袋特征，训练 MultinomialNB，用精确率和召回率评估（不仅仅是准确率——为什么？）。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Naive Bayes | "Simple probabilistic classifier" | 一种在已知类别条件下对特征做条件独立性假设并应用贝叶斯定理的分类器 |
| Conditional independence | "Features don't affect each other" | P(A, B \| C) = P(A \| C) * P(B \| C) —— 在已知 C 时，知道 B 不会提供关于 A 的额外信息 |
| Laplace smoothing | "Add-one smoothing" | 给每个特征加上一个小计数以防止零概率主导预测 |
| Prior | "What you believed before seeing data" | P(class) —— 在观察任何特征之前对每个类别的先验概率 |
| Likelihood | "How well the data fits" | P(features \| class) —— 在已知类别的条件下观察到这些特征的概率 |
| Posterior | "What you believe after seeing data" | P(class \| features) —— 观察特征后对类别的更新概率 |
| Generative model | "Models how data is generated" | 学习 P(X \| Y) 和 P(Y)，然后用贝叶斯定理得到 P(Y \| X) 的模型 |
| Discriminative model | "Models the decision boundary" | 直接学习 P(Y \| X) 而不建模 X 的生成方式 |
| Log probability | "Avoid underflow" | 使用 log P 代替 P，以防许多小数相乘在浮点中变为零 |

## 深入阅读

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) -- 三种变体及数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- 多项式与伯努利文本模型的经典比较
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- 针对文本的 NB 改进
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- 证明了 NB 在样本较少时比 LR 收敛得更快