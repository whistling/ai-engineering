# 概率与分布

> 概率是 AI 用来表达不确定性的语言。

**Type:** 学习  
**Language:** Python  
**Prerequisites:** 第1阶段，第01-04课  
**Time:** ~75 分钟

## 学习目标

- 从头实现 Bernoulli、Categorical、Poisson、均匀和正态分布的 PMF 和 PDF  
- 计算期望值、方差，并使用中心极限定理解释为什么高斯分布普遍存在  
- 使用数值稳定性技巧（减去最大 logit）构建 softmax 和 log-softmax 函数  
- 从 logits 计算交叉熵损失，并将其与负对数似然联系起来

## 问题情境

分类器输出 `[0.03, 0.91, 0.06]`。语言模型从 50,000 个候选词中挑选下一个词。扩散模型通过从学习到的分布中采样来生成图像。这些都是概率在起作用。

模型做出的每一次预测都是一个概率分布。每一个损失函数衡量预测分布与真实分布之间的差距。每一次训练步长都会调整参数，使得一个分布更像另一个分布。没有概率，你无法读懂任何一篇机器学习论文、调试模型，或理解为什么训练损失会变成 NaN。

## 概念

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的一个子集。概率把事件映射为 0 到 1 之间的数值。

```
掷硬币:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

掷一枚骰子:
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了所有概率理论：
1. 对任何事件 A，P(A) >= 0  
2. P(S) = 1（总会有某个结果发生）  
3. 当 A 和 B 不可能同时发生时，P(A 或 B) = P(A) + P(B)

其余的一切（贝叶斯定理、期望、分布）都由这三条规则推导出来。

### 条件概率与独立性

P(A|B) 表示在 B 已发生的条件下 A 的概率。

```
P(A|B) = P(A and B) / P(B)

例子：一副牌
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

当知道一个事件对另一个事件没有信息时，这两个事件被称为独立：

```
独立:   P(A|B) = P(A)
等价于: P(A and B) = P(A) * P(B)
```

掷硬币是独立事件。不放回抽牌则不是独立的。

### 概率质量函数 vs 概率密度函数

离散随机变量有概率质量函数 (PMF)。每个结果都有一个可以直接读到的概率值。

```
PMF: P(X = k)

公平骰子:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量有概率密度函数 (PDF)。单点的密度不是概率。概率来自对区间上密度的积分。

```
PDF: f(x)

P(a <= X <= b) = 从 a 到 b 对 f(x) 积分

f(x) 可以大于 1（密度，不是概率）
从 -inf 到 +inf 对 f(x) 的积分 = 1
```

在 ML 中，这一区别很重要。分类输出是 PMF（离散选择）。VAE 的潜在空间使用的是 PDF（连续）。

### 常见分布

**Bernoulli（伯努利分布）：** 一次试验，两个可能结果。用于建模二分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p, 方差 = p(1-p)
```

**Categorical（多项分布）：** 一次试验，k 个可能结果。用于多分类（softmax 输出）。

```
P(X = i) = p_i,  且 sum p_i = 1
例子: P(猫) = 0.7, P(狗) = 0.2, P(鸟) = 0.1
```

**Uniform（均匀分布）：** 所有结果等概率。用于随机初始化。

```
离散: P(X = k) = 1/n, k 属于 {1, ..., n}
连续: f(x) = 1/(b-a), x 在 [a, b]
```

**Normal（正态/高斯分布）：** 钟形曲线。由均值 mu 和方差 sigma^2 参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

标准正态: mu = 0, sigma = 1
  68% 的数据在 1 sigma 内
  95% 在 2 sigma 内
  99.7% 在 3 sigma 内
```

**Poisson（泊松分布）：** 固定时间/空间内稀有事件的计数。用于建模事件速率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
均值 = lambda, 方差 = lambda
```

### 期望值与方差

期望值是加权平均结果。

```
离散:   E[X] = sum x_i * P(X = x_i)
连续:   E[X] = 对 x * f(x) dx 积分
```

方差衡量围绕均值的离散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
标准差 = sqrt(Var(X))
```

在机器学习里，期望值会出现在损失函数中（对数据分布的平均损失）。方差告诉你模型的稳定性。梯度的高方差意味着训练噪声大。

### 联合分布与边缘分布

联合分布 P(X, Y) 描述两个随机变量的联合行为。

联合 PMF 例子（X = 天气, Y = 是否带伞）:

| | Y=0 (不带伞) | Y=1 (带伞) | 边缘 P(X) |
|---|---|---|---|
| X=0 (晴天) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (下雨) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布通过对另一个变量求和得到：

```
P(X = x) = 对所有 y 求和 P(X = x, Y = y)
```

上表中的行和列合计就是边缘分布。

### 为什么正态分布到处出现

中心极限定理（CLT）：许多相互独立的随机变量的和（或平均）会收敛到正态分布，不依赖于原始分布。

```
掷 1 个骰子: 均匀分布（平坦）
2 个骰子的平均: 三角形（有峰）
30 个骰子的平均: 几乎完美的钟形曲线

这对任意起始分布都成立。
```

因此：
- 测量误差近似正态（许多小的独立来源累积）
- 神经网络权重初始化常用正态分布
- SGD 中的梯度噪声近似正态（许多样本梯度的和）
- 正态分布是给定均值和方差时的最大熵分布

### 对数概率

原始概率会引发数值问题。将许多小概率相乘很快会下溢为零。

```
P(句子) = P(词1) * P(词2) * ... * P(词_n)
         = 0.01 * 0.003 * 0.02 * ...
         -> 0.0（大约 30 项后会下溢）
```

对数概率解决了这个问题。乘法变为加法。

```
log P(句子) = log P(词1) + log P(词2) + ... + log P(词_n)
            = -4.6 + -5.8 + -3.9 + ...
            -> 有界数值（不下溢）
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率总是 <= 0（因为 0 < P <= 1）
- 值越负越不可能
- 交叉熵损失就是正确类别的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 把它们转换为有效的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

性质：
  - 输出都在 (0, 1)
  - 输出和为 1
  - 保持输入的相对顺序
  - exp() 会放大 logits 之间的差异
```

softmax 技巧：在 exponent 之前减去最大 logit 以防止溢出。

```
z = [100, 101, 102]
exp(102) = 溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  （安全）

结果相同，但不会溢出。
```

Log-softmax 将 softmax 与 log 结合以保证数值稳定性。PyTorch 在实现交叉熵时内部使用这个。

### 采样

采样就是从分布中抽取随机值。在 ML 中：
- Dropout 随机采样要置零的神经元
- 数据增强采样随机变换
- 语言模型从预测分布中采样下一个 token
- 扩散模型采样噪声并逐步去噪

从任意分布采样需要技巧，例如逆变换采样、拒绝采样或 reparameterization trick（在 VAE 中使用）。

```figure
gaussian-pdf
```

## 实作

### 步骤 1：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 步骤 2：从头实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 步骤 3：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 步骤 4：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 步骤 5：Softmax 与对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 步骤 6：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 步骤 7：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现与所有可视化位于 `code/probability.py`。

## 应用

使用 NumPy 和 SciPy，上述操作都可以一行完成：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你已经从头实现了这些。现在你知道库函数在做什么。

## 练习

1. 为指数分布实现逆变换采样（inverse transform sampling）。通过采样 10,000 个值并将直方图与真实 PDF 比较来验证结果。  
2. 为两个有偏骰子构建联合分布表。计算边缘分布并检查两个骰子是否独立。  
3. 当正确类别索引为 3 时，计算 5 类分类器在 logits `[2.0, 0.5, -1.0, 3.0, 0.1]` 下的交叉熵损失。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的结果。  
4. 编写一个函数，接收一列对数概率，返回最可能的序列、总对数概率以及等价的原始概率。用每个词概率为 0.01 的 50 词句子来测试它。

## 关键词

| 术语 | 大家怎么说 | 实际意义 |
|------|----------------|----------------------|
| Sample space | "所有可能性" | 实验所有可能结果的集合 S |
| PMF | "概率函数" | 给出每个离散结果确切概率的函数，概率和为 1 |
| PDF | "概率曲线" | 连续变量的密度函数。对区间积分得到概率 |
| Conditional probability | "有条件下的概率" | P(A\|B) = P(A and B) / P(B)。是贝叶斯思维与贝叶斯定理的基础 |
| Independence | "它们互不影响" | P(A and B) = P(A) * P(B)。知道一个事件不会改变另一个事件的概率 |
| Expected value | "平均值" | 所有结果按概率加权的和。损失函数就是一个期望值 |
| Variance | "分散程度" | 距离均值的平方的期望。高方差 = 噪声多、不稳定 |
| Normal distribution | "钟形曲线" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). 由于 CLT 经常出现 |
| Central Limit Theorem | "平均值会变成正态" | 许多独立样本的均值收敛到正态分布，与原始分布无关 |
| Joint distribution | "两个变量的联合" | P(X, Y) 描述 X 与 Y 每种组合的概率 |
| Marginal distribution | "对另一个变量求和" | P(X) = sum_y P(X, Y)。从联合分布恢复单个变量的分布 |
| Log probability | "概率的对数" | log P(x)。把乘法变为加法，防止长序列的数值下溢 |
| Softmax | "把分数变成概率" | softmax(z_i) = exp(z_i) / sum(exp(z_j))。把实值 logits 映射为合法概率分布 |
| Cross-entropy | "损失函数" | -sum(p_true * log(p_predicted)). 衡量两个分布的差异。越小越好 |
| Logits | "原始模型输出" | 在 softmax 之前的未归一化分数。名称来源于 logistic 函数 |
| Sampling | "生成随机值" | 按概率分布生成值。模型如何生成输出 |

## 进一步阅读

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 可视化证明为什么平均值会变成正态  
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 涵盖本文及更多内容的简明参考  
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为何数值稳定性重要以及如何实现它