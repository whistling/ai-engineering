# Statistics for Machine Learning

> Statistics is how you know if your model actually works or just got lucky.

**Type:** 构建
**Language:** Python
**Prerequisites:** 第1阶段，课程 06（概率与分布），07（贝叶斯定理）
**Time:** ~120 分钟

## Learning Objectives

- 从头实现描述性统计、Pearson/Spearman 相关性和协方差矩阵的计算
- 执行假设检验（t 检验、卡方检验）并正确解释 p 值和置信区间
- 使用自助抽样（bootstrap）重采样为任意指标构建置信区间，无需分布假设
- 使用效应量衡量区分统计显著性与实际意义

## The Problem

你训练了两个模型。模型 A 在测试集上的得分是 0.87。模型 B 得分 0.89。你部署了模型 B。三周后，生产指标比之前更差。发生了什么？

模型 B 并未真正优于模型 A。0.02 的差异只是噪声。你的测试集太小，或方差太高，或两者兼而有之。你把随机波动误当成了改进然后上线了。

这种情况经常发生。Kaggle 排行榜的剧烈变动。无法复现的论文。基于几百个样本就宣布赢家的 A/B 测试。根本原因总是相同的：有人跳过了统计学。

统计学给你区分信号和噪声的工具。它告诉你差异是否真实，你应该有多大置信度，以及在信赖结果之前需要多少数据。每个 ML 管道、每次模型比较、每个实验都需要统计学。没有统计学，你就是在猜测。

## The Concept

### Descriptive Statistics: Summarizing Your Data

在建模之前，你需要知道数据的样子。描述性统计把数据集压缩成少数几个捕捉其形状的数字。

**集中趋势度量** 回答“中间在哪里？”

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Mean（均值）是平衡点。Median（中位数）是中间值。当它们出现偏离时，分布是偏斜的。收入分布的均值通常远大于中位数（由于亿万富翁导致的右偏）。训练期间的损失分布常常均值远小于中位数（因为大多数样本很容易）。

**离散程度度量** 回答“数据有多分散？”

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

方差是离均差平方的平均数。标准差是方差的平方根，单位与数据相同，因此更易解释。极差对离群点敏感，几乎不会单独使用。IQR（四分位距）是中间 50% 的范围，对离群点稳健，常用于箱线图和异常点检测。

**百分位数** 将排序后的数据分成 100 份。第 25 百分位（Q1）表示 25% 的值低于该点。第 50 百分位是中位数。第 75 百分位是 Q3。

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

在 ML 中，你会关注推理延迟的百分位、预测置信度的分布以及错误分布的理解。一个平均错误低但 P99 错误很糟糕的模型，在安全关键应用场景下可能毫无用处。

**样本统计量 vs 总体统计量。** 在从样本计算方差时，应除以 (n-1) 而不是 n。这个是贝塞尔校正。它补偿了样本均值不是总体均值的事实。如果分母是 n，你会系统性地低估真实方差。使用 (n-1) 可以使估计无偏。

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

实际应用中：当 n 很大（数千样本）时，差异可以忽略；当 n 很小时（几十个样本），差异很重要。

### Correlation: How Variables Move Together

相关性衡量两个变量之间线性关系的强度和方向。

**Pearson 相关系数** 测量线性关联：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson 假设关系是线性的且两个变量大致服从正态分布。它对离群点敏感，一个极端点可以把 r 从 0.1 拉到 0.9。

**Spearman 秩相关** 测量单调关联：

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

Spearman 捕捉任何单调关系，而不仅限于线性关系。如果 y = x^3，Pearson 的 r < 1，但 Spearman 的 rho = 1。

**何时使用哪种：**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**黄金法则：** 相关不意味着因果。冰淇淋销量和溺水死亡人数相关是因为两者在夏天都上升。你模型的准确率与参数数量相关，但增加参数并不自动提高准确率（见：过拟合）。

### Covariance Matrix

两个变量之间的协方差衡量它们如何共同变化：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

对 d 个特征，协方差矩阵 C 是一个 d x d 矩阵，其中 C[i][j] = Cov(feature_i, feature_j)。对角线条目 C[i][i] 是每个特征的方差。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**与 PCA 的关系。** PCA 对协方差矩阵做特征分解。特征向量是主成分（最大方差的方向）。特征值告诉你每个成分捕获了多少方差。这正是第 10 课讲过的内容：协方差矩阵是分解的正确对象，因为它编码了数据中所有成对的线性关系。

**与相关性的关系。** 相关矩阵是标准化变量（每个除以其标准差）的协方差矩阵。相关性对协方差进行了归一化，使所有值落在 [-1, 1]。

### Hypothesis Testing

假设检验是一个在不确定性下做决策的框架。你从一个主张开始，收集数据，并判断数据是否与该主张一致。

**设置：**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p 值** 是在 H0 为真的前提下，观察到像当前数据一样极端的数据的概率。它并不是 H0 为真的概率。这是统计学中最常见的误解。

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**置信区间** 给出参数的一个合理取值范围：

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

置信区间的宽度反映精度。宽区间表示不确定性高。窄区间表示你的估计精确（但如果数据有偏，精确并不等于准确）。

### The t-test

t 检验比较均值。它有几个变体。

**单样本 t 检验：** 判断总体均值是否与假设值不同？

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**两样本 t 检验（独立样本）：** 两组均值是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

Welch's t-test 不假定方差相等。除非有充分理由认为方差相等，否则总是使用 Welch。

**配对 t 检验：** 当测量成对出现（在相同数据拆分上评估相同模型）：

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在 ML 中，配对 t 检验很常见：你在相同的 10 折交叉验证折上同时运行两个模型并逐对比较它们的得分。

### Chi-squared Test

卡方检验检查观察到的频数是否与期望频数一致。适用于分类数据。

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

举例：如果分类“正向”观测到 120，期望 100；“负向”观测到 80，期望 100，则 chi^2 = 8，在 1 自由度下 p < 0.005，差异显著。

### A/B Testing for ML Models

ML 中的 A/B 测试与网页 A/B 测试不同。模型比较有其特定挑战：

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**流程：**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Statistical Significance vs Practical Significance

一个结果可以在统计上显著但实际意义微乎其微。样本足够大时，即便微小的差别也会变为统计显著。

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**效应量** 衡量差异有多大，与样本量无关：

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

始终同时报告 p 值和效应量。p 值告诉你差异是否真实；效应量告诉你差异是否重要。

### Multiple Comparison Problem

当你测试很多假设时，某些显著性结果会是偶然。若你以 alpha = 0.05 测试 20 个假设，即使全都不真实，你也会期望有 1 个假阳性。

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni 校正：** 把 alpha 除以测试数量。

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

在 ML 中，当你在多个指标上比较模型、测试许多超参数配置或在多个数据集上评估时，这点很重要。

### Bootstrap Methods

Bootstrap（自助抽样）通过有放回地对你的数据重采样来估计统计量的抽样分布。无需关于底层分布的任何假设。

**算法：**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**自助法置信区间（百分位法）：**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**为什么自助法对 ML 很重要：**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**比较模型时用 bootstrap：**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这比配对 t 检验更稳健，因为它不做分布假设。

### Parametric vs Non-parametric Tests

**参数检验** 假设特定分布（通常为正态）：

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**非参数检验** 不做分布假设：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**何时使用非参数：**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**何时使用参数检验：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

在 ML 实验中，你通常样本数较小（5 或 10 个交叉验证折），因此像 Wilcoxon 符号秩检验这样的非参数检验常常比 t 检验更合适。

### Central Limit Theorem: Practical Implications

中心极限定理（CLT）说明样本均值的分布随着 n 增大趋近于正态分布，无论底层总体分布如何。

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**这对 ML 的意义：**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT 不做的事：**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### Common Statistical Mistakes in ML Papers

1. **在训练集上测试。** 必然导致过拟合。始终保留模型训练期间从未见过的数据用于测试。

2. **没有置信区间。** 只报告一个精确的准确率数值而不给出不确定性，会使结果不可复现且不可验证。

3. **忽视多重比较问题。** 测试 50 个配置但只报告表现最好的那个，会夸大假阳性率。

4. **混淆统计显著性和实际意义。** p 值为 0.001 的 0.01% 准确率提升并没有实际价值。

5. **在不平衡数据上使用准确率。** 在负类占 99% 的数据集上达到 99% 的准确率并不意味着模型学到了有价值的东西。应使用精确率、召回率、F1 或 AUC。

6. **有选择性地报告指标。** 只报告模型获胜的指标。诚实的评估应报告所有相关指标。

7. **在训练/测试拆分间泄露信息。** 在拆分之前做归一化，或使用未来数据预测过去数据，都会导致泄露。

8. **小测试集且没有方差估计。** 在 100 个样本上评估并宣称 2% 改进是噪声而不是信号。

9. **假设独立性而数据不独立。** 来自同一患者的医学影像、同一文档的多句子样本——组内观测是相关的。

10. **P-hacking。** 反复尝试不同的检验、子集或排除标准直到 p < 0.05。结果只是搜索的产物。

## Building It

你将实现：

1. **从头实现描述性统计**（均值、 中位数、众数、标准差、百分位、IQR）
2. **相关性函数**（Pearson 和 Spearman，以及协方差矩阵）
3. **假设检验**（单样本 t 检验、两样本 t 检验、卡方检验）
4. **自助法置信区间**（适用于任意统计量，无需假设）
5. **A/B 测试模拟器**（生成数据、检验、检视 I 型与 II 型错误）
6. **统计显著性与实际显著性演示**（展示大样本如何使一切“显著”）

全部从头实现，仅使用 `math` 和 `random`。不使用 numpy、scipy。

## Key Terms

| 术语 | 定义 |
|---|---|
| Mean | 值之和除以计数。对离群点敏感。 |
| Median | 排序后数据的中间值。对离群点稳健。 |
| Standard deviation | 方差的平方根。以原始单位衡量离散程度。 |
| Percentile | 某一百分比的数据低于该值。 |
| IQR | 四分位距。Q3 减 Q1。中间 50% 的范围。 |
| Pearson correlation | 测量两个变量之间的线性关联。范围 [-1, 1]。 |
| Spearman correlation | 使用秩测量单调关联。 |
| Covariance matrix | 所有特征两两协方差组成的矩阵。 |
| Null hypothesis | 关于无效应或无差异的默认假设。 |
| p-value | 在原假设为真时，得到像观察到的数据一样极端结果的概率。 |
| Confidence interval | 在给定置信水平下参数的合理取值范围。 |
| t-test | 检验均值是否显著不同。基于 t 分布。 |
| Chi-squared test | 检验观察频数是否与期望频数不同。 |
| Effect size | 差异的大小，与样本量无关。常用 Cohen's d。 |
| Bonferroni correction | 将显著性阈值除以测试数量以控制假阳性率。 |
| Bootstrap | 有放回重采样以估计统计量的抽样分布。 |
| Type I error | 假阳性：在 H0 为真时拒绝 H0。 |
| Type II error | 假阴性：在 H0 为假时未能拒绝 H0。 |
| Statistical power | 正确拒绝错误 H0 的概率。Power = 1 - Type II error rate。 |
| Central limit theorem | 随着样本量增加，样本均值趋向正态分布。 |
| Parametric test | 假定数据服从特定分布（通常为正态）。 |
| Non-parametric test | 不做分布假设。基于秩或符号的方法。 |