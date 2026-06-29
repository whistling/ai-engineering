# Norms and Distances

> 你的距离函数定义了“相似”的含义。选错了，下游的一切都会出问题。

**Type:** 构建
**Language:** Python
**Prerequisites:** 第1阶段，课程 01（线性代数直觉），02（向量、矩阵与运算）
**Time:** ~90 分钟

## 学习目标

- 从零实现 L1、L2、余弦、Mahalanobis、Jaccard 和编辑距离函数
- 为给定的机器学习任务选择合适的距离度量并解释其他度量为何失败
- 将 L1 和 L2 范数与 LASSO 与 Ridge 正则化及其几何约束区域联系起来
- 演示相同数据在不同度量下产生不同的最近邻结果

## 问题描述

你有两个向量。它们可能是词向量嵌入、可能是用户画像，也可能是像素数组。你需要知道：它们有多接近？

答案完全取决于你选择的距离函数。两个数据点在一种度量下可能是最近邻，在另一种度量下却相距甚远。你的 KNN 分类器、推荐引擎、向量数据库、聚类算法、损失函数——它们都依赖于这个选择。选错了，模型会优化错误的目标。

不存在通用最佳距离。L2 适合空间数据。余弦相似度主导 NLP。Jaccard 处理集合。编辑距离处理字符串。Mahalanobis 考虑相关性。Wasserstein 衡量概率质量移动。每一种都对“相似”有不同的假设。

本课从头构建每一种主要距离函数，展示何时使用它们，并演示相同数据在不同度量下如何产生完全不同的最近邻。

## 概念

### 范数：衡量向量的大小

范数衡量向量的“大小”。任意两个向量之间的距离都可以表示为它们之差的范数：d(a, b) = ||a - b||。因此理解范数就是理解距离。

### L1 范数（曼哈顿距离）

L1 范数是所有分量绝对值的和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它被称为曼哈顿距离，因为它衡量在城市网格上只能沿坐标轴行走时的步数。不能走对角线。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

何时使用 L1：
- 高维稀疏数据（文本特征、one-hot 编码）
- 需要对异常值具有鲁棒性时（单个巨大的差异不会主导结果）
- 特征选择问题（L1 正则化会促进稀疏性）

与 L1 正则化（Lasso）的关系：在损失函数中添加 ||w||_1 会惩罚权重绝对值之和。这会把小权重压为零，从而实现自动特征选择。L1 惩罚在权重空间中产生菱形约束区域，菱形的角落位于轴上，表示某些权重为零。

与损失函数的关系：平均绝对误差（MAE）是预测与目标之间的平均 L1 距离。它对所有误差线性惩罚，与 MSE 相比对异常值更鲁棒。

### L2 范数（欧几里得距离）

L2 范数是直线距离。是各分量平方和的平方根。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这是几何课上学到的距离。n 维空间中的毕达哥拉斯定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

何时使用 L2：
- 低到中维的连续数据
- 特征尺度可比时
- 物理距离（空间数据、传感器读数）
- 像素级别的图像相似度

与 L2 正则化（Ridge）的关系：在损失函数中添加 ||w||_2^2 会惩罚较大的权重。与 L1 不同，它不会把权重压为零，而是按比例收缩所有权重。L2 惩罚产生圆形约束区域，因此在轴上没有角落。权重会变小但很少精确为零。

与损失函数的关系：均方误差（MSE）是 L2 距离平方的平均。平方项对较大误差的惩罚要远大于对小误差的惩罚。

```
MAE (L1 loss):  |y - y_hat|         线性惩罚。对异常值鲁棒。
MSE (L2 loss):  (y - y_hat)^2       二次惩罚。对异常值敏感。
```

### Lp 范数：广义族

L1 和 L2 是 Lp 范数的特例：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的 p 值产生不同形状的“单位球”（从原点到距离为 1 的点的集合）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity 范数（切比雪夫距离）

当 p 趋于无穷大时，Lp 范数收敛到各分量绝对值的最大值。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点之间的距离由它们在某一维上差异最大的那个分量决定，其他维度被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

何时使用 L-infinity：
- 当某一维的最坏情况偏差很重要时
- 棋盘游戏（国王在棋盘上走一步的代价为 1，符合 L-infinity）
- 制造公差（每个维度都必须在规格范围内）

### 余弦相似度与余弦距离

余弦相似度衡量两个向量之间的夹角，忽略它们的幅度。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

范围从 -1（方向相反）到 +1（方向相同）。正交向量的余弦相似度为 0。

余弦距离把相似度转为距离：cosine_distance = 1 - cosine_similarity。范围从 0（方向相同）到 2（方向相反）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么余弦在 NLP 和嵌入中占主导地位：在文本中，文档长度不应影响相似性。关于猫的文档如果比另一篇关于猫的文档长一倍，仍然应该被视为“相似”。余弦相似度忽略幅度（长度），只关心方向。两个词分布相同但长度不同的文档会指向相同方向，余弦相似度为 1.0。

何时使用余弦相似度：
- 文本相似度（TF-IDF 向量、词向量、句向量）
- 任何幅度是噪声、方向是信号的领域
- 推荐系统（用户偏好向量）
- 嵌入检索（向量数据库几乎总是使用余弦或点积）

### 点积相似度 vs 余弦相似度

两个向量的点积为：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

余弦相似度是对点积按两个向量的模长进行归一化。当两个向量都已归一化（模长 = 1）时，点积和余弦相似度相同。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

当它们模长不同时：点积包含模长信息。模长更大的向量会得到更高的点积分数。在一些检索系统中，这很重要，因为你希望“热门”项目排名更高。模长充当隐含的质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实践中：
- 当你只关心方向相似时使用余弦相似度
- 当模长携带有意义信息时使用点积
- 许多向量数据库（Pinecone、Weaviate、Qdrant）允许在两者之间选择
- 如果你的嵌入已经做了 L2 归一化，选择无关紧要

### Mahalanobis 距离

欧几里得距离对所有维度一视同仁。但如果特征之间存在相关性或尺度不同，L2 会给出误导性的结果。

Mahalanobis 距离考虑了数据的协方差结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的协方差矩阵。

直观上：Mahalanobis 距离先对数据进行去相关和归一化（白化），然后在变换后的空间中计算 L2 距离。如果 S 是单位矩阵（独立且方差为 1 的特征），Mahalanobis 距离退化为欧几里得距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

何时使用 Mahalanobis 距离：
- 异常点检测（与均值距离较大的点是异常点）
- 特征具有不同尺度和相关性时的分类
- 当你有足够的数据去估计可靠的协方差矩阵时
- 制造业质量控制（多变量过程监控）

### Jaccard 相似度（用于集合）

Jaccard 相似度衡量两个集合的重叠程度。

```
J(A, B) = |A intersect B| / |A union B|
```

取值从 0（无重叠）到 1（完全相同）。Jaccard 距离 = 1 - Jaccard 相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

何时使用 Jaccard：
- 比较标签、类别或特征的集合
- 基于词是否出现（而非频率）的文档相似度
- 近重复检测（MinHash 为 Jaccard 的近似）
- 比较二值特征向量（存在/不存在数据）
- 评估分割模型（交并比 = Jaccard）

### 编辑距离（Levenshtein 距离）

编辑距离计数将一个字符串变换为另一个字符串所需的最少单字符操作数。操作包括：插入、删除或替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

通过动态规划计算。填充一个矩阵，位置 (i, j) 表示字符串 A 的前 i 个字符与字符串 B 的前 j 个字符之间的编辑距离。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

何时使用编辑距离：
- 拼写检查与纠正
- DNA 序列比对（带权操作）
- 模糊字符串匹配
- 清洗文本数据并去重

### KL 散度（不是距离，但常被当作距离使用）

KL 散度衡量一个概率分布与另一个概率分布的差异。第 09 课有覆盖，但它属于此讨论因为人们常把它当成“距离”来用，尽管它不是。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：KL 散度不是对称的。

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足距离度量的基本要求。它也不满足三角不等式。它是散度，不是距离。

前向 KL（D_KL(P || Q)）是“均值追踪”：Q 尝试覆盖 P 的所有模态。
反向 KL（D_KL(Q || P)）是“模式追踪”：Q 专注于 P 的单个模态。

你会在下面场景看到 KL：
- 变分自编码器（ELBO 中的 KL 项将潜在分布推向先验）
- 知识蒸馏（学生尝试匹配教师的分布）
- RLHF（KL 惩罚使微调后的模型保持接近基模型）
- 策略梯度方法（约束策略更新）

### Wasserstein 距离（地球搬运工距离）

Wasserstein 距离衡量将一个概率分布变换为另一个分布所需的最小“工作量”。把一个分布想象成一堆泥土，另一个分布是一个洞，要把泥土移动到洞里，问需要搬多少泥土以及搬多远。

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于一维分布，它简化为累积分布函数绝对差的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么 Wasserstein 很重要：
- 它是真正的度量（对称，满足三角不等式）
- 即使分布不重叠也能提供梯度（KL 在此情况下会趋于无穷）
- 这个性质使 Wasserstein 在 WGAN（Wasserstein GAN）中成为核心，解决了原始 GAN 的训练不稳定问题

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

何时使用 Wasserstein：
- GAN 训练（WGAN、WGAN-GP）
- 比较可能不重叠的分布
- 最优传输问题
- 图像检索（比较颜色直方图）

### 不同任务为何需要不同距离

| Task | Best distance | Why |
|------|--------------|-----|
| Text similarity | Cosine | Magnitude is noise, direction is meaning |
| Image pixel comparison | L2 | Spatial relationships matter, features are comparable scale |
| Sparse high-dim features | L1 | Robust, does not amplify rare large differences |
| Set overlap (tags, categories) | Jaccard | Data is naturally set-valued, not vectorial |
| String matching | Edit distance | Operations map to human editing intuition |
| Outlier detection | Mahalanobis | Accounts for feature correlations and scales |
| Comparing distributions | KL divergence | Measures information lost by using Q instead of P |
| GAN training | Wasserstein | Provides gradients even when distributions do not overlap |
| Embeddings (vector DB) | Cosine or dot product | Embeddings are trained to encode meaning in direction |
| Recommendation | Dot product | Magnitude can encode popularity or confidence |
| DNA sequences | Weighted edit distance | Substitution costs vary by nucleotide pair |
| Manufacturing QC | L-infinity | Worst-case deviation in any dimension matters |

### 与损失函数的联系

损失函数就是在预测与目标之间应用的距离函数。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的联系

正则化在损失函数中添加权重范数惩罚。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么 L1 会产生稀疏性而 L2 不会：想象二维权重空间中的约束区域。L1 是菱形，L2 是圆形。损失函数的等高线（椭圆）更可能在菱形的角落处与其相切，此时某个权重为零。它们在圆上相切时通常是光滑点，两个权重都非零。

### 最近邻搜索

每种距离函数都对应一个最近邻搜索问题：给定查询点，在数据集中找到最近的点。

在 n 个点、d 维情况下，精确最近邻搜索每次查询的时间为 O(n * d)。对于大规模数据集，这太慢了。

近似最近邻（ANN）算法以牺牲少量精度为代价换取巨大的速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（Hierarchical Navigable Small World）是现代向量数据库中的主导算法。它构建了一个多层图，每个节点连接到其近似最近邻。搜索从顶层开始（稀疏，长跳跃），逐层下降到底层（稠密，短跳跃）。

```figure
norm-unit-balls
```

## 实现

### 步骤 1：实现所有范数和距离函数

参见 `code/distances.py` 获取完整实现。每个函数都从零构建，仅使用基本的 Python 数学操作。

### 步骤 2：相同数据，不同距离，不同邻居

`distances.py` 中的示例创建了一个数据集，选择一个查询点，展示最近邻如何随距离度量而变化。在 L1 下“最近”的点在 L2 或余弦下可能并非最近。

### 步骤 3：嵌入相似度搜索

代码包含一个模拟的嵌入相似度搜索，会使用余弦相似度与 L2 距离来查找与查询最相似的“文档”，展示排名如何不同。

## 使用方法

最常见的实际用途：在向量数据库中查找相似项。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后在向量数据库中检索时，底层发生的就是这些。嵌入模型把文本映射为向量。向量数据库计算查询向量与每个存储向量之间的余弦相似度（或点积），并使用 ANN 算法避免遍历所有向量。

## 练习

1. 计算 (1, 2, 3) 与 (4, 0, 6) 之间的 L1、L2 和 L-infinity 距离。验证对于任意两点有 L-inf <= L2 <= L1 恒成立。证明为何该序关系总是成立。

2. 构造两个向量，使得余弦相似度很高（> 0.9）但 L2 距离很大（> 10）。几何上解释发生了什么。然后再构造两个向量，使得余弦相似度很低（< 0.3）但 L2 距离很小（< 0.5）。

3. 实现一个函数，接受数据集和查询点并分别返回 L1、L2、余弦与 Mahalanobis 距离下的最近邻。找出一个数据集，使得这四种度量对最近邻的判断全部不同。

4. 使用 CDF 方法手工计算 [0.5, 0.5, 0, 0] 与 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离。然后计算 [0.25, 0.25, 0.25, 0.25] 与 [0, 0, 0.5, 0.5] 之间的距离。哪个更大，为什么？

5. 实现 MinHash 用于近似 Jaccard 相似度。生成 100 个随机集合，计算所有对的精确 Jaccard，并用 50、100、200 个哈希函数的 MinHash 近似进行比较。绘制近似误差图。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Norm | "Size of a vector" | A function that maps a vector to a non-negative scalar, satisfying triangle inequality, absolute homogeneity, and zero only for the zero vector |
| L1 norm | "Manhattan distance" | Sum of absolute component values. Produces sparsity in optimization. Robust to outliers |
| L2 norm | "Euclidean distance" | Square root of sum of squared components. The straight-line distance in Euclidean space |
| Lp norm | "Generalized norm" | The p-th root of the sum of p-th powers of absolute components. L1 and L2 are special cases |
| L-infinity norm | "Max norm" or "Chebyshev distance" | The maximum absolute component value. The limit of Lp as p approaches infinity |
| Cosine similarity | "Angle between vectors" | Dot product normalized by both magnitudes. Ranges from -1 to +1. Ignores vector length |
| Cosine distance | "1 minus cosine similarity" | Converts cosine similarity to a distance. Ranges from 0 to 2 |
| Dot product | "Unnormalized cosine" | Sum of component-wise products. Equals cosine similarity times both magnitudes |
| Mahalanobis distance | "Correlation-aware distance" | L2 distance in a space that has been whitened (decorrelated and normalized) using the data covariance matrix |
| Jaccard similarity | "Set overlap" | Size of intersection divided by size of union. For sets, not vectors |
| Edit distance | "Levenshtein distance" | Minimum insertions, deletions, and substitutions to transform one string into another |
| KL divergence | "Distance between distributions" | Not a true distance (not symmetric). Measures extra bits from using Q to encode P |
| Wasserstein distance | "Earth mover's distance" | Minimum work to transport mass from one distribution to another. A true metric |
| Approximate nearest neighbor | "ANN search" | Algorithms (HNSW, LSH, IVF) that find approximately closest points much faster than exact search |
| HNSW | "The vector DB algorithm" | Hierarchical Navigable Small World graph. Multi-layer graph for fast approximate nearest neighbor search |
| L1 regularization | "Lasso" | Adding the L1 norm of weights to the loss. Drives weights to zero (sparsity) |
| L2 regularization | "Ridge" or "weight decay" | Adding the squared L2 norm of weights to the loss. Shrinks weights toward zero without sparsity |
| Elastic Net | "L1 + L2" | Combines L1 and L2 regularization. Handles correlated feature groups better than either alone |

## 延伸阅读

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 的亿级 ANN 检索库
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - 将地球搬运工距离引入 GAN 的论文
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - ANN 的基础算法
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec，奠定了嵌入中使用余弦相似度的实践
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中关于距离度量与邻居算法的实用指南