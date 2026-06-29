# 线性代数直观理解

> 每个 AI 模型不过是戴着花哨帽子的矩阵运算。

**Type:** 学习
**Languages:** Python, Julia
**Prerequisites:** Phase 0
**Time:** ~60 分钟

## 学习目标

- 从零实现向量与矩阵运算（加法、点积、矩阵乘法），使用 Python
- 几何上解释点积、投影与格拉姆-施密特过程的含义
- 使用行约简判断向量集的线性无关性、秩与基
- 将线性代数概念与其在 AI 中的应用联系起来：嵌入、注意力分数、以及 LoRA

## 问题

打开任意一篇机器学习论文。在第一页之内，你会看到向量、矩阵、点积与变换。没有线性代数直观理解时，这些只是符号；有了直观理解，你就能看到神经网络在做什么——在空间中移动点。

你不需要成为数学家。你需要理解这些运算在几何上的含义，然后自己动手编码实现它们。

## 概念

### 向量就是点（也是方向）

向量只是数字的列表。但这些数字有含义——它们是空间中的坐标。

**二维向量 [3, 2]:**

| x | y | 点 |
|---|---|---|
| 3 | 2 | 该向量从原点 (0,0) 指向平面上的 (3, 2) |

该向量的长度为 sqrt(3^2 + 2^2) = sqrt(13)，方向向上且向右。

在 AI 中，向量代表一切：
- 一个词 → 一个 768 维的向量（它在嵌入空间中的“含义”）
- 一张图像 → 数百万像素值组成的向量
- 一个用户 → 表示偏好的向量

### 矩阵就是变换

矩阵把一个向量变换为另一个向量。它可以旋转、缩放、拉伸或投影。

```mermaid
graph LR
    subgraph Before
        A["点 A"]
        B["点 B"]
    end
    subgraph Matrix["矩阵乘法"]
        M["M（变换）"]
    end
    subgraph After
        A2["点 A'"]
        B2["点 B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

在 AI 中，矩阵就是模型：
- 神经网络权重 → 将输入变换为输出的矩阵
- 注意力分数 → 决定关注什么的矩阵
- 嵌入 → 将词映射为向量的矩阵（从训练数据或嵌入层得到）

### 点积测量相似度

两个向量的点积告诉你它们有多相似。

```
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

同方向：      a · b > 0  （相似）
垂直：        a · b = 0  （不相关）
相反方向：    a · b < 0  （不相似）
```

这正是搜索引擎、推荐系统和 RAG 的工作原理——找到点积较高的向量。

### 线性无关性

如果集合中的某个向量不能被表示为其他向量的线性组合，则这些向量是线性无关的。如果 v1、v2、v3 是线性无关的，它们可以张成一个三维空间。如果其中一个是其他向量的组合，则它们只能张成一个平面。

为什么这对 AI 很重要：你的特征矩阵应该有线性无关的列。如果两个特征完全相关（线性相关），模型无法区分它们的影响。这会在回归中引起多重共线性——权重矩阵变得不稳定，输入的小变化会引起输出的巨大波动。

**具体例子：**

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 等于 2*v1 + v2
```

v1 和 v2 是线性无关的——彼此无法通过标量倍数或线性组合得到。但 v3 = 2*v1 + v2，所以集合 {v1, v2, v3} 是线性相关的。这三个向量都位于 xy 平面上。无论如何组合它们，你都无法到达 [0, 0, 1]。你有三个向量，但只有二维的自由度。

在数据集中：如果 feature_3 = 2*feature_1 + feature_2，那么添加 feature_3 对模型没有新增信息。更糟的是，这会使正规方程变得奇异——权重没有唯一解。

### 基与秩

基是能够张成整个空间的最小线性无关向量集。基向量的个数就是空间的维度。

三维空间的标准基是 {[1,0,0], [0,1,0], [0,0,1]}。但三维空间中的任意三个线性无关向量也构成一个有效的基。选择基就是选择坐标系。

矩阵的秩 = 线性无关的列数 = 线性无关的行数。如果秩 < min(rows, cols)，矩阵就是秩亏（秩不足）。这意味着：
- 系统可能有无穷多解（或无解）
- 变换中丢失信息
- 矩阵不可逆

| 情况 | 秩 | 对机器学习的意义 |
|------|----|------------------|
| 满秩 (rank = min(m, n)) | 最大可能 | 最小二乘解唯一。模型条件良好。 |
| 秩不足 (rank < min(m, n)) | 低于最大 | 特征冗余。权重有无穷多解。需要正则化。 |
| 秩为 1 | 1 | 每一列都是某个向量的缩放副本。所有数据都在一条直线上。 |
| 接近秩亏（小的奇异值） | 数值上低 | 矩阵病态。微小输入噪声会导致大的输出变化。使用奇异值分解截断（SVD 截断）或岭回归。 |

### 投影

将向量 a 投影到向量 b 上，得到 a 在 b 方向上的分量：

```
proj_b(a) = (a dot b / b dot b) * b
```

残差（a - proj_b(a)）与 b 垂直。这个正交分解是最小二乘拟合的基础。

投影在机器学习中无处不在：
- 线性回归最小化观测值到列空间的距离 —— 解就是一个投影
- PCA 将数据投影到方差最大的方向
- Transformer 中的注意力计算是将 queries 投影到 keys 上的相似性度量

```mermaid
graph LR
    subgraph Projection["向量 a 在向量 b 上的投影"]
        direction TB
        O["原点"] --> |"b（方向）"| B["b"]
        O --> |"a（原向量）"| A["a"]
        O --> |"proj_b(a)"| P["投影"]
        A -.-> |"残差（垂直）"| P
    end
```

**示例：** a = [3, 4], b = [1, 0]

proj_b(a) = (3*1 + 4*0) / (1*1 + 0*0) * [1, 0] = 3 * [1, 0] = [3, 0]

投影会丢掉 y 分量。这是最简单形式的降维 —— 丢弃你不关心的方向。

### 格拉姆-施密特过程

把任意一组线性无关的向量转换为正交归一基。正交归一表示每个向量长度为 1 并且两两垂直。

算法：
1. 取第一个向量并归一化
2. 取第二个向量，减去它在第一个向量上的投影，然后归一化
3. 取第三个向量，减去它在所有先前向量上的投影，然后归一化
4. 对剩余向量重复上述操作

```
输入:  v1, v2, v3, ... (线性无关)

u1 = v1 / |v1|

w2 = v2 - (v2 dot u1) * u1
u2 = w2 / |w2|

w3 = v3 - (v3 dot u1) * u1 - (v3 dot u2) * u2
u3 = w3 / |w3|

输出: u1, u2, u3, ... (正交归一基)
```

这正是 QR 分解的内部工作方式。Q 是正交归一基，R 保存投影系数。QR 分解用于：
- 求解线性系统（比高斯消元更稳定）
- 计算特征值（QR 算法）
- 最小二乘回归（标准的数值方法）

```figure
eigen-directions
```

## 动手实现

### 第 1 步：从零实现向量（Python）

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
```

### 第 2 步：从零实现矩阵（Python）

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
```

### 第 3 步：这对 AI 为什么重要

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")
```

### 第 4 步：Julia 版本

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a ⋅ b)       # Julia 支持 Unicode 操作符
println("|a| = ", √(a ⋅ a))
println("cosine = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# 矩阵-向量乘法
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("This is a neural network layer.")
```

### 第 5 步：从零实现线性无关性与投影（Python）

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

## 在实践中使用

现在用 NumPy 做同样的事 —— 这是你在实际中会用到的：

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### 使用 NumPy 的秩、投影与 QR

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")
```

### PyTorch —— 张量是带自动求导的向量

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

点积对 x 的梯度就是 y。PyTorch 自动计算了这一点。神经网络中的每个操作都由类似的运算（矩阵乘法、点积、投影等）构成，自动微分会跟踪它们的梯度。

你刚刚从零实现了 NumPy 用一行完成的功能。现在你知道底层发生了什么。

## 发布成果

本课将产出：
- `outputs/prompt-linear-algebra-tutor.md` -- 一个用于 AI 助手通过几何直观教授线性代数的提示词

## 关联

本课中的所有内容都与现代 AI 的具体部分相关：

| 概念 | 出现在哪里 |
|------|-----------|
| 点积 | Transformer 中的注意力分数，RAG 中的余弦相似度 |
| 矩阵乘法 | 每个神经网络层，所有线性变换 |
| 线性无关 | 特征选择，避免多重共线性 |
| 秩 | 判断系统是否可解，LoRA（低秩适配） |
| 投影 | 线性回归（投影到列空间）、PCA |
| 格拉姆-施密特 / QR | 数值求解器、特征值计算 |
| 正交归一基 | 稳定的数值计算，白化变换 |

值得特别提到 LoRA。它通过将权重更新分解为低秩矩阵来微调大型语言模型。与更新一个 4096×4096 的权重矩阵（1600 万参数）相比，LoRA 更新两个矩阵：4096×16 和 16×4096（13.1 万参数）。秩为 16 的约束意味着 LoRA 假设权重更新存在于完整 4096 维空间的一个 16 维子空间中。这就是线性代数在实际工作中发挥作用的例子。

## 练习

1. 实现 `Vector.angle_between(other)`，返回两个向量间的角度（单位：度）
2. 创建一个 2D 缩放矩阵，使 x 坐标翻倍、y 坐标变为三倍，然后将其应用到向量 [1, 1]
3. 给定 5 个随机的类词向量（维度 50），使用余弦相似度找出最相似的两个
4. 验证格拉姆-施密特的输出是否真正正交归一：检查每对向量的点积是否为 0 且每个向量的模是否为 1
5. 创建一个秩为 2 的 3x3 矩阵。使用 `rank()` 方法验证。然后解释其列张成的几何对象是什么
6. 将向量 [1, 2, 3] 投影到 [1, 1, 1] 上。几何上，这个结果表示什么？

## 关键词

| 术语 | 大家怎么说 | 它真正的含义 |
|------|-----------|--------------|
| Vector | "一支箭" | 一个数字列表，表示 n 维空间中的点或方向 |
| Matrix | "一张数字表" | 一个将向量从一个空间映射到另一个空间的变换 |
| Dot product | "乘加求和" | 衡量两个向量对齐程度的量 —— 相似性搜索的核心 |
| Embedding | "某种 AI 魔法" | 表示某物（词、图像、用户）含义的向量 |
| Linear independence | "它们不重合" | 集合中没有向量可以由其他向量表示 |
| Rank | "有多少维" | 矩阵中线性无关的列（或行）的个数 |
| Projection | "影子" | 一个向量在另一个向量方向上的分量 |
| Basis | "坐标轴" | 能张成空间的最小线性无关向量集 |
| Orthonormal | "垂直且单位长度" | 两两正交且每个向量长度为 1 |