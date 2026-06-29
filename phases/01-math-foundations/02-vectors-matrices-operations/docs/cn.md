# 向量、矩阵与运算

> 每个神经网络不过是在做矩阵乘法并加上一些额外步骤。

**Type:** 构建
**Languages:** Python, Julia
**Prerequisites:** 阶段 1，课程 01（线性代数直觉）
**Time:** ~60 分钟

## 学习目标

- 构建一个带有逐元素运算、矩阵乘法、转置、行列式和逆的 Matrix 类
- 区分逐元素乘法与矩阵乘法，并解释各自适用的场景
- 使用从零实现的 Matrix 类，仅用它实现单个全连接神经网络层（`relu(W @ x + b)`）
- 解释广播规则以及神经网络框架中偏置相加是如何工作的

## 问题背景

你想要构建一个神经网络。你读到一行代码，看见：

```
output = activation(weights @ input + bias)
```

这里的 `@` 是矩阵乘法。`weights` 是一个矩阵。`input` 是一个向量。如果你不知道这些操作是什么，这一行就是魔法。如果你知道，它就是一层前向传播的全部：三步运算。

每张图像在模型中都是像素值的矩阵。每个词嵌入是一个向量。每个神经网络的每一层都是矩阵变换。不了解矩阵运算，就无法构建 AI 系统，就像不会使用变量就无法写代码一样。

本课从头构建这种流利度。

## 概念

### 向量：有序的数字列表

向量是带有方向和大小的数字列表。在 AI 中，向量表示数据点、特征或参数。

```
v = [3, 4]        -- 一个二维向量
w = [1, 0, -2]    -- 一个三维向量
```

二维向量 `[3, 4]` 指向平面上坐标 (3, 4)。它的长度（模）为 5（3-4-5 直角三角形）。

### 矩阵：数字的网格

矩阵是二维网格。行和列。一个 m x n 矩阵有 m 行 n 列。

```
A = | 1  2  3 |     -- 2x3 矩阵（2 行，3 列）
    | 4  5  6 |
```

在神经网络中，权重矩阵将输入向量变换为输出向量。一个有 784 个输入和 128 个输出的层使用一个 128x784 的权重矩阵。

### 为什么形状很重要

矩阵乘法有严格规则：`(m x n) @ (n x p) = (m x p)`。内维度必须匹配。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- 有效
```

如果在 PyTorch 中遇到形状不匹配错误，原因通常就是这个。

### 运算一览

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | 元素逐一相加 | 将偏置加到输出上 |
| Scalar multiply | 缩放每个元素 | 学习率 * 梯度 |
| Matrix multiply | 变换向量 | 层的前向传播 |
| Transpose | 翻转行和列 | 反向传播 |
| Determinant | 单个数字的概括 | 检查是否可逆 |
| Inverse | 撤销变换 | 求解线性系统 |
| Identity | 什么也不做的矩阵 | 初始化、残差连接 |

### 逐元素乘法 vs 矩阵乘法

这个区别经常让初学者困惑。

逐元素：对应位置相乘。两个矩阵必须形状相同。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法：行与列的点积。内维度必须匹配。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

不同的运算、不同的结果、不同的规则。

### 广播（Broadcasting）

当你把偏置向量加到输出矩阵时，形状可能不匹配。广播会将较小的数组扩展以匹配较大的数组。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

广播会把向量沿行扩展：

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架都会自动处理广播。理解它能避免在形状看似正确但代码运行时却产生困惑的情况。

```figure
vector-projection
```

## 动手实现

### 第 1 步：Vector 类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 第 2 步：包含核心运算的 Matrix 类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 第 3 步：运行查看效果

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 第 4 步：与神经网络的连接

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是一个单层全连接层：`output = relu(W @ x + b)`。每个神经网络中的每个全连接层都正是做这件事。

## 使用它

NumPy 用更少的代码行并且快几个数量级地完成上面所有操作。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` 操作符会调用 `__matmul__`。NumPy 使用用 C 和 Fortran 编写并优化过的 BLAS 例程来实现它。数学相同，但速度快 100 倍。

NumPy 中的广播示例：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 会自动将一维偏置沿行广播。这就是每个神经网络框架中偏置相加的工作方式。

## 交付

本课会输出一个用于通过几何直觉教授矩阵运算的提示词，见 `outputs/prompt-matrix-operations.md`。

这里构建的 Matrix 类是我们在第 3 阶段，第 10 课中构建微型神经网络框架的基础。

## 练习

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()` 并确认是否得到单位矩阵。用三个不同的 2x2 矩阵尝试。当行列式为零时会发生什么？

2. **实现 3x3 逆矩阵。** 在 Matrix 类中扩展以使用伴随矩阵法（adjugate method）计算 3x3 矩阵的逆。与 NumPy 的 `np.linalg.inv` 进行对比测试。

3. **构建一个两层网络。** 仅使用你实现的 Matrix 类（不使用 NumPy），创建一个两层神经网络：输入 (3) -> 隐藏 (4) -> 输出 (2)。随机初始化权重，运行前向传递，并验证所有形状是否正确。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Vector | "An arrow" | 一个有序的数字列表。在 AI 中：高维空间中的一个点。 |
| Matrix | "A table of numbers" | 一个线性变换。将向量从一个空间映射到另一个空间。 |
| Matrix multiply | "Just multiply the numbers" | 第一矩阵的每一行与第二矩阵的每一列做点积。顺序很重要。 |
| Transpose | "Flip it" | 交换行和列。把 m x n 矩阵变为 n x m。在反向传播中很关键。 |
| Determinant | "Some number from the matrix" | 测量矩阵对面积（2D）或体积（3D）的缩放程度。为零意味着变换压缩了一个维度。 |
| Inverse | "Undo the matrix" | 能撤销变换的矩阵。只有当行列式不为零时才存在。 |
| Identity matrix | "The boring matrix" | 相当于乘以 1 的矩阵。用于残差连接（ResNets）。 |
| Broadcasting | "Magic shape fixing" | 通过在缺失的维度上重复来扩展较小的数组以匹配较大的数组。 |
| Element-wise | "Regular multiplication" | 对应位置相乘。两个数组必须具有相同形状（或可广播）。 |

## 延伸阅读

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 对本课覆盖的每个运算提供可视化直觉
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 采用的精确规则
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 面向机器学习的线性代数简洁参考