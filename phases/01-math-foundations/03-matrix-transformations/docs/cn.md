# 矩阵变换

> 矩阵是一台重塑空间的机器。了解它对每一个点的作用，你就理解了整个变换。

**Type:** 构建
**Languages:** Python, Julia
**Prerequisites:** 阶段 1，课程 01-02（线性代数直觉，向量与矩阵运算）
**Time:** ~75 分钟

## 学习目标

- 构建旋转、缩放、剪切和反射矩阵，并将它们应用到 2D 和 3D 点上
- 通过矩阵乘法组合多个变换并验证顺序的重要性
- 从特征方程推导 2x2 矩阵的特征值和特征向量
- 解释为什么特征值决定 PCA 方向、RNN 稳定性和谱聚类行为

## 问题背景

你读到 PCA 时会看到 “求协方差矩阵的特征向量”。你读到模型稳定性时会看到 “检查所有特征值的模是否小于 1”。你读到数据增强时会看到 “应用一个随机旋转”。在你理解矩阵在几何上对空间做了什么之前，这些说法都不会有直观意义。

矩阵不仅仅是数字的网格。它们是空间机器。旋转矩阵会旋转点。缩放矩阵会拉伸点。剪切矩阵会倾斜点。神经网络对数据施加的每一个变换，都是这些操作之一或它们的组合。本课让这些操作变得具体可感。

## 概念

### 将变换表示为矩阵

每一个二维线性变换都可以写成一个 2x2 矩阵。矩阵会精确告诉你基向量 [1, 0] 和 [0, 1] 的去向。其他所有点都由此推导。

```mermaid
graph LR
    subgraph Before["标准基"]
        e1["e1 = [1, 0]（沿 x）"]
        e2["e2 = [0, 1]（沿 y）"]
    end
    subgraph Transform["矩阵 M"]
        M["M = 列是新的基向量"]
    end
    subgraph After["变换后的 M"]
        e1p["e1' = 新的 x 基"]
        e2p["e2' = 新的 y 基"]
    end
    e1 --> M --> e1p
    e2 --> M --> e2p
```

### 旋转

二维中围绕角度 theta 的旋转保持距离和角度不变。它沿圆弧移动每一个点。

```mermaid
graph LR
    subgraph Before["旋转前"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Rot["旋转 45 度"]
        R["R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]"]
    end
    subgraph After["旋转后"]
        Ap["A'(0.71, 2.12)"]
        Bp["B'(-1.41, 1.41)"]
    end
    A --> R --> Ap
    B --> R --> Bp
```

在三维中，你会绕一个轴旋转。每个轴都有自己对应的旋转矩阵：

```
Rz(theta) = | cos  -sin  0 |     绕 z 轴旋转
            | sin   cos  0 |     （x-y 平面旋转，z 保持不变）
            |  0     0   1 |

Rx(theta) = | 1   0     0    |   绕 x 轴旋转
            | 0  cos  -sin   |   （y-z 平面旋转，x 保持不变）
            | 0  sin   cos   |

Ry(theta) = |  cos  0  sin |     绕 y 轴旋转
            |   0   1   0  |     （x-z 平面旋转，y 保持不变）
            | -sin  0  cos |
```

### 缩放

缩放沿每个轴独立地拉伸或压缩。

```mermaid
graph LR
    subgraph Before["缩放前"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Scale["缩放 sx=2, sy=0.5"]
        S["S = [[2, 0], [0, 0.5]]"]
    end
    subgraph After["缩放后"]
        Ap["A'(4, 0.5)"]
        Bp["B'(0, 1)"]
    end
    A --> S --> Ap
    B --> S --> Bp
```

### 剪切

剪切让一条轴保持不变，同时倾斜另一条轴。它把矩形变为平行四边形。

```mermaid
graph LR
    subgraph Before["剪切前"]
        A["A(1, 0)"]
        B["B(0, 1)"]
    end
    subgraph Shear["x 方向剪切，k=1"]
        Sh["Shx = [[1, k], [0, 1]]"]
    end
    subgraph After["剪切后"]
        Ap["A(1, 0) 未改变"]
        Bp["B'(1, 1) 平移"]
    end
    A --> Sh --> Ap
    B --> Sh --> Bp
```

剪切矩阵：
- `Shx = [[1, k], [0, 1]]` 将 x 平移 k * y
- `Shy = [[1, 0], [k, 1]]` 将 y 平移 k * x

### 反射

反射沿某轴或某线将点镜像翻转。

```mermaid
graph LR
    subgraph Before["反射前"]
        A["A(2, 1)"]
    end
    subgraph Reflect["关于 y 轴反射"]
        R["[[-1, 0], [0, 1]]"]
    end
    subgraph After["反射后"]
        Ap["A'(-2, 1)"]
    end
    A --> R --> Ap
```

反射矩阵：
- 关于 y 轴反射: `[[-1, 0], [0, 1]]`
- 关于 x 轴反射: `[[1, 0], [0, -1]]`

### 组合：链式变换

先应用变换 A 然后 B 等价于将它们的矩阵相乘：`result = B @ A @ point`。顺序很重要。先旋转再缩放的结果与先缩放再旋转不同。

```mermaid
graph LR
    subgraph Path1["先旋转 90° 然后缩放 (2, 0.5)"]
        P1["(1, 0)"] -->|"旋转 90°"| P2["(0, 1)"] -->|"缩放"| P3["(0, 0.5)"]
    end
```

组合后：`S @ R = [[0, -2], [0.5, 0]]`

```mermaid
graph LR
    subgraph Path2["先缩放 (2, 0.5) 然后旋转 90°"]
        Q1["(1, 0)"] -->|"缩放"| Q2["(2, 0)"] -->|"旋转 90°"| Q3["(0, 2)"]
    end
```

组合后：`R @ S = [[0, -0.5], [2, 0]]`

结果不同。矩阵乘法不满足交换律。

### 特征值和特征向量

大多数向量在矩阵作用下方向会改变。特征向量是特殊的：矩阵只会对其进行缩放，而不会旋转它们。缩放的因子就是特征值。

```
A @ v = lambda * v

v 是特征向量（保持方向的向量）
lambda 是特征值（拉伸的倍数）

示例: A = | 2  1 |
           | 1  2 |

特征向量 [1, 1] 对应特征值 3:
  A @ [1,1] = [3, 3] = 3 * [1, 1]     （同方向，被放大 3 倍）

特征向量 [1, -1] 对应特征值 1:
  A @ [1,-1] = [1, -1] = 1 * [1, -1]  （同方向，不变）
```

这个矩阵沿 [1, 1] 方向将空间拉伸 3 倍，而沿 [1, -1] 方向保持不变。其他方向都是这两个方向的线性组合。

### 特征分解

如果一个矩阵有 n 个线性无关的特征向量，就可以分解为：

```
A = V @ D @ V^(-1)

V = 列为特征向量的矩阵
D = 对角的特征值矩阵
V^(-1) = V 的逆

这表示：先旋转到特征向量坐标系，沿每个轴进行缩放，然后再旋转回去。
```

### 为什么特征值重要

**PCA。** 协方差矩阵的特征向量是主成分。特征值告诉你每个成分捕获了多少方差。按特征值排序，保留前 k 个，就完成了降维。

**稳定性。** 在循环网络和动力系统中，模大于 1 的特征值会导致输出发散。模小于 1 会导致输出消失。这一句话概括了梯度消失/爆炸问题的本质。

**谱方法。** 图神经网络使用邻接矩阵的特征值。谱聚类使用拉普拉斯矩阵的特征值。特征向量揭示了图的结构。

### 行列式作为体积缩放因子

变换矩阵的行列式告诉你它把面积（二维）或体积（三维）缩放了多少。

```
det = 1:   面积保持（旋转）
det = 2:   面积翻倍
det = 0:   空间被压缩到更低维（奇异）
det = -1:  面积保持但方向翻转（反射）

| det(旋转) | = 1        （总是如此）
| det(缩放 sx, sy) | = sx * sy
| det(剪切) | = 1           （面积保持）
| det(反射) | = -1          （方向翻转）
```

```figure
matrix-transform
```

## 实现

### 步骤 1：从零实现变换矩阵（Python）

```python
import math

def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

def reflection_x():
    return [[1, 0], [0, -1]]

def reflection_y():
    return [[-1, 0], [0, 1]]

def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]

point = [1.0, 0.0]
angle = math.pi / 4

rotated = mat_vec_mul(rotation_2d(angle), point)
print(f"Rotate (1,0) by 45 deg: ({rotated[0]:.4f}, {rotated[1]:.4f})")

scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
print(f"Scale (1,1) by (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")

sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
print(f"Shear (1,1) kx=1: ({sheared[0]:.1f}, {sheared[1]:.1f})")

reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
print(f"Reflect (2,1) across y: ({reflected[0]:.1f}, {reflected[1]:.1f})")
```

### 步骤 2：变换的组合

```python
R = rotation_2d(math.pi / 2)
S = scaling_2d(2, 0.5)

rotate_then_scale = mat_mul(S, R)
scale_then_rotate = mat_mul(R, S)

point = [1.0, 0.0]
result1 = mat_vec_mul(rotate_then_scale, point)
result2 = mat_vec_mul(scale_then_rotate, point)

print(f"Rotate 90 then scale: ({result1[0]:.2f}, {result1[1]:.2f})")
print(f"Scale then rotate 90: ({result2[0]:.2f}, {result2[1]:.2f})")
print(f"Same? {result1 == result2}")
```

### 步骤 3：从零求特征值（2x2）

对于 2x2 矩阵 `[[a, b], [c, d]]`，特征值由特征方程求解：`lambda^2 - (a+d)*lambda + (ad - bc) = 0`。

```python
def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]

A = [[2, 1], [1, 2]]
vals = eigenvalues_2x2(A)
print(f"Matrix: {A}")
print(f"Eigenvalues: {vals[0]:.4f}, {vals[1]:.4f}")

for val in vals:
    vec = eigenvector_2x2(A, val)
    result = mat_vec_mul(A, vec)
    scaled = [val * vec[0], val * vec[1]]
    print(f"  lambda={val:.1f}, v={[round(x,4) for x in vec]}")
    print(f"    A@v = {[round(x,4) for x in result]}")
    print(f"    l*v = {[round(x,4) for x in scaled]}")
```

### 步骤 4：行列式作为体积缩放因子

```python
def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

print(f"det(rotation 45) = {det_2x2(rotation_2d(math.pi/4)):.4f}")
print(f"det(scale 2,3)   = {det_2x2(scaling_2d(2, 3)):.1f}")
print(f"det(shear kx=1)  = {det_2x2(shearing_2d(1, 0)):.1f}")
print(f"det(reflect y)   = {det_2x2(reflection_y()):.1f}")

singular = [[1, 2], [2, 4]]
print(f"det(singular)     = {det_2x2(singular):.1f}")
print("Singular: columns are proportional, space collapses to a line.")
```

## 使用现成库

NumPy 提供了所有这些操作的高效实现。

```python
import numpy as np

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

point = np.array([1.0, 0.0])
print(f"Rotate (1,0) by 45 deg: {R @ point}")

S = np.diag([2.0, 3.0])
composed = S @ R
print(f"Scale(2,3) after Rotate(45): {composed @ point}")

A = np.array([[2, 1], [1, 2]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"  A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

print(f"\ndet(R) = {np.linalg.det(R):.4f}")
print(f"det(S) = {np.linalg.det(S):.1f}")

B = np.array([[3, 1], [0, 2]], dtype=float)
vals, vecs = np.linalg.eig(B)
D = np.diag(vals)
V = vecs
reconstructed = V @ D @ np.linalg.inv(V)
print(f"\nEigendecomposition A = V @ D @ V^-1:")
print(f"Original:\n{B}")
print(f"Reconstructed:\n{reconstructed}")
```

### 使用 NumPy 的 3D 旋转

```python
def rotation_3d_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotation_3d_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

point_3d = np.array([1.0, 0.0, 0.0])
rotated_z = rotation_3d_z(np.pi / 2) @ point_3d
rotated_x = rotation_3d_x(np.pi / 2) @ point_3d

print(f"\n3D point: {point_3d}")
print(f"Rotate 90 around z: {np.round(rotated_z, 4)}")
print(f"Rotate 90 around x: {np.round(rotated_x, 4)}")
```

## 发布价值

本课为 PCA（阶段 2）和神经网络权重分析建立了几何基础。此处实现的特征值/特征向量算法就是生产级 ML 系统中用于降维、谱聚类和稳定性分析的基础算法。

## 练习

1. 将旋转、缩放和剪切应用到单位正方形（角点为 [0,0], [1,0], [1,1], [0,1]）。打印每种变换后的角点坐标。验证旋转是否保持角点之间的距离。

2. 用特征方程手算矩阵 [[4, 2], [1, 3]] 的特征值。然后用你从零实现的函数和 NumPy 验证结果。

3. 创建三个变换的组合（旋转 30 度，缩放 [1.5, 0.8]，x 方向剪切 kx=0.3），并将其应用到 8 个排列成圆形的点上。打印变换前后的坐标。计算组合矩阵的行列式并验证它等于各单独变换行列式的乘积。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| 旋转矩阵 (Rotation matrix) | "Spins things" | 一个正交矩阵，它沿圆弧移动点同时保持距离和角度。行列式恒为 1。 |
| 缩放矩阵 (Scaling matrix) | "Makes things bigger" | 一个对角矩阵，沿各轴独立拉伸或压缩。行列式为各缩放因子的乘积。 |
| 剪切矩阵 (Shearing matrix) | "Slants things" | 将一个坐标按另一个坐标成比例平移的矩阵，将矩形变为平行四边形。行列式为 1。 |
| 反射 (Reflection) | "Mirrors things" | 沿轴或平面翻转空间的矩阵。行列式为 -1。 |
| 组合 (Composition) | "Do two things" | 通过相乘变换矩阵来链式操作。B @ A 表示先应用 A，再应用 B。 |
| 特征向量 (Eigenvector) | "Special direction" | 矩阵只对其做缩放不做旋转的方向。变换的指纹。 |
| 特征值 (Eigenvalue) | "How much it stretches" | 矩阵对其特征向量的缩放因子。可以为负（翻转）或复数（对应旋转）。 |
| 特征分解 (Eigendecomposition) | "Break the matrix apart" | 将矩阵写成 V @ D @ V^(-1)，把矩阵分解为基本的缩放方向和幅度。 |
| 行列式 (Determinant) | "A single number from a matrix" | 变换对面积（2D）或体积（3D）的缩放因子。为零表示变换不可逆。 |
| 特征方程 (Characteristic equation) | "Where eigenvalues come from" | det(A - lambda * I) = 0。其多项式的根即为特征值。 |

## 延伸阅读

- [3Blue1Brown: 线性变换 (Linear Transformations)](https://www.3blue1brown.com/lessons/linear-transformations) -- 对矩阵如何重塑空间的可视化直觉  
- [3Blue1Brown: 特征向量与特征值 (Eigenvectors and Eigenvalues)](https://www.3blue1brown.com/lessons/eigenvalues) -- 关于特征向量几何意义的最佳可视化讲解  
- [MIT 18.06 Lecture 21: Eigenvalues and Eigenvectors](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) -- Gilbert Strang 的经典讲解