# 数值稳定性

> 浮点数是一个会泄漏的抽象。在训练过程中它会咬你一口，而且你不会预料到。

**Type:** Build  
**Language:** Python  
**Prerequisites:** Phase 1, Lessons 01-04  
**Time:** ~120 分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的溢出、下溢与灾难性消失（catastrophic cancellation）
- 使用中心差分验证解析梯度与数值梯度的一致性
- 解释为什么在训练中 bfloat16 优于 float16，以及损失缩放如何防止梯度下溢

## 问题背景

你的模型训练了三个小时，然后 loss 变成 NaN。你加了一个打印语句。step 9,000 时 logits 还正常。step 9,001 时它们变成了 `inf`。到 step 9,002 每个梯度都是 `nan`，训练就死掉了。

或者：你的模型训练完成但准确率比论文差 2%。你检查了一切。架构一致。超参数一致。数据一致。问题是论文用的是 float32，而你用了 float16 且没有做正确的缩放。32 位的累积舍入误差悄悄吞掉了你的精度。

或者：你从头实现了交叉熵损失。在小的 logits 上工作正常。当 logits 超过 100 时返回 `inf`。softmax 溢出了，因为 `exp(100)` 比 float32 能表示的要大。每个 ML 框架都有两行的技巧来处理这个问题。你并不知道有这个技巧。

数值稳定性不是理论问题。它决定了一次训练是成功还是悄然失败。每个严重的 ML bug 最终都会归结到浮点问题。

## 概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准将实数存为浮点值。一个浮点数由三部分组成：符号位、指数和尾数（有效数）。

```
Float32 布局（总共 32 位）：
[1 符号] [8 指数] [23 尾数]

值 = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度（有效数字位数）。指数决定范围（一个数能有多大或多小）。

```
格式       位数   指数位   尾数位   十进制有效位数   范围（近似）
float64    64     11      52       ~15-16           +/- 1.8e308
float32    32     8       23       ~7-8             +/- 3.4e38
float16    16     5       10       ~3-4             +/- 65,504
bfloat16   16     8       7        ~2-3             +/- 3.4e38
```

float32 给出约 7 位十进制精度。这意味着它能区分 1.0000001 和 1.0000002，但不能区分 1.00000001 和 1.00000002。超过 7 位之后就是四舍五入噪声。

float16 只有大约 3 位精度。它能表示的最大数是 65,504。对于在训练中常见的 logits、梯度和激活值而言，这是个令人担忧的上限。

bfloat16 是 Google 针对 float16 范围问题的回答。它具有与 float32 相同的 8 位指数（相同范围，最高可达 3.4e38），但只有 7 位尾数（比 float16 精度更低）。对于神经网络训练而言，范围比精度更重要，因此 bfloat16 通常更适合。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 在二进制浮点中不能被精确表示。在基数 2 中它是一个循环小数：

```
0.1 的二进制表示 = 0.0001100110011001100110011...（无限循环）
```

float32 把它截断为 23 位尾数。存储的值大约是 0.100000001490116。同理，0.2 存储为大约 0.200000002980232。它们的和为 0.300000004470348，而不是 0.3。

```
在 Python 中：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 有影响，因为：

1. 类似 `if loss < threshold` 的损失比较可能给出错误判断  
2. 累加许多小值（成千上万步的梯度更新）会偏离真实和  
3. 校验和与可重现性测试在用 `==` 比较浮点时会失败

解决方法：绝不要用 `==` 比较浮点。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性消失（Catastrophic Cancellation）

当你减去两个几乎相等的浮点数时，有效数字会相互抵消，剩下的就是被放大的舍入噪声。

```
a = 1.0000001    （在 float32 中存储为 1.00000011920929）
b = 1.0000000    （在 float32 中存储为 1.00000000000000）

真实差值： 0.0000001
计算差值： 0.00000011920929

相对误差：19.2%
```

单次减法就造成了 19% 的相对误差。在 ML 中，这种情况会在以下场景出现：

- 用带大均值的数据计算方差：E[x^2] - E[x]^2，当 E[x] 很大时
- 减去几乎相等的对数概率
- 用太小的 epsilon 计算有限差分梯度

修正方法：重排公式以避免减去大且接近的数。对于方差，用 Welford 算法或先对数据中心化。对于对数概率，全程在对数空间中操作。

### 溢出与下溢

溢出发生在结果太大以至于无法表示。下溢发生在结果太小（接近零，低于可表示的最小正数）。

```
Float32 边界：
  最大值：    3.4028235e+38
  最小正值（正规化）：1.175e-38
  最小正值（非正规化）：1.401e-45
  溢出：      任何 > 3.4e38 的数变为 inf
  下溢：      任何 < 1.4e-45 的数变为 0.0
```

在 ML 中 `exp()` 是造成溢出的主要来源：

```
exp(88.7)  = 3.40e+38   （勉强适合 float32）
exp(89.0)  = inf         （溢出）
exp(-87.3) = 1.18e-38    （刚好高于下溢）
exp(-104)  = 0.0         （下溢为零）
```

`log()` 则在另一个方向出现问题：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      （没有问题）
log(1e-46) = -inf        （输入先下溢为 0，然后 log(0) = -inf）
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。组合 `log(exp(x))` 若不使用恰当技巧将是一片雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上很危险。如果有任意一个 `x_i` 很大，`exp(x_i)` 会溢出。如果所有 `x_i` 都非常负，所有 `exp(x_i)` 都会下溢为零，从而 `log(0)` 为 `-inf`。

技巧：在指数运算前先减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么可行：减去 `max(x)` 后最大的指数项为 `exp(0) = 1`。不可能溢出。至少有一项是 1，所以和至少为 1，`log(1) = 0`。不会下溢为 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    （加减 c）
= log(sum(exp(x_i - c) * exp(c)))               （exp(a+b) = exp(a)*exp(b)）
= log(exp(c) * sum(exp(x_i - c)))               （把 exp(c) 提取出来）
= c + log(sum(exp(x_i - c)))                    （log(a*b) = log(a) + log(b)）
```

设 c = max(x)，则溢出被消除了。

这个技巧出现在 ML 的各处：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率累加
- 高斯混合模型
- 变分推断

### 为什么 Softmax 需要最大值减法技巧

Softmax 将 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

不使用技巧时，logits 为 [100, 101, 102] 会导致溢出：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

这些会溢出 float32（最大值 ~3.4e38）？实际上：
exp(88.7) 已经接近 float32 上限。
exp(100) 在 float32 中会是 inf。
```

使用技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率与原来相同。计算安全。这个不是优化，而是正确性的前提。

### NaN 与 Inf：检测与预防

`nan`（Not a Number）和 `inf`（无穷大）会在计算中病毒式传播。一次梯度更新中的 `nan` 会使权重变为 `nan`，进而使所有后续输出变为 `nan`。训练在一步内就可能死亡。

如何产生 `inf`：
- 对很大的正数做 `exp()`
- 除以零：`1.0 / 0.0`
- float32 在累加中溢出

如何产生 `nan`：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数做 `sqrt()`
- 对负数做 `log()`
- 任何包含已有 `nan` 的运算

检测：

```python
import math

math.isnan(x)       # 如果 x 是 nan 则为 True
math.isinf(x)       # 如果 x 是 +inf 或 -inf 则为 True
math.isfinite(x)    # 如果 x 既不是 nan 也不是 inf 则为 True
```

预防策略：

1. 对 `exp()` 的输入做截断：`exp(clamp(x, -80, 80))`  
2. 在分母加上 epsilon：`x / (y + 1e-8)`  
3. 在 `log()` 内加上 epsilon：`log(x + 1e-8)`  
4. 使用稳定实现（log-sum-exp、stable softmax）  
5. 梯度裁剪以防止权重爆炸  
6. 在调试期间每次前向后检查 `nan`/`inf`

### 数值梯度检查

解析梯度（来自反向传播）可能有 bug。数值梯度检查通过有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ≈ (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 级别的精度，比前向差分 `(f(x+h) - f(x)) / h`（仅 O(h)）要好得多。

h 的选择：太大近似错误，太小则灾难性消失会破坏结果。典型取值为 h = 1e-5 到 1e-7。

检查方式：计算解析梯度与数值梯度的相对差异。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确  
- relative_error < 1e-5：可接受，很可能正确  
- relative_error > 1e-3：可能有问题  
- relative_error > 1：梯度完全错误

在实现新层或损失函数时务必检查梯度。PyTorch 提供 `torch.autograd.gradcheck()` 来做这件事。

### 混合精度训练

现代 GPU 有专用硬件（Tensor Cores）可以用 float16 做矩阵乘法，比 float32 快 2-8 倍。混合精度训练就是利用这一点：

```
1. 保留一份 float32 的权重主副本
2. 前向用 float16（更快）
3. 损失用 float32（防止溢出）
4. 反向用 float16（更快）
5. 将梯度缩放为 float32
6. 更新 float32 主权重
```

纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。float16 会把小于 ~6e-8 的数下溢为零。模型停止学习，因为所有梯度更新都为零。

修复方法是损失缩放（loss scaling）：

```
1. 将 loss 乘以一个大的缩放因子（例如 1024）
2. 反向传播计算 (loss * 1024) 的梯度
3. 所有梯度变为原来的 1024 倍（推到 float16 的下溢范围之上）
4. 在更新权重前将梯度除以 1024
5. 净效果：更新与未缩放一致，但避免了下溢
```

动态损失缩放会自动调整缩放因子。以较大值（例如 65536）开始。如果梯度溢出为 `inf`，则将因子减半。如果连续 N 步没有溢出，则将因子翻倍。

### bfloat16 vs float16：为什么 bfloat16 在训练中占优

```
float16:   [1 符号] [5 指数]  [10 尾数]
bfloat16:  [1 符号] [8 指数]  [7 尾数]
```

float16 有更多精度（10 位尾数对比 7 位），但范围有限（最大 ~65,504）。bfloat16 精度更低但与 float32 具有相同的指数范围（最大 ~3.4e38）。

对于训练神经网络：

- 激活和 logits 在训练尖峰期间经常超过 65,504。float16 会溢出；bfloat16 可以处理。  
- float16 需要损失缩放来避免下溢，但 bfloat16 通常不需要，因为其指数范围覆盖了梯度大小的谱。  
- bfloat16 是对 float32 的简单截断：丢掉尾数的低 16 位。转换简单，并且在指数层面是无损的。

float16 更适合推理，在那里数值有界且精度更重要。bfloat16 更适合训练，在那里范围更重要。这也是为什么 TPU 和现代 NVIDIA GPU（A100、H100）都原生支持 bfloat16。

### 梯度裁剪

梯度爆炸发生在梯度在许多层中呈指数增长（常见于 RNN、深层网络和 transformer）。一次大的梯度可能在一步内破坏所有权重。

两种裁剪方法：

Clip by value（按值裁剪）：对每个梯度元素独立截断。

```
grad = clamp(grad, -max_val, max_val)
```

简单但可能改变梯度向量的方向。

Clip by norm（按范数裁剪）：缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保持了梯度方向。这正是 `torch.nn.utils.clip_grad_norm_()` 的做法，是标准选择。

典型值：transformer 常用 `max_norm=1.0`，强化学习常用 `max_norm=0.5`，较简单网络可用 `max_norm=5.0`。

梯度裁剪不是权宜之计，而是一种安全机制。没有它，一次异常的 batch 可能产生足以毁掉数周训练的大梯度。

### 归一化层作为数值稳定器

BatchNorm、LayerNorm 和 RMSNorm 通常被解释为有助于收敛的正则化器。它们同时也是数值稳定器。

没有归一化时，激活可能在层间呈指数增长或衰减：

```
Layer 1: 值在 [0, 1]
Layer 5: 值在 [0, 100]
Layer 10: 值在 [0, 10,000]
Layer 50: 值在 [0, inf]
```

归一化在每层重新中心并重新缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

epsilon（通常为 1e-5）避免当所有激活相同时除以零。学习参数 `gamma` 和 `beta` 允许网络恢复任何所需的尺度。

这能在网络中保持数值安全范围，防止前向时溢出和反向时梯度爆炸。

### 常见的 ML 数值错误

Bug：训练若干 epoch 后 loss 为 NaN。  
原因：logits 太大，softmax 溢出。或学习率太高导致权重发散。  
修复：使用稳定的 softmax（最大值减法），降低学习率，加入梯度裁剪。

Bug：loss 停留在 log(num_classes)。  
原因：模型输出接近均匀分布。通常表示梯度消失或模型根本没有学到东西。  
修复：检查数据标签是否正确，验证损失函数，检查死 ReLU。

Bug：验证准确率比预期低 1-3%。  
原因：混合精度但没有正确做损失缩放。梯度下溢悄然把小的更新置为零。  
修复：启用动态损失缩放，或切换到 bfloat16。

Bug：某些层的梯度范数为 0.0。  
原因：死 ReLU（所有输入为负），或 float16 下溢。  
修复：使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

Bug：模型在一张 GPU 上正常，在另一张上结果不同。  
原因：浮点累加的顺序非确定性。GPU 并行约简在不同硬件上加法顺序不同，而浮点加法不可交换。  
修复：接受小差异（例如 1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受性能损失。

Bug：在损失计算中 `exp()` 返回 `inf`。  
原因：把未做最大值减法的原始 logits 传给 `exp()`。  
修复：使用 `torch.nn.functional.log_softmax()`，它内部实现了 log-sum-exp。

Bug：从 float32 切换到 float16 后训练发散。  
原因：float16 无法表示小于 ~6e-8 的梯度或大于 65,504 的激活。  
修复：使用带损失缩放的混合精度（AMP），或改用 bfloat16。

```figure
logsumexp-stability
```

## 实作

### 步骤 1：演示浮点精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤 2：实现朴素与稳定的 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### 步骤 3：实现稳定的 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 会返回 inf
```

### 步骤 4：实现稳定的交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤 5：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用示例

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现（含所有边界情况示例）请参见 `code/numerical.py`。

## 交付成果

本课产出：
- `code/numerical.py`，包含稳定的 softmax、log-sum-exp、交叉熵、梯度检查以及混合精度模拟  
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 与数值问题

这些稳定实现将在 Phase 3 的训练循环构建与 Phase 4 的注意力机制实现中再次出现。

## 练习

1. 灾难性消失。使用朴素公式 `E[x^2] - E[x]^2`（float32）计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后使用 Welford 在线算法计算。将结果与真实方差（0.6667）比较误差。

2. 精度探测。找出在 Python 中最小的正 float32 值 `x`，使得 `1.0 + x == 1.0`。这就是机器精度（machine epsilon）。验证它是否等于 `numpy.finfo(numpy.float32).eps`。

3. Log-sum-exp 边界情况。用你的 `logsumexp_stable` 测试以下情况：(a) 所有值相等，(b) 一个值远大于其余，(c) 所有值非常负（如 -1000）。验证在这些情况下稳定版本能给出正确结果，而朴素版本可能失败。

4. 对神经网络层做梯度检查。实现单层线性变换 `y = Wx + b` 及其解析反向传播。用 `numerical_gradient` 验证 3x2 权重矩阵的正确性。

5. 损失缩放实验。模拟 float16 训练：生成范围在 [1e-9, 1e-3] 的随机梯度，转换为 float16，测量变为零的比例。然后应用损失缩放（乘以 1024），转换为 float16，再缩回，比较零值比例的变化。

## 关键词

| 术语 | 常说法 | 实际含义 |
|------|--------|---------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf, nan）的国际标准。每个现代 CPU 和 GPU 都实现了它。 |
| 机器精度（Machine epsilon） | "精度极限" | 在给定浮点格式下，使得 1.0 + e != 1.0 的最小值 e。对于 float32，大约是 1.19e-7。 |
| 灾难性消失（Catastrophic cancellation） | "减法造成的精度损失" | 当减去几乎相等的浮点数时，有效数字被抵消，舍入噪声成为结果主导因素。 |
| 溢出（Overflow） | "数太大" | 结果超过可表示的最大值并变为 inf。例如 exp(89) 会使 float32 溢出。 |
| 下溢（Underflow） | "数太小" | 结果比可表示的最小正数还小并变为 0.0。例如 exp(-104) 在 float32 下会下溢。 |
| Log-sum-exp 技巧 | "先减去最大值" | 通过提取 exp(max(x)) 来计算 log(sum(exp(x)))，以防止溢出和下溢。在 softmax、交叉熵和对数概率计算中常用。 |
| 稳定 softmax（Stable softmax） | "不会爆炸的 softmax" | 在指数运算前减去 max(logits)，数值上等价，但避免了溢出。 |
| 梯度检查（Gradient checking） | "验证反向传播" | 将解析梯度与有限差分计算的数值梯度比较，用以捕捉实现中的错误。 |
| 混合精度（Mixed precision） | "前向用 float16，关键处用 float32" | 在速度敏感操作中使用低精度浮点，在数值敏感操作中使用高精度浮点。典型加速为 2-3x。 |
| 损失缩放（Loss scaling） | "防止梯度下溢" | 在反向传播前将损失乘以大常数让梯度落在 float16 表示范围内，更新前再除以相同常数。 |
| bfloat16 | "Brain 浮点格式" | Google 的 16 位格式，具有 8 位指数（与 float32 相同范围）和 7 位尾数（精度低于 float16）。训练中更常用。 |
| 梯度裁剪（Gradient clipping） | "限制梯度范数" | 缩放梯度向量使其范数不超过阈值，防止一次大的梯度毁掉模型。 |
| NaN | "Not a Number" | 由未定义运算（0/0、inf-inf、sqrt(-1)）产生的特殊浮点值，会被后续运算传播。 |
| Inf | "Infinity" | 由溢出或除以零产生的特殊浮点值。与 NaN 的某些组合（inf - inf, inf * 0）会生成 NaN。 |
| 数值梯度（Numerical gradient） | "暴力求导" | 通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。慢但可靠，用于验证。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考，内容密集但完整  
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- 引入 float16 训练中损失缩放的 NVIDIA 论文  
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch 中混合精度的实用指南  
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- 解释 Google 为何在 TPU 上选择此格式  
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 一种减少浮点求和舍入误差的算法