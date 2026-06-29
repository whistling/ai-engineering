# Introduction to JAX

> PyTorch 会就地修改张量。TensorFlow 构建计算图。JAX 编译纯函数。最后这一点会改变你看待深度学习的方式。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10，基础 NumPy
**Time:** ~90 分钟

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数神经网络代码
- 解释 PyTorch 的即时变异（eager mutation）与 JAX 的函数式编译模型之间的关键设计差异
- 应用 jit 编译和 vmap 向量化，相对于朴素 Python 加速训练循环
- 在 JAX 中训练一个简单网络，并将显式状态管理与 PyTorch 的面向对象方法对比

## 问题背景

你知道如何在 PyTorch 中构建神经网络。定义一个 `nn.Module`，调用 `.backward()`，执行优化器的 step。它可以工作，数百万人都在用。

但 PyTorch 有一个内生约束：它在 Python 中逐条急切地跟踪操作。每个 `tensor + tensor` 都是一次单独的内核调用。每个训练步骤都会重新解释同一段 Python 代码。这在大多数场景下没问题，直到你需要在 2,048 个 TPU 上训练一个 5400 亿参数的模型。那时开销会把你击垮。

Google DeepMind 在 JAX 上训练 Gemini。Anthropic 在 JAX 上训练 Claude。这些都不是小规模工作——它们是地球上最大的神经网络训练运行。它们选择 JAX，因为 JAX 把你的训练循环当作可编译的程序，而不是一系列 Python 调用。

JAX 是带有三大超能力的 NumPy：自动微分、JIT 编译到 XLA、自动向量化。你编写一个处理单个样本的函数。JAX 会给你一个能够处理批次、计算梯度、编译为机器码并在多设备上运行的函数。所有这些都不需要修改原始函数。

## 概念

### JAX 哲学

JAX 是一个函数式框架。没有类、没有可变状态、没有 `.backward()` 方法。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| 带有状态的 `nn.Module` 类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 即时执行（eager execution） | 通过 XLA 的 JIT 编译 |
| 对批次手动 `for x in batch:` 循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 不可变的数组 pytree |

这不仅仅是风格偏好。这是编译器的约束。JIT 编译要求纯函数——相同的输入总是产生相同的输出、没有副作用。正是这种限制使得 100 倍的加速成为可能。

### jax.numpy：熟悉的接口

JAX 在加速设备上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

函数名相同。广播规则相同。切片语义相同。但数组位于 GPU/TPU 上，且每个操作都可被编译器追踪。

一个关键区别：JAX 的数组是不可变的。不能做 `a[0] = 5`。取而代之：`a = a.at[0].set(5)`。这会让人觉得尴尬一周，随后你会理解——不可变性是让 `grad`、`jit` 和 `vmap` 这类变换可组合的原因。

### jax.grad：函数式自动微分

PyTorch 将梯度附加到张量（`.grad`）。JAX 将梯度附加到函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接受一个函数并返回一个新的函数，该函数计算梯度。没有 `.backward()` 调用。张量上也没有保存计算图。梯度只是另一个你可以调用、组合或 JIT 编译的函数。

这可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导、三阶导、雅可比、海森矩阵——都可以通过组合 `grad` 获得。PyTorch 也能做到（例如 `torch.autograd.functional.hessian`），但那是后加的。在 JAX 中，自动微分是基础设施级别的。

限制：`grad` 只作用于纯函数。不能在追踪时打印（打印会在跟踪期间执行，而非实际运行时）。不能修改外部状态。随机数生成必须显式管理 PRNG key。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次调用时，JAX 会对函数进行 tracing —— 记录发生了哪些操作，但不实际执行它们。然后将这个 trace 交给 XLA（Accelerated Linear Algebra），Google 的 TPU/GPU 编译器。XLA 会融合操作、消除多余的内存拷贝，并生成优化过的机器码。

后续调用会跳过 Python。编译后的代码在加速器上以 C++ 速度运行。

JIT 有益的场景：
- 训练步骤（相同计算重复数千次）
- 推理（相同模型，输入不同）
- 任何被多次以相似形状输入调用的函数

JIT 不适合的场景：
- 依赖于值的 Python 控制流（如 `if x > 0`，其中 x 是被跟踪的数组）
- 一次性计算（编译开销超过运行时间）
- 调试（跟踪会隐藏真实执行）

控制流限制是真实存在的。`jax.lax.cond` 用以替代 `if/else`。`jax.lax.scan` 用以替代 `for` 循环。这些不是可选项——它们是换取编译能力的代价。

### vmap：自动向量化

你编写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 可把它提升为处理一个批次的函数：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 表示：对 `params` 不做批处理（共享），对 `x` 的第 0 轴做批处理。无需手写 `for` 循环、无需重塑、无需在代码中穿线批次维度。JAX 会自动识别批次维度并向量化整个计算。

这不是语法糖。`vmap` 生成融合的向量化代码，运行速度比 Python 循环快 10-100 倍。并且它可以与 `jit` 和 `grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度。只需一行。在 PyTorch 中这几乎不可能做到（除非用技巧）。

### pmap：跨设备的数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 会在所有可用设备（GPU/TPU）上复制函数并拆分批次。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 用于跨设备同步梯度。

Google 使用 `pmap`（及其后继 `shard_map`）在成千上万的 TPU v5e 芯片上训练 Gemini。编程模型：编写单设备版本，封装为 `pmap`，完成。

### Pytrees：通用数据结构

JAX 在“pytrees”上操作——嵌套的列表、元组、字典和数组的组合。你的模型参数就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 变换 —— `grad`, `jit`, `vmap` —— 都知道如何遍历 pytrees。`jax.tree.map(f, tree)` 会对每个叶子节点应用 `f`。这就是优化器如何一次性更新所有参数的方式：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法。没有参数注册。树结构就是模型。

### 函数式 vs 面向对象

PyTorch 将状态存放在对象内部：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用带显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数通过传入。没有存储。没有变异。这使得每个函数都易于测试、组合和编译。与此同时，这也意味着你需要自己管理 params —— 或使用像 Flax 或 Equinox 这样的库。

### JAX 生态系统

JAX 提供原语（primitives）。库提供易用性：

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | 神经网络层 | 带显式状态的 `nn.Module` |
| **Equinox** (Patrick Kidger) | 神经网络层 | 基于 pytree，风格更 Pythonic |
| **Optax** (DeepMind) | 优化器 + 学习率调度 | 可组合的梯度变换 |
| **Orbax** (Google) | 检查点（checkpointing） | 保存/恢复 pytrees |
| **CLU** (Google) | 指标 + 日志 | 训练循环辅助工具 |

Optax 是标准的优化器库。它将梯度变换（Adam、SGD、裁剪）与参数更新分离，使得组合变换变得极其简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时使用 JAX 或 PyTorch

| 因素 | JAX | PyTorch |
|------|-----|--------|
| TPU 支持 | 一级（Google 自己构建） | 社区维护（torch_xla） |
| GPU 支持 | 良好（通过 XLA 的 CUDA） | 最佳（原生 CUDA） |
| 调试 | 较难（跟踪 + 编译） | 简单（即时执行，逐行调试） |
| 生态 | 研究导向（Flax、Equinox） | 庞大（HuggingFace、torchvision 等） |
| 招聘 | 小众（Google/DeepMind/Anthropic） | 主流（各处需求） |
| 大规模训练 | 优势（XLA、pmap、mesh） | 良好（FSDP、DeepSpeed） |
| 原型速度 | 较慢（函数式管理开销） | 更快（就地变异，快速迭代） |
| 生产推理 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 谁在用 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

坦率的回答是：除非你有明确理由使用 JAX，否则使用 PyTorch。那些理由通常包括 —— 可获取 TPU、需要逐样本梯度、需要在极大规模上多设备训练，或者你在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每次随机操作都需要显式的 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始这令人烦躁。但它能保证在多设备和编译情况下的可复现性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 场景下无法保证的。

```figure
batchnorm-effect
```

## 实战构建

### 步骤 1：环境与数据

我们将使用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP。输入维度 784，两个隐藏层为 256 和 128 神经元，输出 10 类。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 步骤 2：参数初始化

没有类。只是一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He 初始化，手动完成。三个 PRNG key 从一个种子中分裂出来。每个权重都是嵌套字典中的不可变数组。

### 步骤 3：前向计算

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。参数输入，得到预测输出。没有 `self`，没有存储状态。`loss_fn` 从头计算交叉熵 —— softmax、取对数、求负均值。

### 步骤 4：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 在一次传递中同时返回 loss 值和梯度。`@jax.jit` 装饰器会将这两个函数编译为 XLA。第一次调用后，每个训练步骤都无需触碰 Python。

### 步骤 5：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个 epoch。大约 97% 的测试准确率。第一轮慢（JIT 编译）。第 2-10 轮会很快。

注意缺失了什么：没有 `.zero_grad()`、没有 `.backward()`、没有 `.step()`。整个更新是一个组合函数调用。梯度被计算、由 Adam 变换并应用到参数上 —— 所有这些都在 `train_step` 内完成。

## 使用指南

### Flax：Google 的标准

Flax 是最常用的 JAX 神经网络库。它把 `nn.Module` 引入，但采用显式状态管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与 PyTorch 相同，但 `params` 与模型对象分离。`model.init()` 创建参数。`model.apply(params, x)` 用于前向计算。模型对象本身不保存状态。

### Equinox：更 Pythonic 的替代

Equinox（作者 Patrick Kidger）将模型表示为 pytrees：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是一个 pytree。无需 `.apply()`。参数只是模型的叶子节点。这更贴近 JAX 的思维方式。

### Optax：可组合的优化器

Optax 将梯度变换与更新分离：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减 —— 全部作为变换链组合。每个变换接收梯度、修改它们并传给下一个。没有单体式的优化器类。

## 部署与注意事项

**安装：**

```bash
pip install jax jaxlib optax flax
```

GPU 支持：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意点：**

- 第一次 JIT 调用很慢（编译开销）。在基准测试前进行 warm-up。
- 避免在 JIT 内对 JAX 数组使用 Python 循环。使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 可以在 JIT 内工作。普通 `print()` 则不行。
- 用 `jax.profiler` 或 TensorBoard 进行分析。XLA 编译可能会掩盖瓶颈。
- JAX 默认会预分配 GPU 内存的 75%。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可禁用该行为。

**检查点（Checkpointing）：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md` -- 用于选择合适 JAX 优化器配置的提示文档
- `outputs/skill-jax-patterns.md` -- 关于 JAX 中函数式模式的技能总结

## 练习

1. 在 MLP 中加入 dropout。在 JAX 中，dropout 需要 PRNG key —— 在前向过程中传递一个 key 并为每个 dropout 层拆分它。比较有无 dropout 的测试准确率。

2. 使用 `jax.vmap` 为一批 32 张 MNIST 图像计算逐样本梯度。计算每个样本的梯度范数。哪些样本具有最大的梯度，为什么会这样？

3. 将手写的前向函数替换为一个通用的 `mlp_forward(params, x)`，使其适用于任意层数。使用 `jax.tree.leaves` 自动确定深度。

4. 对比带和不带 `@jax.jit` 的训练步骤性能。对各自执行 100 步计时。在你的硬件上加速倍数是多少？第一次调用的编译开销有多大？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。比较有无裁剪的训练效果。绘制训练过程中梯度范数随时间的变化以观察影响。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|---------|
| XLA | “让 JAX 快的东西” | Accelerated Linear Algebra —— 一个将计算图融合并为 GPU/TPU 生成优化内核的编译器 |
| JIT | “即时编译（Just-in-time）” | JAX 在首次调用时对函数做 tracing，编译为 XLA，后续调用运行编译版本 |
| 纯函数 | “无副作用” | 输出仅依赖输入的函数 —— 无全局状态、无变异、无显式 key 的随机性 |
| vmap | “自动批处理（Auto-batching）” | 将处理单个样本的函数变换为处理批次的函数，无需重写代码 |
| pmap | “自动并行（Auto-parallelism）” | 在多设备上复制函数并拆分输入批次 |
| Pytree | “嵌套的数组字典” | JAX 可遍历和变换的任意嵌套结构（lists、tuples、dicts、arrays） |
| Tracing | “记录计算” | JAX 用抽象值执行函数以构建计算图，而不是计算实际结果 |
| 函数式自动微分 | “函数的 grad” | 通过变换函数而不是将梯度存储附加到张量来计算导数 |
| Optax | “JAX 的优化器库” | 一套可组合的梯度变换 —— Adam、SGD、裁剪、调度等 |
| Flax | “JAX 的 nn.Module” | Google 的 JAX 神经网络库，添加层抽象同时保持显式状态 |

## 延伸阅读

- JAX 文档：https://jax.readthedocs.io/ -- 官方文档，包含关于 grad、jit、vmap 的优秀教程
- “JAX: composable transformations of Python+NumPy programs” (Bradbury et al., 2018) — 解释设计哲学的原始论文
- Flax 文档：https://flax.readthedocs.io/ -- Google 的 JAX 神经网络库文档
- Patrick Kidger, “Equinox: neural networks in JAX via callable PyTrees and filtered transformations” (2021) — Flax 的更 Python 化替代
- DeepMind, “Optax: composable gradient transformation and optimisation” — 标准的优化器库
- “You Don't Know JAX” (Colin Raffel, 2020) — 来自 T5 作者的实用指南，覆盖 JAX 的陷阱与常见模式