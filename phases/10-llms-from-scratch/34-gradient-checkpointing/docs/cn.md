# 梯度检查点与激活重计算

> 反向传播会保留每一个中间激活。在 70B 参数和 128K 上下文的情况下，这在每个 rank 上会产生 3 TB 的激活。检查点通过在计算量上做权衡来节省内存：重算而不是保存。问题是丢弃哪些段，而答案不是“全部丢弃”。

**Type:** 构建  
**Languages:** Python（使用 numpy，可选 torch）  
**Prerequisites:** Phase 10 Lesson 04（预训练 Mini-GPT）、Phase 10 Lesson 05（扩展与分布式）  
**Time:** ~70 分钟

## 问题

训练一个 transformer 时，会为每一层保存在反向传播中需要微分的每个操作的输入：注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、归一化输出以及残差流。对于隐藏维度为 `d`、序列长度 `L`、批次大小 `B` 的一层，这大约是每层 `12 * B * L * d` 个浮点数。

对于 `d=8192, L=8192, B=1`，每层在 BF16 下大约是 800 MB。64 层模型的激活量是 51 GB —— 这还是在你乘上微批次大小、加上注意力 softmax 中间量（每头为 `L^2`）以及张量并行的部分副本之前的估算。

两方面的代价：BF16 权重加上优化器状态可能装得下 80GB，但激活会把你推超。梯度检查点（又名激活重计算）是常见的解决办法。丢弃大部分激活；在反向过程中重新做前向以恢复它们。代价是额外 FLOPs。收益是内存按检查点段与总层数的比率下降。

如果简单地做，检查点会导致每步前向大约增加 33% 的 FLOPs。做得好——按照 Korthikanti 等人的“智能选择”做选择性检查点——你可以在不到 5% 的 FLOP 开销下节省 5 倍内存。再配合 FP8 矩阵乘、FSDP 异地（offload）、以及专家并行 MoE，这一点非常关键：你既负担不起内存，也负担不起浪费的计算。

## 概念

### 反向传播实际上需要什么

`output = layer(input)`。反向需要 `grad_input` 和 `grad_params`。为了计算它们需要：

- `input`（线性层需要它来计算 `grad_params = input.T @ grad_output`）
- 一些激活导数的中间量（ReLU/GELU/softmax 的导数依赖于激活值本身）

前向会在自动求导图中自动保存这些。每个 `tensor.retain_grad()` 和每个需要其输入的操作都会保存引用。

### 朴素全检查点

把网络分成 `N` 个段。前向时，只保存每个段的*输入*。当反向需要中间量时，重新运行该段的前向以将它们物化，然后再求导。

示例：32 层 transformer 被分成 32 个单层段。

- 内存：保存 32 个层输入（小）对比 保存 32 *（每层激活体积）（巨大）。
- 额外计算：每段多一次前向，即总前向 FLOPs 约增加 ~33%（因为反向是 2x 前向，完整一步变成 1 + 1 + 2 = 4 单位而不是 1 + 2 = 3）。

这就是最初 Chen et al. 2016 的方法：每 `sqrt(L)` 层放一个检查点，以平衡内存与计算。对于 L=64，这就是 8 个检查点。

### 选择性检查点（Korthikanti 2022）

并非所有激活的成本相同。注意力 softmax 的输出是 `B*L*L*heads`，并且随序列长度 *二次增长*。FFN 隐藏激活是 `B*L*4d`，按线性增长。对于长序列，softmax 占主导。

选择性检查点保留那些便于存储的激活（线性投影、残差等），只重算代价高的（注意力）。你需要付出最小的 FLOPs 来重算，但能节省 O(L^2) 的内存。

Megatron-Core 将其实现为“selective”激活重计算。大多数 2024+ 的前沿训练运行都在使用它。

### 异地（Offload）

重算的替代方案：在前向和反向之间将激活传到 CPU RAM。需要 PCIe 带宽；当空闲带宽大于重计算成本时有利。混合策略很常见：对部分层做检查点，对部分做 offload。

FSDP2 把 offload 作为一等选项。当 GPU 受内存限制而 CPU-GPU 传输有余量时，offload 更有优势。

### 重算成本模型

对每步，若在 `L` 层中每 `k` 层做一次朴素检查点：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 每段每层再多一次前向
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

使用选择性检查点只重算注意力核，而不是整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活体积：`A`。对于 `L` 层，总激活内存：`L * A`。

完全检查点（段大小为 1）：只保存 `L * input_volume`（对于标准 transformer 约为 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层做一次检查点：保存 `L/k * A`，并在激活段内保留 `k-1` 层的激活。

在 `k = sqrt(L)` 时，内存和重算成本都随 `sqrt(L)` 缩放 —— 对于均匀代价层这是最优折衷。

### 什么时候不做检查点

- 管道并行阶段内已经在传输（in-flight）的最内层。它们无论如何都要完成。
- 第一层和最后一层如果占据了阶段的大量计算（在 transformer 中少见）。
- 注意力核已使用 FlashAttention —— Flash 已经很快地重算 softmax，因此额外的层级检查点收益不大。

### 实现模式

1. 函数包装：用 `torch.utils.checkpoint.checkpoint(fn, input)` 包裹一个段。PyTorch 只保存 `input`，在反向时重算其他内容。
2. 装饰器式：标记可以检查点的层；trainer 在配置时决定哪些段被包裹。
3. 手工显式重算：自己写反向过程，调用自定义的 `recompute_forward`，用保存的输入重复前向。

这三种在功能上等价。包装器是常用的习惯用法。

### 与 TP / PP / FP8 的交互

- Tensor parallel：检查点的输入在重算时必须被 gather 或者重新分散（rescatten）；需要处理通信开销。
- Pipeline parallel：典型模式是对每个 pipeline stage 的前向做检查点，以便反向顺序的微批次可以重用激活内存。
- FP8 重算：重算期间更新的 amax 历史必须与原始前向一致，否则 FP8 比例会漂移。大多数框架会快照（snapshot）这些尺度信息。

## 实现

### 第 1 步：带段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第 2 步：需要所有激活的朴素反向

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 第 3 步：每 k 层检查点的内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 第 4 步：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 第 5 步：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 第 6 步：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 第 7 步：选择性重算决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用方法

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` —— PyTorch 中的规范包装器。包裹一个函数；只保存输入，在反向时重算。
- **Megatron-Core 激活重计算**：支持 `selective`、`full` 与 `block` 模式。2024+ 的前沿训练中广泛采用。
- **FSDP2 offload**：使用 `module.to_empty(device="cpu")` 配合 FSDP2 的 `offload_policy`，将激活分片到 CPU 而不是重算。
- **DeepSpeed ZeRO-Offload**：针对优化器状态和激活的 CPU offload，可与检查点机制互补。

## 发布成果

本课的产物为 `outputs/prompt-activation-recompute-policy.md` —— 一个接收你的模型配置（层数、hidden、seq、batch）和可用 GPU 内存并输出逐层重算策略（none / selective / full / offload）的提示（prompt）。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（全激活）与 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）。参数梯度必须在机器精度下相同。
2. 扫描段大小 `k` 从 1 到 `L`。绘制 FLOP 开销与内存。找到曲线的“拐点”。
3. 实现选择性检查点：保存注意力模块的输入但不保存其中间量。测量序列长度为 8192 时 32 层模型下选择性检查点相比全层检查点的 FLOP 开销。
4. 添加 offload。把段输入保存到一个模拟的“CPU 缓冲区”（一个单独的列表）。把“PCIe 带宽”模拟为 bytes/time，找到 offload 与重算的折返点。
5. 在真实的 PyTorch transformer 上对比有无 `torch.utils.checkpoint` 的情况。测量内存（通过 `torch.cuda.max_memory_allocated`）和每步时间。

## 关键术语

| 术语 | 大家怎么说 | 实际意思 |
|------|-----------|---------|
| Gradient checkpointing | “通过重做前向来节省内存” | 只保存段的输入；在反向时重算中间量以获得用于梯度的张量 |
| Activation recomputation | “和检查点一样” | HPC 场景下对同一技术的称呼 |
| Segment size (k) | “每个检查点包含多少层” | 被丢弃并在一起重算的层数 |
| Selective checkpointing | “Korthikanti 的技巧” | 只重算高成本的激活（注意力 softmax）；保留便宜的激活 |
| Full checkpointing | “朴素版本” | 在每个段中重算每一层的中间量 |
| Block checkpointing | “粗粒度” | 对整个 transformer block 做检查点；最粗的粒度 |
| FLOP overhead | “计算税” | 每步的额外 FLOPs = (重算 FLOPs) / (前向 + 反向 FLOPs)；朴素约 33%，选择性约 5% |
| Activation offload | “传到 CPU” | 在前向->反向 期间把激活移到 CPU RAM；重算的替代方案 |
| sqrt-L rule | “经典最优” | 对于均匀代价的层，最佳检查点间隔是 sqrt(L) 层 |
| Attention-softmax volume | “O(L^2) 问题” | L^2 * heads * batch 的浮点数；在长上下文下主导激活内存 |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 将梯度检查点形式化的原始论文  
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 选择性激活重计算与形式化成本分析  
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 通过反向模式重物化（rematerialization）实现常数内存的替代方法  
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 大规模训练中的激活 offload  
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API  
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- selective、full 与 block 模式的说明