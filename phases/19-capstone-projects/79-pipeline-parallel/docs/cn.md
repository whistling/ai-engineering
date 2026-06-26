# Pipeline Parallel and Bubble Analysis

> Tensor parallelism splits the matrix multiply across ranks. Pipeline parallelism splits the model across ranks, one stage per rank. Microbatches flow through the pipeline. The empty time at the start and end is the bubble; minimising it is the whole craft.

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 19 Track C 第42-49课
**Time:** ~90 分钟

## 学习目标

- 将一个顺序模型拆分成 N 个阶段，并模拟跨 N 个 rank 的前向流水线。
- 使用 GPipe 调度（先前向填充全部 microbatch，再反向排空）对 M 个微批次进行调度并计算 bubble 比例。
- 将 bubble 与 Megatron-LM 和 PipeDream 中使用的交错 1F1B 调度进行比较。
- 论证阶段分配：每阶段相同的计算量比相同的参数量更重要。

## 问题背景

一个 70B 参数的模型在 fp16 下仅参数就需要 140 GB。没有消费级 GPU 可以容纳它。ZeRO-3 将参数分片到多个 rank，但在每次前向时仍然需要每个 rank allgather 整层参数，每层付出 log(N) 的跳数。流水线并行走另一条路：把模型切成 N 个阶段，每个 rank 放一个阶段。第 1 层的前向在 rank 0 上完成并把激活张量传给 rank 1；rank 1 运行第 2 层并传给 rank 2；以此类推。反向按相反方向流动。内存按线性下降，因为每个 rank 只保存一个阶段；但计算是串行的，这就是 bubble（空闲）的问题。

bubble 是流水线开始和结束处的空闲时间（开始时等待第一个微批次到达最后阶段；结束时等待最后一个微批次反向流完）。对于 M 个微批次和 N 个阶段，每阶段的 bubble 比例为 (N-1)/(M+N-1)。当 M=8、N=4 时为 27%。当 M=64、N=4 时为 4.5%。当每 step 有很多微批次时 bubble 会变小，这意味着每个微批次的批量要小，这就是驱动微批次设计的约束。

## 概念图

```mermaid
flowchart LR
  R0[rank 0：阶段 0 / 层 0] --> R1[rank 1：阶段 1 / 层 1]
  R1 --> R2[rank 2：阶段 2 / 层 2]
  R2 --> R3[rank 3：阶段 3 / 损失]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### GPipe 调度

先用所有 M 个微批次把流水线在前向方向填满，然后再开始任何反向；每个微批次的激活都必须保留直到它的反向完成，所以显存随 M 线性增长。前向需要 M+N-1 个周期，反向再需要 M+N-1 个周期。每个阶段的有用工作是 2M 周期；每个阶段的 bubble 是 2(N-1) 周期。当每个前向或反向都算作一个时间单位时，bubble 比例为 (N-1)/(M+N-1)。选择 M 远大于 N 可以掩盖 bubble。

### 1F1B 调度

交错：一旦某个微批次的前向到达最后阶段，就立刻开始它的反向并让它流回去。该调度在每个阶段交替执行一次前向和一次反向。bubble 仍为 N-1，但激活内存被限制为流水线深度，而不是微批次数。生产环境的流水线使用 1F1B（Megatron、PipeDream）。本课先实现 GPipe 因为它更简单，再把 1F1B 留作练习。

### 为什么每阶段相同的计算量很重要

如果阶段 0 花 50 ms 而阶段 1 花 100 ms，则每个周期都会被阶段 1 限制。其他阶段每个周期会空闲 50 ms 等待阶段 1 释放。相同的参数数量不是正确的划分维度：Transformer 的计算主要由注意力和 MLP 主导，embedding 层参数多但计算少。阶段划分应当使每阶段的 FLOPs 相近，而不是权重数相近。

### 微批次与批次

流水线运行 M 个大小为 B 的 microbatch。有效批量是 M*B。一次流水线 step 结束时的梯度即是对合并后的 M*B 个样本的梯度。bubble 比例取决于 M；优化器看到的是 M*B。调整 M 意味在 bubble（M 高时更低）与每微批次显存（GPipe 下 M 高时显存更高）之间权衡。

## 构建实现

`code/main.py` 实现了：

- `PipelineStage`：一个小的 `nn.Module`，保存单个阶段的参数并暴露 `forward(activation)`。
- `Pipeline(stages, num_microbatches)`：在模拟阶段上使用模拟的每阶段墙钟时间编排 GPipe 调度。
- `bubble_fraction(num_stages, num_microbatches)`：闭式解 (N-1)/(M+N-1)。
- 一个 4 阶段演示，打印每微批次的轨迹并测量 bubble 比例。

运行：

```bash
python3 code/main.py
```

输出：按阶段、按微批次的甘特图和与闭式预测对比的 bubble 百分比。

## 生产环境中的模式

三种模式使流水线并行足够可靠以投入生产。

**激活检查点与流水线配合。** 在 GPipe 上有 M 个微批次在飞时，激活内存是单个微批次的 M 倍。激活检查点在反向时重算前向，用计算换显存；二者结合使得流水线在长序列下变得可行。

**阶段平衡通过测量而不是假设。** 生产团队会做一次分析运行，在目标硬件上测量每层的实际计算量（FLOPs 和墙钟时间），然后基于测量结果进行分区。Megatron-LM 的 `--num-layers-per-stage` 参数接受列表以允许在各阶段的每层成本不同的情况下使用不均等的层数。

**发送-接收调度必须避免死锁。** 如果流水线中每个阶段都先发送再接收，会在网络层面死锁。标准修复是交错：偶数 rank 的阶段先 send 再 recv，奇数 rank 的阶段先 recv 再 send。本课显式调度 ranks 以便该模式可见。

## 使用示例

生产环境工具：

- **Megatron-LM。** 大规模流水线并行的参考实现。使用 1F1B，并支持 tensor + pipeline + data parallel 的组合。
- **DeepSpeed Pipeline。** 与 ZeRO 集成；ZeRO-1 + pipeline 是最大开源模型的常见组合。
- **PyTorch Pipe。** PyTorch 原生的 pipeline 包装，基于 `torch.distributed.pipeline.sync.Pipe` 构建。

## 上线细节

第 80 课将每阶段参数分片存入分片检查点。第 81 课在端到端示例上组合 DDP + ZeRO + pipeline（在精神上；示例在运行时保持流水线为模拟状态）。

## 练习

1. 实现 1F1B 并验证 bubble 比例与 GPipe 相同，但激活内存被限制。
2. 在更深的模型上分析真实的每阶段时间并通过测量结果重新平衡阶段。
3. 在流水线微批次上加入梯度累积，并检验梯度是否等于等效全批次前向的梯度。
4. 将流水线与激活检查点配对，测量显存下降与计算成本的变化。
5. 将流水线与 DDP 结合（每个流水线 rank 在数据并行组中有复制），并推理 2D 调度。

## 术语表

| 术语 | 常说法 | 实际含义 |
|------|--------|----------|
| 流水线并行 (Pipeline) | "Model parallel along depth" | 每个 rank 一个阶段，激活在阶段间流动 |
| Bubble（气泡/空闲） | "Pipeline idle time" | 在开始和结束处有 (N-1) 步某些阶段没有工作 |
| 微批次 (Microbatch) | "Slice of the batch" | 一个前向/反向单元；随着 M 增大 bubble 缩小 |
| GPipe | "Fill then drain" | 所有 M 个前向在任何反向之前完成；激活内存高 |
| 1F1B | "Interleaved schedule" | 每阶段交替执行一次前向一次反向；激活内存有界 |

## 延伸阅读

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Megatron-LM pipeline parallel docs](https://github.com/NVIDIA/Megatron-LM)
- Phase 19 Lesson 76 - the send/recv primitives the schedule uses
- Phase 19 Lesson 78 - ZeRO is orthogonal to pipeline and often combined