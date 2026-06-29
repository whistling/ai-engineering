# DualPipe 并行

> DeepSeek-V3 在 2,048 个 H800 GPU 上训练，MoE 的专家分散在各节点。跨节点专家的 all-to-all 通信造成每 1 GPU 小时的计算就对应 1 GPU 小时的通信成本。GPU 有一半时间处于空闲。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，它将前向和反向计算与由它们触发的 all-to-all 通信重叠。气泡缩小，通过率上升，而且在 Expert Parallelism 已经把专家分散到各 rank 的情况下，保持两个模型参数副本（给出“Dual”名称的原因）代价很低。本课是一个 Learn 型的实操讲解，说明 DualPipe 实际做了什么以及为什么 Sea AI Lab 的 DualPipeV 改进在牺牲略微增大的气泡的换取下，能将 2x 参数开销降为 1x。

**Type:** 学习  
**Languages:** Python（标准库，调度模拟器）  
**Prerequisites:** Phase 10 · 05（分布式训练、FSDP、DeepSpeed），Phase 10 · 14（开放模型架构与 MoE）  
**Time:** ~60 分钟

## 学习目标

- 列出 DualPipe 前向-反向 chunk 的四个组成部分，以及为什么每个部分都拥有自己的重叠窗口。
- 解释大规模下的流水线气泡问题，以及“无气泡”（bubble-free）在实际技术含义上和市场宣传上的差别。
- 手动追踪 8 个 PP ranks 与 16 个微批次的 DualPipe 调度，并确认前向与反向流填补彼此的空闲时隙。
- 说明 DualPipeV（Sea AI Lab，2025）所做的权衡：在 Expert Parallelism 不活跃时，舍弃 2 倍参数复制以换取略微增大的气泡。

## 问题描述

在 2k H800 GPU 上训练 671B MoE 模型会遇到三个叠加的瓶颈：

1. **内存压力。** 每个 GPU 保存模型的一段切片。序列长度 8k、61 层、128 个头下的激活内存极大。
2. **流水线气泡。** 传统流水线并行（GPipe、1F1B）会在等待阶段输入或梯度时让 GPU 空闲。在 8 个阶段时，即便使用 1F1B 调度，大约 12% 的 GPU 时间也可能成为气泡。
3. **跨节点 all-to-all。** 带专家并行（Expert Parallelism, EP）的 MoE 将专家散播到节点上。每次前向都会触发一次 dispatch 的 all-to-all 将 token 发送到专家，以及一次 combine 的 all-to-all 将结果汇总。在 2k GPU 规模下，这很容易达到 1:1 的计算与通信比率。

每个问题都有各自的解决方案：内存使用梯度检查点（gradient checkpointing）、流水线气泡用 Zero Bubble（Sea AI Lab，2023），all-to-all 用专家并行的高效通信内核。DualPipe 的目标是把这些解法协同起来。它在单个前向-反向 chunk 内重叠计算和通信，同时从流水线两端注入微批次，并利用所得调度将 all-to-all 隐藏在计算窗口中。

报告结果：在 DeepSeek-V3 的 14.8T token 训练中，几乎消除了流水线气泡，GPU 利用率超过 95%。

## 概念

### 流水线并行回顾

把 N 层模型分拆到 P 个设备。设备 `i` 保存层 `i * N/P .. (i+1) * N/P - 1`。一个微批次沿着设备 0 到 P-1 做前向，然后从 P-1 回到 0 做反向。每个设备只有在上游设备发送输出后才能开始自己的前向；只有在下游发送上游梯度后才能开始反向。

GPipe（Huang 等，2019）一次只调度一个微批次，浪费了大量 GPU 时间。1F1B（Narayanan 等，2021）为多个微批次交错前向与反向。Zero Bubble（Qi 等，2023）把反向拆成两部分——针对输入的反向（B）和针对权重的反向（W）——并调度它们来填满气泡。Zero Bubble 之后，流水线几乎紧凑。

DualPipe 是下一步改进。它在此基础上添加两项思想：

### 思想 1：chunk 分解

每个前向 chunk 拆成四个组成部分：

- **Attention。** Q/K/V 投影、注意力计算、输出投影。
- **All-to-all dispatch。** 将 token 发往其专家的跨节点通信。
- **MLP。** MoE 专家计算（专家内的 MLP）。
- **All-to-all combine。** 将专家输出带回的跨节点通信。

一个反向 chunk 则包含这些部分对应的梯度计算。DualPipe 将它们调度成：前向的 all-to-all dispatch 与下一 chunk 的 attention 计算并行；前向的 all-to-all combine 与后续 chunk 的 MLP 计算并行。

### 思想 2：双向调度

大多数流水线调度从阶段 0 注入微批次，向 P-1 流动。DualPipe 从两个端同时注入微批次。阶段 0 会看到来源于其端的前向微批次；阶段 P-1 也会看到来源于其端的前向微批次。两股流在中间相遇。

要实现这一点，设备 `i` 必须同时保存早期流水线层 `i` 和晚期流水线层 `P - 1 - i`。这就是 DualPipe 的“dual”部分：每个设备保存两个所需的模型层副本（各方向一个）。在 DeepSeek-V3 的规模下，这是 2 倍的参数复制成本。之所以可行，是因为 Expert Parallelism 已经把 MoE 专家分散得很细，使得将非专家层复制两次的额外开销相对很小。

关键在于：一个方向的前向流与另一个方向的反向流恰好在单向调度将出现气泡的地方重叠。气泡消失了。

### 手动追踪的调度示例

考虑 P = 4 ranks，8 个微批次，分为 4 个正向 / 4 个反向。时间从左向右移动；行表示设备 rank。

```
           时间 →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

读取 "F4/F5R" 的表示：rank 1 在同一时隙同时运行微批次 4 的前向（沿左向右的方向）和微批次 5 的前向（沿右向左的方向）。这就是“双向”在操作层面的含义。

在 rank 2 处交叉流更早重叠，在 rank 0 和 P-1 处较晚。在调度的稳定中间阶段，每个 rank 都在运行某个方向的前向和另一个方向的反向的重叠。计算保持繁忙。前向的 all-to-all dispatch 被反向计算遮挡（hide），前向的 all-to-all combine 被后续的前向/MLP 计算遮挡。气泡被挤压出去。

### 气泡计算

标准 1F1B 流水线气泡（每个 rank 的浪费时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 改进将其降低但并非归零。在稳定阶段，如果微批次数能被流水线深度的 2 倍整除，DualPipe 的稳定阶段可以实现零气泡。在稳定阶段外（热身和冷却），仍有一些气泡，但它不会随微批次数增长——这是论文强调的关键属性。

市场宣传中的“无气泡”意义是：气泡不会随微批次数增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）指出，只有当 Expert Parallelism 不是瓶颈时，才能实现完全零气泡；在 EP 驱动的 all-to-all 存在时，总会存在一些调度折衷。

### DualPipeV — 精简版

Sea AI Lab（2025）观察到，当 EP 通信重叠不是主要瓶颈时，2 倍参数复制是浪费。他们的 DualPipeV 将双向注入折叠成一个“V 形”调度，能够在单副本参数上运行。气泡比 DualPipe 稍大，但内存节省显著。DeepSeek 在其开源 DualPipe 实现中以 DualPipeV 作为 EP-off 模式被采纳。

权衡如下：

| Feature | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Param copies per device | 2 | 1 | 1 | 1 |
| Bubble vs micro-batches | constant | small growth | grows | grows |
| Compute-comm overlap | full | partial | minimal | partial |
| Use when | EP-heavy MoE | dense or EP-light | baseline | any pipeline |

### 对 14.8T token 训练的意义

DeepSeek-V3 的预训练在 2,048 个 H800 GPU 上消耗 14.8T token，约 2.8M GPU 小时。使用朴素的 1F1B，会有 12–15% 的时间损失在流水线气泡上——约 34–42 万 GPU 小时，相当于训练一个完整的 70B 模型的成本。DualPipe 回收了大部分这部分损失。没有内部日志很难精确量化其贡献，但论文声称平均 GPU 利用率超过 95%。

对于较小的训练（少于 1k GPU），DualPipe 可能是过度设计——流水线气泡相对于总成本较小，稠密模型训练也很少触及 all-to-all 瓶颈。对于多千 GPU 规模的前沿 MoE 训练，它实际上是必需的。

### 在堆栈中的位置

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 在 ranks 间切分模型参数；DualPipe 在 ranks 间调度计算。两者可以结合使用。
- 与 **ZeRO-3** 梯度分片兼容。两副本参数复制的账务需要与 ZeRO 的分片梯度协同。
- 需要为特定集群拓扑调优的 **自定义 all-to-all 内核**。DeepSeek 的开源内核是参考实现。

```figure
expert-capacity
```

## 使用方法

`code/main.py` 是一个流水线调度模拟器。它接受 `(P, n_micro_batches, schedule)` 并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 在稳定阶段的利用率。它是教学工具——这些数字与论文中的定性论断一致，但并不等同于生产环境下的测量加速声明。

模拟器的价值：用不同的 P 和微批次数运行，观察 1F1B 的气泡比例如何增长，而 DualPipe 则不会增长。

实际训练运行的集成注意事项：

- 选择一个能被微批次数整除的流水线并行深度（pipeline depth）。
- 确保你的专家并行（EP）拓扑支持双向 all-to-all。DeepSeek 的内核是参考实现。
- 预期第一次调试调度时会耗费一周左右时间。账务管理很琐碎。
- 监控每个 rank 的 GPU 利用率，而非仅看聚合。DualPipe 的收益来自缩紧慢进程（stragglers）。

## 交付物

本课产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规格（GPU 数量、拓扑、互连、模型形状），它会建议流水线并行策略、要使用的调度算法，以及在目标规模上的预期气泡比例。

## 练习

1. 运行 `code/main.py`：`(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)`。计算 GPU 利用率差异，并将其换算为每百万 token 训练可以恢复的 GPU 小时数。

2. 手工绘制 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。用微批次 ID 和方向标记每个时隙。找出首次不存在气泡的时隙。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）中的图 5。识别 DualPipe 前向 chunk 内 all-to-all dispatch 的重叠窗口。解释计算调度如何将其隐藏。

4. 计算 DualPipe 对于一个 70B 稠密模型（P=8 流水线阶段）和一个 671B MoE 模型（P=16 流水线阶段）所带来的 2 倍参数开销。说明为何 MoE 情况下的开销相对较小（大多数参数是专家，并在大的 EP 组中被分片）。

5. 将 DualPipe 与 Chimera（2021 年的一个竞品双向调度器）比较。根据论文第 3.4 节，指出 DualPipe 增加了 Chimera 所没有的两个具体属性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|--------|--------|
| 流水线气泡（pipeline bubble） | “每个 rank 的空闲时间” | GPU 周期被浪费，因为某个流水线阶段在等待输入或梯度 |
| 1F1B | “默认的流水线调度” | 一前一后交错调度（one forward / one backward）；DualPipe 所对比的基线 |
| Zero Bubble | “Sea AI Lab 2023” | 将反向拆成 B（输入梯度）和 W（权重梯度）；几乎完全紧缩流水线 |
| DualPipe | “DeepSeek-V3 的调度” | 双向流水线 + 计算-通信重叠；气泡不会随微批次数增长 |
| DualPipeV | “Cut-in-half” | V 形改进，放弃 2 倍参数复制以换取略大一些的气泡 |
| Chunk | “流水线工作的单位” | 一个微批次在一个流水线阶段上的一次前向或反向执行 |
| All-to-all dispatch | “把 token 发送到专家” | 将 token 路由到它们分配的 MoE 专家的跨节点通信 |
| All-to-all combine | “把专家输出带回来” | 在 MLP 之后收集专家输出的跨节点通信 |
| Expert Parallelism (EP) | “专家分布在 GPUs 上” | 在 ranks 之间分片 MoE 专家，使不同 GPU 保存不同的专家 |
| Pipeline Parallelism (PP) | “层分布在 GPUs 上” | 在 ranks 之间分片模型层；DualPipe 对此维度做调度 |
| Bubble fraction | “浪费的 GPU 时间比例” | (bubble_time / total_time)；DualPipe 将其驱向接近零 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — DualPipe 的主要参考文献  
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，包含 DualPipeV（Cut-in-half）模式  
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble 的先驱论文  
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — 指出 DualPipeV 思路的分析文章，为 DeepSeek 的 EP-off 模式提供依据  
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 对比的 1F1B 基线论文  
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — 原始流水线并行与气泡问题论文