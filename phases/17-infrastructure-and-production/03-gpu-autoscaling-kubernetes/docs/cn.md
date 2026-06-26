# GPU 在 Kubernetes 上的自动扩缩 — Karpenter、KAI Scheduler、Gang Scheduling

> 三层而非一层。Karpenter 动态调度节点（约一分内完成，比 Cluster Autoscaler 快 40%）。KAI Scheduler 负责 gang scheduling（群调度）、拓扑感知和分层队列 — 它能防止 7-of-8 的部分分配陷阱（七个节点等待、浪费资源只因缺一个 GPU）。应用层的自动缩放器（NVIDIA Dynamo Planner、llm-d 的 Workload Variant Autoscaler）基于推理专用信号进行扩缩 —— 队列深度、KV 缓存利用率 —— 而不是 CPU/DCGM 占空比。经典的 HPA 陷阱是 `DCGM_FI_DEV_GPU_UTIL` 是占空比测量：100% 既可能是 10 个请求也可能是 100 个请求。vLLM 预分配 KV 缓存内存，因此内存永远不会触发缩容。本课教你如何组合三层并避免默认的 Karpenter `WhenEmptyOrUnderutilized` 策略（它会在推理中断时终止正在运行的 GPU 作业）。

**Type:** 学习  
**Languages:** Python（stdlib，模拟队列深度自动扩缩器的玩具示例）  
**Prerequisites:** 阶段 17 · 02（Inference Platform Economics），阶段 17 · 04（vLLM Serving Internals）  
**Time:** ~75 分钟

## 学习目标

- 绘制三层自动扩缩结构图（节点供给、群体调度、应用层）并指出每层使用的工具名称。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 是 vLLM 错误的 HPA 信号，并列出两个替代信号（队列深度、KV 缓存利用率）。
- 描述群体调度（Gang scheduling）以及 KAI Scheduler 防止的部分分配失败模式（7/8 GPU 空闲）。
- 说明 Karpenter 的合并/整合策略 `WhenEmptyOrUnderutilized` 会终止正在运行的 GPU 作业，并指出 2026 年的安全替代设置。

## 问题场景

你的团队在 Kubernetes 上交付了一个 LLM 服务。你将 HPA 的信号设置为 `DCGM_FI_DEV_GPU_UTIL`。服务在工作时间内一直卡在 100% 利用率，HPA 从不扩容 —— 它已经认为你满载了。你手动增加一个副本；TTFT 下降。HPA 仍然不扩容。这个信号在“欺骗”你。

另一个问题，你使用 Cluster Autoscaler 管理节点。凌晨 2 点来了一个 1M token 的请求；集群花了 3 分钟来调度一个节点，请求超时。

再有一次，你部署了一个需要跨 2 个节点使用 8 个 GPU 的 70B 模型。集群有 7 个空闲 GPU，剩下的 1 个分布在 3 个节点上。Cluster Autoscaler 为缺失的 1 个 GPU 新增节点。七个节点空等了 4 分钟，消耗资金，而 Kubernetes 在最后一个 GPU 可用前一直无法调度完成。

三层，三种不同的失效模式。到 2026 年，GPU 感知的自动扩缩不再是“打开 HPA”。它是由节点供给、群体调度和基于应用信号的扩缩三者组成。

## 概念

### 第 1 层 — 节点供给（Karpenter）

Karpenter 监视 Pending 的 pod 并在约 45–60 秒内供给节点（Cluster Autoscaler 对 GPU 节点通常需要 90–120 秒）。它会根据 `NodePool` 约束动态选择实例类型 —— 如果你的 pod 需要 8 个 H100，而集群没有匹配节点，Karpenter 会直接供给一个合适的节点，而不是扩展现有的节点组。

合并陷阱（The consolidation trap）：Karpenter 的默认 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池来说是危险的。它会终止正在运行的 GPU 节点以迁移 pods 到更便宜或更合适大小的实例。对于推理工作负载，这意味着会驱逐运行中的请求并在新节点上重新加载 70B 模型。代价是数分钟的容量丧失和请求失败。

对 GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

允许 Karpenter 在节点真实空闲一小时后再合并，但绝不驱逐正在运行的作业。

### 第 2 层 — 群体调度（KAI Scheduler）

KAI Scheduler（项目原名 "Karp"）处理默认 kube-scheduler 无法做到的事情：

- Gang scheduling（群体调度） — 要么全部调度，要么全部不调度。一个需要 8 个 GPU 的分布式推理 pod，要么 8 个都启动，要么都不启动。否则你会遭遇部分分配陷阱：7/8 个 pod 启动、无限等待并浪费资源。
- 拓扑感知（topology awareness） — 知道哪些 GPU 共享 NVLink、哪些在同一机架、哪些节点之间有 InfiniBand。并据此放置 pods。像 DeepSeek-V3 67B 的张量并行工作负载必须保持在同一 NVLink 域内，KAI Scheduler 会遵守这些要求。
- 分层队列（hierarchical queues） — 多个团队在同一 GPU 池竞争时可配置优先级与配额。只有在优先级规则允许时，Team A 的生产流量才会被 Team B 的训练作业抢占。

KAI 通常作为一个 secondary scheduler 部署，与你的 kube-scheduler 并行；你通过注解让工作负载使用它。Ray 和 vLLM 的生产栈都支持集成 KAI。

### 第 3 层 — 应用层信号

HPA 陷阱：`DCGM_FI_DEV_GPU_UTIL` 是占空比（duty-cycle）指标 —— 它测量在每个采样区间 GPU 是否在做计算。100% 利用率既可能是 10 个并发请求，也可能是 100 个；GPU 在两种情况下都被视为“忙”。基于占空比进行扩缩是在盲目扩缩。

更糟的是，vLLM 等引擎会预分配 KV 缓存内存（通过 `--gpu-memory-utilization` 控制）。内存使用会在只有一个请求时也接近 90%，因此基于内存的 HPA 永远不会触发缩容。

2026 年的替代信号：

- 队列深度（等待 prefill 的请求数）。
- KV 缓存利用率（有多少块被分配给活跃序列）。
- 每副本的 P99 TTFT（你的 SLA 信号）。
- Goodput（满足所有 SLO 的每秒请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并扩缩副本。它们在 LLM 服务场景中取代了传统 HPA。

### 何时使用哪一层

| Scale decision | Tool |
|----------------|------|
| 增删节点 | Karpenter |
| 调度多 GPU 作业 | KAI Scheduler |
| 增删副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 先占并驱逐低优先级 | KAI Scheduler 队列 |

### 异构分离的 prefill/decode 增加复杂性

如果你运行分离式（disaggregated）prefill/decode（参考阶段 17 · 17），你会有两类 pod，触发扩缩的信号不同：prefill pod 基于队列深度扩缩，decode pod 基于 KV 缓存压力扩缩。llm-d 将这些作为不同的 `Service` 暴露，并为每个角色配置 HPA。不要试图用一个单一的 HPA 同时覆盖两类角色。

### 冷启动同样重要

冷启动缓解（阶段 17 · 10）会让节点供给时间直接影响用户体验。Karpenter 的 45–60 秒预热加上 20GB 模型加载和引擎初始化，意味着从零到响应可能需要 2–5 分钟。对于 SLO 关键路径，保持一个热池（例如 `min_workers=1`），或者在应用层使用类似 Modal 的 checkpointing。

### 你应该记住的数字

- Karpenter 节点供给：约 45–60s，Cluster Autoscaler：约 90–120s（GPU 节点）。
- KAI Scheduler 防止部分分配浪费 —— 7-of-8 陷阱。
- 将 `DCGM_FI_DEV_GPU_UTIL` 用作 HPA 信号是错误的；使用队列深度或 KV 利用率。
- Karpenter 的 `WhenEmptyOrUnderutilized` 会终止正在运行的 GPU 作业。对推理使用 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## 使用方法

`code/main.py` 模拟了一个三层自动扩缩器在突发 GPU 负载下的表现。比较了天真的 HPA（占空比）、基于队列深度的 HPA，以及启用 KAI 群体调度的扩缩。报告未满足的请求数、空闲 GPU 分钟数以及一个综合评分。

## 交付结果

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载特性和 SLO，它会设计一个三层自动扩缩方案。

## 练习

1. 运行 `code/main.py`。在突发负载下，天真的占空比 HPA 丢弃了多少请求，而基于队列深度的 HPA 捕获了这些请求？差异来自哪里？
2. 为集群设计一个用于 Llama 3.3 70B FP8 在 H100 SXM5 上服务的 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个用于将非 GPU 工作负载排除在这些节点之外的 taint。
3. 你的团队报告部署卡在 Pending 状态并提示“GPUs available but pod won't schedule”。诊断——这是 Karpenter、kube-scheduler 还是 KAI Scheduler 的问题？哪些指标可以确认？
4. 为分离式 prefill pods 选择一个自动扩缩信号，为 decode pods 选择另一个。说明你的理由。
5. 计算 `WhenEmptyOrUnderutilized` 合并陷阱对一个 24x7 生产服务的成本，假设平均每天有 60 次请求丢失事件且 P99 TTFT > 10s。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|----------|
| Karpenter | "the node provisioner" | Kubernetes 节点自动供给器；亚分钟级别的节点供给 |
| Cluster Autoscaler | "the old scaler" | Kubernetes 节点自动扩缩的前身；较慢、基于节点组 |
| KAI Scheduler | "the GPU scheduler" | 用于群体调度 + 拓扑感知 + 队列管理的二级调度器 |
| Gang scheduling | "all or nothing" | 群体调度（Gang scheduling）：要么原子性地调度 N 个 pod，要么都延后 |
| Topology awareness | "rack-aware" | 基于 NVLink/IB/机架放置 pod 的拓扑感知 |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU utilization" | 占空比指标；不是 LLM 的扩缩信号 |
| Queue depth | "waiting requests" | 等待请求数；prefill 绑定扩缩的正确信号 |
| KV cache utilization | "memory pressure" | KV 缓存利用率；decode 绑定扩缩的正确信号 |
| Consolidation | "Karpenter consolidation" | Karpenter 的节点合并（终止节点以迁移到更便宜的实例） |
| `WhenEmpty + 1h` | "safe consolidation" | 不会驱逐正在运行 GPU 作业的安全策略 |

## 拓展阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置示例。  
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — 合并策略语义和 GPU 安全默认配置。  
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 的扩缩信号说明。  
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 的集成示例。  
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 针对托管 Kubernetes 的最佳实践指南。  
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 的设计。