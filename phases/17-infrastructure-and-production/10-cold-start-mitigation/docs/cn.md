# Cold Start Mitigation for Serverless LLMs

> 一个 20 GB 的模型镜像从冷启动到可提供服务需要 5-10 分钟（7B）到 20+ 分钟（70B）。在真正的无服务器世界里，这不是“预热”——这是一次宕机。缓解措施在五个层面上运作：预填充节点镜像（AWS 的 Bottlerocket，双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer，vLLM 原生支持）、GPU 内存快照（Modal 检查点，重启速度最高可快 10 倍）、预热池（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM 管道，延迟降低 10-200 倍），以及将输入令牌（KB）而不是 KV 缓存（GB）迁移的实时迁移。Modal 将冷启动下限公布为 2-4 秒；Baseten 默认 5-10 秒，预热后可达亚秒。本课将教你如何测量、预算并组合这五个层面。

**Type:** 学习  
**Languages:** Python (stdlib，玩具冷启动路径模拟器)  
**Prerequisites:** Phase 17 · 02 (推理平台经济学), Phase 17 · 03 (GPU 自动伸缩)  
**Time:** ~60 分钟

## Learning Objectives

- 列举冷启动缓解的五个层面，并为每个层面命名一个工具或模式。
- 将总冷启动时间计算为（节点供给） +（权重下载）+（权重加载到 HBM）+（引擎初始化）的和，针对一个 70B 模型进行计算。
- 解释为什么实时迁移传输输入令牌（KB）而不是 KV 缓存（GB），以及其代价是什么（重新计算）。
- 指出预热池的权衡（为空闲 GPU 付费或接受冷启动尾延迟），以及在何种 SLA 阈值下 `min_workers > 0` 成为强制性。

## The Problem

你的无服务器 LLM 端点在夜间缩容到 0。早上 8 点流量激增。第一个请求需要等待：

1. Karpenter 供应一个 GPU 节点：45-60 秒。
2. 容器拉取一个包含权重的 30 GB 镜像：120-300 秒。
3. 引擎将权重加载到 HBM：45-120 秒，取决于模型大小和存储速度。
4. vLLM 或 TRT-LLM 初始化 CUDA 图、KV 缓存池、分词器：10-30 秒。

总计：220-510 秒（大致 3-8 分钟）才会返回第一个令牌。你的 SLA 是 2 秒。你通过部署预热池（`min_workers=1`）来缓解，问题似乎消失了——但现在你要为一台空闲 GPU 24×7 付费。如果你的服务有 5 个产品，每个都有一个预热副本，那就是 5 × 24 × 30 = 3,600 GPU 小时/月，无论是否有用户调用。

冷启动缓解的目标是在保持无服务器经济性的同时，尽可能接近始终在线的延迟表现。

## The Concept

### Layer 1 — pre-seeded node images (Bottlerocket)

在 AWS 上，Bottlerocket 的双卷架构将操作系统与数据分离。将包含已预拉取容器镜像的数据卷做快照；在你的 `EC2NodeClass` 中引用该快照 ID。新节点启动时，权重已经在本地 NVMe 上——步骤 2 以及部分步骤 3 消失。与 Karpenter 原生兼容。典型节省：大型模型每次冷启动减少 2-4 分钟。

在 GCP 等价为：预先打包容器层的自定义 VM 镜像。在 Azure 上则可使用托管磁盘快照并采用相同模式。

### Layer 2 — model streaming (Run:ai Model Streamer)

无需在回答第一个请求前加载完整文件，可以按层将权重流式加载到 GPU 内存，并在第一个 Transformer 块就位后开始处理。NVIDIA 的 Run:ai Model Streamer 已在 vLLM 2026 版本中原生集成。支持 S3、GCS 和本地 NVMe。通过将 I/O 与计算设置重叠，大模型的权重加载时间大约减半。

### Layer 3 — GPU memory snapshots (Modal)

Modal 在首次加载后对 GPU 状态（权重、CUDA 图、KV 缓存区域）做检查点。后续重启直接反序列化到 HBM——比重新初始化快 10 倍。这是最接近“2 秒内启动一个预热 GPU”的方法。权衡：快照与 GPU 拓扑绑定，因此如果 Karpenter 将你迁移到不同 SKU，就需要重新做检查点。

### Layer 4 — warm pools (min_workers=1)

最简单的缓解措施：始终保留一个副本就绪。成本就是一台 GPU 的小时费率 24×7。对小模型而言代价惨烈（为避免 30 秒冷启动，你需支付 $0.85-$1.50/小时），对大模型则相对划算（为避免 5 分钟冷启动支付 $4/小时）。预热池成为必须的 SLA 阈值：通常在 70B+ 模型上 TTFT P99 < 60 秒 时需要 `min_workers > 0`。

### Layer 5 — tiered loading (ServerlessLLM)

ServerlessLLM 将存储视为层级：NVMe（快但大）、DRAM（中等但可分层）、HBM（小但瞬时）。权重预加载到 DRAM，根据需求再加载到 HBM。论文报告与天真的磁盘到 HBM 加载相比，冷加载延迟降低 10-200 倍。生产采用仍在早期，但已与 vLLM 存在集成。

### Layer 6 — live migration (bonus pattern)

当节点不可用（抢占、节点驱逐）时，传统模式是冷启动另一个副本并清空请求队列。实时迁移则把输入令牌（千字节级）移动到已加载模型的目标节点，并在目标上重新计算 KV 缓存。重新计算通常比网络传输 GB 级的 KV 缓存便宜。适用于解耦部署（disaggregated deployments）。

### The warm-pool math

对于 P99 TTFT SLA 为 2 秒的服务，问题不在于“要不要预热池”，而在于“要多少预热副本，以及为哪些路径保留它们”。

- 高价值交互路径（实时聊天、语音代理）：`min_workers=1-2`。
- 后台批处理路径（夜间分类）：接受缩容到 0，5-10 分钟冷启动可容忍。
- 高级付费层：为每个租户配置 `min_workers` 的专用容量。

### Measure before optimizing

70B 模型在新节点上的冷启动结构（示例）：

| Phase | Time | Mitigation |
|-------|------|-----------|
| Node provision | 50s | Bottlerocket + pre-seeded image, warm pool |
| Image pull | 180s | Pre-seeded data volume (eliminate) |
| Weights to HBM | 75s | Model streamer (halve); GPU snapshot (eliminate) |
| Engine init | 20s | Persistent CUDA graph cache |
| First forward | 3s | Min inherent latency |
| **Total cold** | **328s** | |
| **Total with mitigations** | **~15s** | 22x reduction |

### Numbers you should remember

- Modal 冷启动：2-4 秒（使用 GPU 快照）。
- Baseten 默认冷启动：5-10 秒；预热后可达亚秒级。
- 原始 70B 冷启动：3-8 分钟。
- Run:ai Model Streamer：权重加载速度约提升 ~2 倍。
- ServerlessLLM 分层加载：冷加载延迟降低 10-200 倍（论文数据）。

## Use It

`code/main.py` 模拟带或不带每种缓解措施的冷启动路径。报告总冷启动时间、预热池成本，以及预热池比自行承担冷启动成本划算的盈亏平衡请求率。

## Ship It

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型大小和流量形态，选择应当叠加的缓解措施。

## Exercises

1. 运行 `code/main.py`。计算当预热副本比通过额外请求丢弃来支付冷启动税更划算时的盈亏平衡请求率。
2. 你部署了一个 13B 模型，P99 TTFT SLA 为 3 秒。选择能达到该目标的最小缓解栈（最少层数）。
3. Bottlerocket 预填充消除了镜像拉取，但权重仍需从快照加载到 HBM。若 70B 模型的快照支撑 NVMe 读取速率为 7 GB/s，计算墙钟时间。
4. 你的无服务器提供商提供 GPU 快照（Modal），但你的团队拒绝使用，理由是“快照会泄露 PII（个人可识别信息）”。请双方论证——现实风险是什么，缓解方法有哪些（一次性快照、加密、命名空间隔离）？
5. 设计一个分层的预热池策略：为付费用户、试用用户和批处理工作分别保留多少个预热副本？给出计算过程。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Cold start | "the big pause" | 从请求到在新副本上返回第一个令牌的时间 |
| Warm pool | "always-on minimum" | 使用 `min_workers >= 1` 保持至少一个副本就绪 |
| Pre-seeded image | "baked AMI" | 节点镜像中已驻留容器权重 |
| Bottlerocket | "AWS node OS" | AWS 的容器优化操作系统，支持双卷快照 |
| Model streamer | "streaming load" | 将权重 I/O 与计算设置重叠的流式加载 |
| GPU snapshot | "checkpoint to HBM" | 在加载后序列化 GPU 状态；重启时反序列化到 HBM |
| Tiered loading | "NVMe + DRAM + HBM" | 存储层级；按需加载到更快的层 |
| Live migration | "move tokens" | 迁移输入（KB），在目标上重新计算 KV |
| `min_workers` | "warm replicas" | Serverless 的最小保活副本数 |
| Scale-to-zero | "full serverless" | 空闲时无成本；接受完整的冷启动成本 |

## Further Reading

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal 发布的基准和检查点架构文档。  
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预填充数据卷快照模式。  
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 将权重加载与计算设置重叠的实现。  
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — 预热应对方案。  
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计。  
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 针对解耦部署的实时迁移方案。