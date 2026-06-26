# Multi-Region LLM Serving and KV Cache Locality

> 轮询负载均衡（round-robin）对带缓存的 LLM 推理实际上有害。一个没有落到持有其前缀的节点上的请求需要支付完整的预填充成本——在长提示下 P50 大约为 800 ms，而缓存命中约为 ~80 ms。到 2026 年，生产模式是缓存感知路由器（vLLM Router 用 Rust 实现、llm-d router），它消费 KV-cache 事件并在前缀哈希匹配时进行路由。近期研究（GORGO）将跨区域网络延迟作为路由目标中的显式项。商业性的“跨区域推理”产品（Bedrock cross-region inference、GKE multi-cluster gateways）把推理当作不透明体处理——它们处理可用性，而不是首个令牌时间（TTFT）。JPMorgan 和 Mayo Clinic 在 2024 年 11 月进行了 us-east-1 的故障切换演练，耗时约 22 分钟。灾难恢复（DR）现实是：32% 的 LLM 灾难恢复失败是因为团队备份了权重却忘记了 tokenizer 文件或量化配置。

**Type:** 学习  
**Languages:** Python (stdlib, toy prefix-cache-aware router simulator)  
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)  
**Time:** ~60 分钟

## 学习目标

- 说明为什么轮询负载均衡会破坏带缓存的推理并量化 TTFT 的惩罚。  
- 画出一个缓存感知路由器的图：输入（KV-cache 事件）、算法（前缀哈希匹配）、平局决胜（GPU 利用率）。  
- 说出导致 32% 灾难恢复失败的驱动因素（缺失 tokenizer 文件 / 量化配置），并列出三文件的 DR 清单。  
- 区分商业跨区域产品（Bedrock CRI、GKE Multi-Cluster Gateway）与 KV 感知路由。

## 问题描述

你的服务在 us-east-1、us-west-2 和 eu-west-1 运行。你在前端放了一个 ALB 并使用轮询。生产中的前缀缓存命中率下降到 8%。TTFT P50 翻三倍。你的 vLLM 日志显示每个请求都在支付完整的预填充成本。

轮询对于无状态服务是最优的。LLM 推理本质上是有状态的——KV 缓存编码了模型所见的一切。盲目路由就是路由到错误的缓存。

另外，你的团队有一个灾难恢复计划。你把模型权重跨区域备份到 S3。区域性故障发生；你尝试故障切换；副本拒绝启动。你忘记了 tokenizer.json、量化配置和 RoPE 缩放配置在另一个没有同步的桶里。

多区域 LLM 服务是一个缓存问题、路由问题和 DR 卫生问题——不是传统负载均衡器的问题。

## 概念

### 缓存感知路由

请求携带一个提示到达。路由器对前缀（例如前 512 个 token）做哈希；它询问每个副本“你有这个前缀的缓存吗？”。副本在分发/订阅通道上发布 KV-cache 事件，说明它们何时分配和驱逐块。路由器选择匹配的副本；如果没人命中，则回退到基于 GPU 利用率的平局决胜。

vLLM Router（Rust，2026 生产栈）：订阅 `kv.cache.block_added` 事件，维护前缀哈希 → 副本索引的映射，并以 O(1) 查找进行路由。若无匹配则回退到最小队列深度（least-queue-depth）。

llm-d router：相同模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

SGLang RadixAttention（Phase 17 · 06）是副本内部的等价物。跨副本路由是彻底的上游问题。

### 数字说明

在 2K token 提示、Llama 3.3 70B FP8、H100 上的 TTFT：
- 缓存命中（同一副本、前缀在位）：~80 ms。  
- 缓存未命中（冷预填充）：~800 ms。

10 倍差距。如果你的路由器在副本之间命中 60–80% 的前缀缓存，你可以近似得到单副本性能在 N 副本容量下的效果。如果命中率为 10%，则近似天真的扩展效果。

### 跨区域的新约束——网络延迟

区域间 RTT：
- us-east-1 ↔ us-west-2：~65 ms。  
- us-east-1 ↔ eu-west-1：~75 ms。  
- us-east-1 ↔ ap-southeast-1：~220 ms。

如果路由将 us-east-1 的请求转到 ap-southeast-1 上的热前缀，节省的预填充时间（800 → 80 ms）会被 440 ms 的往返延迟吞没。GORGO（2026 年研究）将此显式化——应联合最小化 `prefill_time + network_latency`，而不是只考虑 prefill。通常结论是保持区域内路由，除非是巨大的多 MB 前缀，在那种情况下预填充时间占优。

### 商业“跨区域推理”对这个问题无解

AWS Bedrock cross-region inference 会在容量压力时自动将请求路由到其他区域。它优化的是可用性而不是 TTFT，并把推理当作不透明操作。GKE Multi-Cluster Gateway 同理——服务级别的故障切换，不考虑 KV 缓存。

即使使用这些服务，你仍然需要应用层的缓存感知路由器。它们处理“us-east-1 完全不可用”的场景；缓存感知路由处理的是 TTFT 优化。

### 灾难恢复卫生——32% 的缺失文件问题

2026 年广泛引用的数据：32% 的 LLM 灾难恢复失败发生是因为团队备份了权重却忘记了：

- `tokenizer.json` 或 `tokenizer.model`  
- 量化配置（`quantize_config.json`、AWQ 缩放参数、GPTQ 零点）  
- 模型特定配置（RoPE 缩放、注意力掩码、聊天模板）  
- 引擎配置（`vllm_config.yaml`、采样默认值、LoRA 适配器清单）

修复方法是一个最少包含三类文件的 DR 清单：

1. HF 模型仓库下的所有文件（权重 + 配置 + tokenizer）。  
2. 引擎特定的服务配置。  
3. 部署清单（K8s YAML、Dockerfile、依赖锁文件）。

另外：每季度进行一次 DR 演练。JPMorgan 在 2024 年 11 月的 us-east-1 演练只用时 22 分钟恢复，是因为运行并排练了演练手册。

### 数据驻留（Data residency）是正交问题

欧盟客户的 PHI 不能离开欧盟。如果你的缓存感知路由器把巴黎发起的请求为了前缀匹配而发送到 us-east-1，你就违反了 GDPR，无论 TTFT 是否有提升。在优化缓存之前，先按驻留边界对路由器进行分区。

### 应记住的数字

- 缓存命中与未命中的 TTFT 差距：约 10 倍（2K 提示下 80 ms vs 800 ms）。  
- 区域间 RTT 美欧：~75 ms。  
- DR 失败：32% 忘记 tokenizer/量化配置。  
- JPMorgan us-east-1 故障切换 2024 年 11 月：22 分钟（30 分钟 SLA）。

## 使用说明

`code/main.py` 模拟了三种路由策略（轮询、区域缓存感知、全局缓存感知）在多区域工作负载下的表现。会报告缓存命中率、TTFT P50/P99 和跨区域计费。

## 交付物

本课程产出 `outputs/skill-multi-region-router.md`。给定区域、驻留约束和 SLA，设计一个路由方案。

## 练习

1. 运行 `code/main.py`。在 75 ms RTT 的前提下，提示长度在什么程度上跨区域路由会优于仅本地路由？  
2. 你的缓存命中率从 70% 跌到 12%。诊断三个可能的原因并列出可以确认每种原因的可观测项。  
3. 为在 vLLM 中服务的 70B AWQ 量化模型（含 5 个 LoRA 适配器）设计一个 DR 清单。列出每个文件和配置。  
4. 阐述 Bedrock cross-region inference 对于对 TTFT SLO 有严格要求的金融科技公司是否“足够”。引用具体行为。  
5. 一条来自巴黎的请求在 us-east-1 匹配到了前缀。你会路由过去吗？写出该策略。

## 术语表

| 术语 | 大家怎么说 | 实际是什么意思 |
|------|-----------|----------------|
| Cache-aware routing | “smart LB” | 基于前缀哈希匹配路由到持有 KV-cache 的副本（缓存感知路由） |
| KV-cache events | “cache pub-sub” | 副本发布块添加/驱逐事件；路由器建立索引 |
| Prefix hash | “cache key” | 用作路由查找的前 N 个 token 的哈希 |
| GORGO | “cross-region routing research” | arXiv 2602.11688；将网络延迟作为显式项纳入考量 |
| Cross-region inference | “Bedrock CRI” | AWS 产品；实现可用性故障切换，但不考虑 TTFT |
| DR manifest | “the backup list” | 恢复所需的所有文件清单——不仅仅是权重 |
| Data residency | “GDPR boundary” | 关于哪些区域可以看到用户数据的法律约束 |
| RTT | “round-trip time” | 网络往返时延；美欧约 75 ms，美亚约 220 ms |
| LLM-aware LB | “cache-hit LB” | 一类产品：缓存感知路由器 |

## 延伸阅读

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)  
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 带网络延迟项的跨区域 KV-cache 重用研究。  
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)  
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 可用性故障切换文档。  
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — 缓存感知路由器源码。