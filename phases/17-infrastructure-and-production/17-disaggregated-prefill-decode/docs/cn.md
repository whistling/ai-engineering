# 分离式 Prefill/Decode — NVIDIA Dynamo 与 llm-d

> Prefill（预填充）受计算约束；decode（解码）受内存带宽约束。在同一 GPU 上同时运行两者会浪费其中一种资源。分离式架构将它们放到不同的资源池中，并通过 NIXL（RDMA/InfiniBand 或 TCP 回退）在它们之间传输 KV 缓存。NVIDIA Dynamo（GTC 2025 发布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上 — 其 Planner Profiler + SLA Planner 能自动按比率匹配 prefill:decode 以满足 SLO。NVIDIA 在这个范围内公布了吞吐量提升 —— developer.nvidia.com（2025-06）在中等延迟场景下对 GB200 NVL72 + Dynamo 的 DeepSeek-R1 MoE 报告了大约 6 倍的改进，Dynamo 产品页面（developer.nvidia.com，未注明日期）宣称在 GB300 NVL72 + Dynamo 上对比 Hopper 可实现高达 50 倍的 MoE 吞吐。社区所说的“30x”是基于完整栈 Blackwell + Dynamo + DeepSeek-R1 的汇总；我们未找到单一原始出处精确说是 30x，因此将其视为方向性说法。llm-d（Red Hat + AWS）是 Kubernetes 原生：将 prefill / decode / router 作为独立服务并为每个角色设置 HPA。llm-d 0.5 增加了分层 KV 卸载、缓存感知的 LoRA 路由、UCCL 网络以及 scale-to-zero。经济学：多家客户披露的内部汇总显示，在保持相同 SLA 的情况下，从同置服务切换到使用 Dynamo 的分离式架构，年推理开销为 $2M 级别时可节省约 30–40%（即每年约 $600–800K）；这个 $2M→$600–800K 的数字是内部合成数据，不是单一公开案例研究 — 作为数量级参考而非引用文献。短提示（<512 令牌、短输出）没有理由承担传输成本。

**Type:** 学习  
**Languages:** Python（标准库，示例性离散化 vs 同置 模拟器）  
**Prerequisites:** Phase 17 · 04（vLLM 服务内部原理）、Phase 17 · 08（推理指标）  
**Time:** ~75 分钟

## 学习目标

- 解释为什么 prefill 和 decode 需要不同的 GPU 分配，并量化同置时的资源浪费。
- 绘制分离式架构图：prefill 池、decode 池、通过 NIXL 的 KV 传输、路由器。
- 指出在何种条件下分离式不划算（短提示、短输出）。
- 区分 NVIDIA Dynamo（位于栈上）与 llm-d（Kubernetes 原生），并将二者匹配到各自的运维场景。

## 问题

你在 8 块 H100 上运行 Llama 3.3 70B。在混合负载（长提示 + 短输出）下，GPU 在解码阶段空闲，因为大部分计算集中在 prefill。换一种负载（短提示 + 长输出），则相反。prefill 与 decode 同置意味着你会对两者同时进行过度配置。

预算影响：20–40% 的 GPU 时间可能被错误资源占用而浪费。你可能为了运行内存带宽受限的 decode 而购买了更多计算能力的 H100，或者为了运行计算密集的 prefill 而购买了有更高 HBM 带宽的 H100。两种情况都造成昂贵的浪费。

分离式将 prefill 和 decode 拆到各自针对瓶颈进行规模化的池中。KV 缓存从 prefill 池通过高带宽互连传到 decode 池。

## 概念

### 为什么瓶颈不同

Prefill（预填充）——对整个输入提示进行一次 Transformer 前向计算。矩阵乘法占主导；受计算约束。H100 的 FP8 在有用吞吐量上可达约 2000 TFLOPS。批量效率较好——一次前向可以处理多个令牌。

Decode（解码）——逐个令牌生成，每次迭代都要读取完整权重。受内存带宽约束。HBM3 的带宽约为 3 TB/s。仅在高并发下批量效率良好——权重读取成本可以在批次中摊销。

同置：你需要购买同时适配两种瓶颈的 GPU。H100 两方面表现都不错，但成本相同。在规模化场景中，你希望 prefill 池使用以计算为主的 H100，decode 池使用以内存为主的 H200 或结合激进量化的硬件/配置。

### 架构

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的节点间传输。优先使用 RDMA/InfiniBand，无法时回退到 TCP。传输延迟是实实在在的 —— 对 70B FP8，4K 令牌提示的 KV 缓存通常需要约 20–80 ms。这也是为什么短提示不适合分离式的原因：传输开销超过了收益。

### Dynamo 与 llm-d 的区别

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：
- 位于 vLLM、SGLang、TRT-LLM 之上，作为编排器。
- Planner Profiler 测量工作负载，SLA Planner 自动配置 prefill:decode 比率。
- Rust 核心，支持 Python 可扩展性。
- 吞吐量提升：NVIDIA 报告在中等延迟场景下，DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上约为基线的 6 倍（developer.nvidia.com，2025-06）；社区关于在完整 Blackwell + Dynamo 堆栈上“高达 30x”的说法是汇总性方向性数据，缺乏单一原始来源。
- GB300 NVL72 + Dynamo：Dynamo 产品页称对比 Hopper 可在 MoE 上实现高达 50 倍吞吐（developer.nvidia.com，未注明日期）。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- 将 prefill / decode / router 作为独立的 Kubernetes Service。
- 基于角色的 HPA，使用队列深度（prefill）/ KV 使用率（decode）等信号。
- `topologyConstraint packDomain: rack` 将 prefill+decode 团簇打包到同一机架以获得高速 KV 传输。
- llm-d 0.5（2026）：引入分层 KV 卸载、缓存感知的 LoRA 路由、UCCL 网络层、scale-to-zero。

如果你想要一个由上层栈管理的编排器，选用 Dynamo。如果你需要 Kubernetes 原生原语并且已承诺使用 CNCF 生态，选用 llm-d。

### 经济学

内部合成（非单一公开案例研究 — 用作数量级参照）：

- 同置服务下年推理开销约 $2M。
- 切换到使用 Dynamo 的分离式架构。
- 请求量相同，P99 延迟 SLA 不变。
- 报告的节省：每年 $600K–$800K（节省 30–40%）。
- 无需新增硬件。

我们将这一数字合成自多家客户披露，而非单个可引用的案例；接近的公开数据点包括 Baseten 在 Dynamo KV 路由下实现 2x 更快 TTFT / 61% 更高吞吐（baseten.co，2025-10），以及 VAST + CoreWeave 在 40–60% KV 命中率下预测的每美元令牌数提升 60–130%（vastdata.com，2025-12）。节省来自于对每个池的正确右配；prefill 密集型工作负载（如带 8K+ 前缀的 RAG）比平衡负载受益更多。

### 何时不应分离

- 提示 < 512 令牌且输出 < 200 令牌：传输开销压倒收益。
- 小型集群（< 4 GPU）：池的多样性不足。
- 团队无法操作两个具有按角色伸缩的 GPU 池：Dynamo 可以帮忙，但运维并非完全无痛。
- 无 RDMA 网络：TCP 的传输成本更高。

### 路由器与 Phase 17 · 11 的集成

分离式路由器对 KV 缓存是感知的（见 Phase 17 · 11）。请求会落到持有其前缀的 decode 池上 —— 若未命中，则会流经 prefill → decode。命中率与分离策略相互影响 —— 缓存感知路由器决定是否真的需要新的 prefill。

### Blackwell 上的 MoE 是真实数据的来源

GB300 NVL72 + Dynamo 在 MoE 上相对于 Hopper 基线显示 50x 的吞吐提升。MoE 的专家路由在 prefill 阶段计算密集，而在 decode 阶段对内存带宽和专家缓存敏感，因此分离式是双重收益。2026 年的前沿模型服务以 MoE 为主（如 DeepSeek-V3、未来的 GPT-5 变体）。

### 你应该记住的数字

基准数据会变动 —— NVIDIA 和推理栈会定期更新结果。引用前请复查。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo：中等延迟场景下约 6 倍吞吐提升（developer.nvidia.com，2025-06）；社区关于完整 Blackwell + Dynamo 堆栈“高达 30x”的说法为方向性汇总，缺乏单一原始出处。
- GB300 NVL72 + Dynamo：MoE 上相对于 Hopper 可达 50x（developer.nvidia.com，未注明日期）。
- 节省锚点（内部合成，非单一案例研究）：在 $2M 年度支出下，保持 SLA 不变可以节省约 $600–800K/年。
- 分离阈值：提示 >512 令牌且输出 >200 令牌。
- 通过 NIXL 传输 KV：在 70B FP8 上，4K 提示的 KV 传输大约 20–80 ms。

## 使用示例

`code/main.py` 模拟了同置与分离式服务。会报告吞吐量、每请求成本以及提示长度的交叉点。

## 发布产物

本课件会生成 `outputs/skill-disaggregation-decider.md`。给定工作负载与集群，决定是否应该分离式部署。

## 练习

1. 运行 `code/main.py`。在什么提示长度下分离式开始优于同置？
2. 为一个 RAG 服务（P99 前缀长度 8K，输出 300）设计 prefill 池与 decode 池。
3. Dynamo vs llm-d：在一个纯 Kubernetes 团队且没有 Python 运行时偏好时，选哪个？
4. 计算 KV 传输成本：4K prefill 在 70B FP8 上 ≈ 500 MB KV。在 RDMA 100 GB/s 下传输 = 5 ms。在 TCP 10 GB/s 下 = 50 ms。哪个对你的 SLA 有影响？
5. MoE 的专家路由会改变 KV 访问模式。对于每个令牌激活不同专家的 MoE，分离式如何表现？

## 关键术语

| 术语 | 常说法 | 实际含义 |
|------|--------|----------|
| Disaggregated serving | “split prefill/decode” | 将预填充/解码拆到独立的 GPU 池 |
| NIXL | “NVIDIA transport” | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | “the orchestrator” | 位于 vLLM/SGLang/TRT-LLM 之上的协调器 |
| llm-d | “Kubernetes native” | Red Hat + AWS 的 K8s 分离式栈 |
| Planner Profiler | “Dynamo auto-config” | 测量工作负载，配置池比率 |
| SLA Planner | “Dynamo policy” | 自动按速率匹配 prefill:decode 以满足 SLO |
| `packDomain: rack` | “llm-d topology” | 将 prefill+decode 打包在同机架以加速 KV 传输 |
| UCCL | “unified collective” | llm-d 0.5 的网络层，实现 scale-to-zero |
| MoE expert routing | “expert per token” | DeepSeek-V3 的模式；分离式架构有助于处理 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)