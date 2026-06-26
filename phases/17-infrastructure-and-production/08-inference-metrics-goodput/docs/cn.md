# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定推理部署是否可用。TTFT 是 prefill（预填充）加上队列延迟加上网络延迟。TPOT（等价于 ITL）是按 token 计算的内存受限解码成本。端到端延迟是 TTFT 加上 TPOT 乘以输出长度。吞吐率是整个集群每秒处理的 token 数。但对产品来说真正重要的是 goodput — 同时满足所有 SLO 的请求比例。高吞吐但低 goodput 意味着你在处理很多最终没有按时到达用户的 token。2026 年在 TRT-LLM 上对 Llama-3.1-8B-Instruct 的参考值：平均 TTFT 162 ms，平均 TPOT 7.33 ms，平均端到端 1,093 ms。始终报告 P50、P90、P99 — 切勿只报均值。并注意测量陷阱：GenAI-Perf 在 ITL 计算中排除了 TTFT，而 LLMPerf 则包含它；两个工具会对同一次运行的 TPOT 给出不同结果。

**Type:** 学习  
**Languages:** Python (stdlib、玩具百分位数计算器和 goodput 报告器)  
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)  
**Time:** ~60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐率 和 goodput，并指出每个指标衡量的组件。
- 解释为什么均值不是 LLM 服务的正确统计量以及如何阅读 P50/P90/P99。
- 构造一个 SLO 多约束（例如 TTFT<500 ms 且 TPOT<15 ms 且 E2E<2 s），并针对其计算 goodput。
- 列出两个对相同运行在 TPOT 上存在分歧的基准工具并解释原因。

## 问题

“我们的吞吐率是 15,000 token/s。”那又怎样？如果 40% 的请求端到端超过了 2 秒，用户就会放弃会话。仅有吞吐率并不能告诉你产品是否可用。

推理存在多条延迟轴，每一类延迟的失败模式不同。Prefill（预填充）是计算受限并随提示长度线性增长。Decode（解码）是内存受限并随批量大小变化。排队延迟是运营问题。网络是物理距离问题。你需要针对每一个分别度量，并需要百分位数，还需要一个表明“用户是否如期得到结果”的复合指标 — 这就是 goodput。

## 概念

### TTFT — time to first token

`TTFT = queue_time + network_request + prefill_time`

当提示很长时，prefill 占主导。在 Llama-3.3-70B FP8 在 H100 上，32k 的提示纯 prefill 大约需要 ~800 ms。队列时间是调度器在高负载下的行为。网络请求是含 TLS 的线网上传/下发时间。TTFT 是用户在任何内容流回之前看到的延迟。

### TPOT / ITL — inter-token latency

同一数量的多种名称。`TPOT`（每输出 token 时间）、`ITL`（token 间延迟）、`decode latency per token` — 通常都是同一概念。它是第一个 token 之后连续流式 token 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在相同的 Llama-3.3-70B H100 堆栈并使用 chunked prefill（分片预填充）时，TPOT 的均值约为 ~7 ms。若不使用分片预填充，在邻接序列进行长预填充期间，TPOT 可能飙升到 50 ms。关注 P99，而不是均值。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 token），E2E 由 TPOT 主导。对于提示很长且输出短的场景，E2E 由 TTFT 主导。请按输出长度分组报告 E2E。

### 吞吐率

`throughput = total_output_tokens / elapsed_time`

聚合指标。告诉你集群效率。不能反映单个请求的健康状况。

### Goodput — 你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是一个多约束。只有当每一条约束都满足时，请求才被认为是“良好”的。高吞吐但 60% goodput 就是失败。目标是较低吞吐但 >=99% goodput。

到 2026 年，goodput 被用于 MLPerf Inference v6.0 提交以及 AI 平台提供商的内部 SLA 跟踪。

### 为什么均值是错误的统计量

LLM 的延迟分布是右偏的。一次解码批次里如果有一个长 prefill 的邻居，可以发送 500 个 token 的 TPOT ~7 ms，而另外 20 个 token 的 TPOT ~60 ms。平均 TPOT 为 9 ms，但 P99 TPOT 为 65 ms。用户经常命中的是 P99 — 这就是他们离开的原因。

始终报告三元组 (P50, P90, P99)。对用户体验而言，P99 是优化的重点。

### 参考数值 — Llama-3.1-8B-Instruct 在 TRT-LLM，2026

- mean TTFT: 162 ms
- mean TPOT: 7.33 ms
- mean E2E: 1,093 ms
- P99 TPOT: 依分片预填充配置不同在 10-25 ms 之间波动

这些是 NVIDIA 发布的参考点。它们会随模型大小变化（70B 大小通常会放大 3-5 倍）、硬件（H100 vs B200 约 3x）和负载而改变。

### 测量陷阱

两个在 2026 年最常用的基准工具会对同一次运行的 TPOT 得出不同结果：

- **NVIDIA GenAI-Perf**：在 ITL 计算中排除了 TTFT。ITL 从第 2 个 token 开始计算。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个 token 开始计算。

对于一个 TTFT 为 500 ms、解码总耗时 700 ms 且输出 100 个 token 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择会改变结果。

始终说明所用工具和定义。始终发布定义。

### 构造 SLO

一个在 2026 年针对 70B 聊天模型的合理面向消费者的 SLO：

- TTFT P99 <= 800 ms
- TPOT P99 <= 25 ms
- E2E P99 <= 3 s（针对 <300-token 的输出）
- Goodput 目标 >= 99%

企业级 SLO 会收紧 TTFT（200-400 ms）并放松 E2E。关键是把它们写下来，三者都去度量，并将 goodput 作为一个复合指标进行跟踪。

### 如何测量

- 运行真实流量或逼近真实的合成流量（例如使用 LLMPerf 的 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 目标在基准运行中达到峰值并发的 2 倍。
- 运行 30-50 次迭代，取合并样本的百分位数。
- 发布时注明工具名称、工具版本、模型、硬件、并发度、提示分布。

```figure
throughput-latency
```

## 使用示例

`code/main.py` 是一个玩具级的 goodput 计算器。生成合成的延迟分布，应用 SLO，并计算 goodput。还会展示相同 trace 上 GenAI-Perf 与 LLMPerf 在 TPOT 计算上的差异。

## 交付产物

本课产出 `outputs/skill-slo-goodput-gate.md`。给定工作负载和 SLO，它会生成一个 CI/CD 就绪的基准配方，用 goodput（而不是吞吐率）来作为部署的门禁。

## 练习

1. 运行 `code/main.py`。生成带有 1% 尾部尖刺的分布。当你将 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 一个厂商宣称 “Llama 3.3 70B H100 上 15,000 tok/s”。在信任这个数字之前，你会问哪三条问题？
3. 为什么分片预填充（chunked prefill）能保护 P99 TPOT 而不能保护均值 TPOT？
4. 为语音助手构造一个面向消费者的 SLO（用户听到第一个 token，而不是阅读）。哪个指标对用户最可见？
5. 阅读 LLMPerf 的 README 和 GenAI-Perf 的文档。找出三个工具在其他哪些指标上存在分歧。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| TTFT | "time to first token" | Queue + network + prefill；在长提示下由 prefill 主导 |
| TPOT | "time per output token" | 第一个 token 之后每个 token 的内存受限解码成本 |
| ITL | "inter-token latency" | 在大多数工具中与 TPOT 相同（并非全部 — 参见 GenAI-Perf） |
| E2E | "end to end" | TTFT + TPOT * output_len；外加响应侧的网络延迟 |
| Throughput | "tok/s" | 集群效率；没有延迟百分位数时无意义 |
| Goodput | "SLO-met rate" | 同时满足所有 SLO 约束的请求比例 |
| P99 | "tail" | 最差 1/100 的延迟；用户体验的衡量指标 |
| SLO multi-constraint | "the joint" | 三个延迟上限的 AND；任一项违例即判请求失败 |
| GenAI-Perf vs LLMPerf | "the tool trap" | 两个工具在 ITL 是否包含 TTFT 上存在分歧 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的权威定义。  
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 另一套定义和测量方案。  
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 在真实部署上的应用测量。  
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准。  
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的基准工具。  
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业认可的基于 goodput 的基准。