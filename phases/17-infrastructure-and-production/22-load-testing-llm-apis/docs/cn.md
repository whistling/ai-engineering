# Load Testing LLM APIs — Why k6 and Locust Lie

> 传统的负载测试工具并非为流式响应、可变输出长度、基于 token 的指标或 GPU 饱和度设计。两大陷阱会让大多数团队中招。GIL 陷阱：Locust 的基于 token 的测量在 Python GIL 下进行分词，这在高并发下与请求生成竞争；分词积压会膨胀报告的 token 间延迟 —— 瓶颈是你的客户端，而不是服务器。提示词-单一性陷阱：在循环中使用相同提示词只测试 token 分布的一点；真实流量具有可变长度和多样的前缀匹配。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 修复了这个问题。到 2026 年的工具映射：面向 LLM 的工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token 级准确性；**k6 v2026.1.0** + **k6 Operator 1.0 GA (Sept 2025)** — 支持流式、Kubernetes 原生、通过 TestRun/PrivateLoadZone CRD 分布式，适合 CI/CD 门控；Vegeta 用于 Go 常速饱和测试；Locust 2.43.3 仅在配合 LLM-Locust 扩展用于流式时可用。负载模式：稳态、爬坡、突发（自动扩缩容测试）、浸泡（内存泄漏）。

**Type:** Build  
**Languages:** Python（标准库、示例真实感提示生成器 + 延迟收集器）  
**Prerequisites:** Phase 17 · 08 (Inference Metrics)，Phase 17 · 03 (GPU Autoscaling)  
**Time:** ~75 分钟

## 学习目标

- 解释使通用负载测试工具对 LLM API 报假情况的两个反模式（GIL 陷阱、提示词-单一性陷阱）。
- 为给定目的选择工具：LLMPerf（基准运行）、k6 + 流式扩展（CI 门）、guidellm（大规模合成）、GenAI-Perf（NVIDIA 参考）。
- 设计四种负载模式（稳态、爬坡、突发、浸泡）并指明每种捕获的故障模式。
- 使用输入 token 的均值与标准差而不是固定长度构建逼真的提示词分布。

## 问题

你用 k6 在 500 并发用户下测试了 LLM 端点。测试通过。你发布了。在生产环境，只有 200 实际用户时服务崩溃了 —— P99 TTFT 飙升，GPU 被打满。

发生了两件事。首先，k6 发送了 500 个相同的提示词 —— 你的请求合并与前缀缓存让它看起来像在处理 500 个并发解码，实际上只处理了一个。其次，k6 并不像人的感知那样跟踪流式响应的 token 间延迟；它看到的是一个 HTTP 连接，而不是 500 个以不同时间间隔到达的 token。

针对 LLM 的负载测试是一门独立学科。

## 概念

### GIL 陷阱（Locust）

Locust 使用 Python，并在客户端在 GIL 下进行分词。在高并发下，分词器会在请求生成之后排队等候。报告的 token 间延迟包含了客户端分词的积压。你以为是服务器慢；其实是测试工具的限制。

修复方法：LLM-Locust 扩展将分词移动到独立进程，或者使用编译语言的测试框架（如 k6，或 LLMPerf 使用的 tokenizers.rs）。

### 提示词-单一性陷阱

所有已知的负载测试工具都允许你只配置一个提示词。在 10,000 次迭代的循环测试中，每次都发送完全相同的提示词。服务器每次都看到相同的前缀 —— 前缀缓存命中率接近 100%，吞吐看起来非常好。

修复方法：从提示词分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150` —— 多样的长度，多样的内容。

### 四种负载模式

1. **Steady-state（稳态）** — 恒定 RPS，持续 30–60 分钟。捕获：基线性能回归。  
2. **Ramp（爬坡）** — 在 15 分钟内线性增加 RPS 从 0 到目标。捕获：容量断点、预热异常。  
3. **Spike（突发）** — 突然 3–10 倍 RPS 持续 2 分钟然后恢复。捕获：自动扩缩容延迟、队列饱和、冷启动影响。  
4. **Soak（浸泡）** — 稳态运行 4–8 小时。捕获：内存泄漏、连接池漂移、可观测性数据溢出。

### 2026 年工具映射

**LLMPerf**（Anyscale） — Python 实现但分词由 Rust 支持。支持均值/标准差提示长度。支持流式。默认性能测试首选。

**NVIDIA GenAI-Perf** — NVIDIA 的参考实现。使用 Triton 客户端；覆盖全面的指标。注意它的 ITL（inference time metric）排除了 TTFT；LLMPerf 包含 TTFT。两个工具在同一台服务器上会产生不同的 TPOT 值。

**LLM-Locust**（TrueFoundry） — 修复 GIL 陷阱的 Locust 扩展。保留熟悉的 Locust DSL + 流式指标。

**guidellm** — 用于大规模合成基准测试的工具。

**k6 v2026.1.0** + **k6 Operator 1.0 GA (Sept 2025)**：
- k6 本身为 Go（编译型，无 GIL），新增了对流式指标的支持。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 实现 Kubernetes 原生的分布式测试。
- 适合用于 CI/CD 门控和 SLA 测试。

**Vegeta** — Go 实现，比 k6 精简。用于恒定速率的 HTTP 饱和测试。不是 LLM 专用，但适合测试网关/速率限制。

**Locust 2.43.3 原版** — 对 LLM 有 GIL 陷阱。只有配合 LLM-Locust 扩展才合适。

### CI 中的 SLA 门

在 PR 上运行 k6，要求：

- 每次 30–50 次迭代，按基线 RPS 运行。  
- 门控：P50/P95 TTFT，5xx < 5%，TPOT 在阈值内。  
- 违反则中断构建。

### 逼真的提示词分布

从真实流量样本构建（如果有）或来自已发布的分布（例如用于对话的 ShareGPT 提示、用于代码的 HumanEval）。将均值 + 标准差输入到 LLMPerf。避免任何循环单一提示词的测试。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。  
- k6 v2026.1.0：支持流式指标。  
- 典型 LLMPerf 运行：并发 X 下的 100–1000 个请求。  
- 典型 CI 门：每个 PR 执行 30–50 次迭代。  
- 四种模式：稳态、爬坡、突发、浸泡。

## 使用示例

`code/main.py` 模拟了带有逼真提示词分布的负载测试，测量有效 TPOT，并演示了提示词单一性的误导效果。

## 交付物

本课产出 `outputs/skill-load-test-plan.md`。根据工作负载和 SLA，选取工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。比较单一分布与逼真分布 —— 差距在哪里？  
2. 为 CI 门编写 k6 脚本：在 100 并发、运行 5 分钟的条件下，TTFT P95 < 800 ms。  
3. 你的浸泡测试显示内存以每小时 50 MB 的速度增长。列出三种可能的原因以及用于区分它们的观测/诊断手段。  
4. 突发测试从 10 RPS 到 100 RPS。若已部署 Karpenter + vLLM 生产栈（Phase 17 · 03 + 18），预期的恢复时间是多少？  
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 在同一台服务器上报告 TPOT=11ms。解释原因。

## 术语表

| Term | 人们所说 | 它实际的含义 |
|------|----------|--------------|
| LLMPerf | “the LLM harness” | Anyscale 的基准工具，支持流式 |
| GenAI-Perf | “NVIDIA tool” | NVIDIA 的参考基准工具 |
| LLM-Locust | “Locust for LLMs” | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | “synthetic benchmark” | 大规模合成基准工具 |
| k6 Operator | “K8s k6” | 基于 CRD 的分布式 k6 |
| GIL trap | “Python client overhead” | 客户端分词积压膨胀了报告延迟 |
| Prompt-uniformity trap | “single-prompt lie” | 循环相同提示词命中缓存，吞吐被夸大 |
| Steady-state | “constant load” | 固定 RPS，持续若干分钟 |
| Ramp | “linear up” | 在给定时长内线性增加到目标速率 |
| Spike | “burst test” | 突然倍增然后恢复的负载 |
| Soak | “long test” | 持续数小时以检测泄漏 |

## 扩展阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)  
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)  
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)  
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)  
- [LLMPerf](https://github.com/ray-project/llmperf)  
- [k6 Operator](https://github.com/grafana/k6-operator)