# Capstone 14 — Speculative-Decoding Inference Server

> EAGLE-3 在 vLLM 0.7 中在真实流量上实现了 2.5-3x 吞吐量。P-EAGLE（AWS 2026）进一步推动了并行投机。SGLang 的 SpecForge 在大规模上训练了 draft heads。Red Hat 的 Speculators hub 为常见开源模型发布了对齐的 drafts。TensorRT-LLM 将投机性解码作为 NVIDIA 的一等公民。到 2026 年，生产级推理栈是 vLLM 或 SGLang，配合 EAGLE 系列 drafts、FP8 或 INT4 量化，以及基于 queue-wait 的 HPA。本结题项目目标是在两个开源模型上以 ≥2.5x 基线吞吐提供服务，并给出完整的尾延迟报告。

**Type:** Capstone  
**Languages:** Python (serving), C++ / CUDA (kernel inspection), YAML (configs)  
**Prerequisites:** Phase 3（深度学习）, Phase 7（transformers）, Phase 10（从头构建 LLMs）, Phase 17（基础设施）  
**Phases exercised:** P3 · P7 · P10 · P17  
**Time:** 30 小时

## 问题

到 2026 年，投机性解码已经成为常用技术。EAGLE-3 draft heads 在目标模型的隐藏态上训练，预测前 N 个 token；目标模型在一次验证传递中核验这些候选。60–80% 的接受率能带来 2–3x 的端到端吞吐提升。vLLM 0.7 已原生集成该功能。SGLang + SpecForge 提供训练管线。Red Hat 的 Speculators 发布了针对 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 的对齐 drafts。

真正的工艺在于服务运行，而非模型本身。接受率会随流量分布漂移（ShareGPT vs 代码 vs 特定领域数据）。当被拒绝时的尾延迟通常比不使用投机更差——因此你必须在多个批次大小下报告 p99，而不仅仅是稳态的 tokens/sec。每 1M token 的成本与 Anthropic / OpenAI API 的比较是可信度的关键。

## 概念

投机性解码包含两层。一个 draft 模型（EAGLE-3 head、ngram，或更小的目标对齐模型）在每一步提出 k 个候选 token。目标模型在一次传递中验证所有 k 个；任何被接受的前缀会替换贪心路径。接受率取决于 draft 与目标的对齐程度以及输入分布。

在大多数流量上，EAGLE-3 优于 ngram drafts。P-EAGLE 对更深的 draft 树运行并行投机。权衡点在于：由于验证传递更大，被拒绝时的 P99 延迟更高。服务配置必须按批次大小分桶报告延迟以揭示这一点。

部署在 Kubernetes 上。vLLM 0.7 在每个 GPU 或张量并行分片上运行一个副本。HPA 根据 queue-wait 自动伸缩而非基于 CPU。FP8（Marlin）和 INT4（AWQ）量化将 GPU 内存控制在 H100 / H200 的可用范围内。端到端报告包括吞吐量、接受率、在 batch 1/8/32 下的 p50/p99，以及 $/1M tokens。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- Serving：vLLM 0.7 或 SGLang 0.4
- 投机方法：EAGLE-3 draft heads、P-EAGLE 并行投机、ngram 回退
- Draft 训练：SpecForge（SGLang）或 Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8（Marlin）、INT4（AWQ）
- 部署：Kubernetes + NVIDIA device plugin；HPA 基于 queue-wait 指标
- 评估：ShareGPT、MT-Bench-v2、GSM8K、HumanEval，用于衡量跨领域的接受率
- 参考：TensorRT-LLM 的投机性解码作为厂商基线

## 构建步骤

1. 目标模型准备。选择 Llama 3.3 70B。使用 Marlin 将其量化为 FP8。在 1x H100（或 2x 张量并行）上部署 vLLM 0.7。

2. Draft 来源。从 Red Hat Speculators 拉取对齐的 EAGLE-3 draft head（或通过 SpecForge 训练）。加载到 vLLM 的 speculative-decoding 配置中。

3. 基线数据。在启用投机前记录：batch 1/8/32 的 tokens/s、p50/p99 延迟、GPU 利用率。并发布这些基线数据。

4. 启用 EAGLE-3。切换配置并重新运行相同基准。报告加速比、接受率、p99 尾延迟差异。

5. P-EAGLE。启用并行投机；比较更深的 draft 树与串行 EAGLE-3 的表现。报告 P-EAGLE 有利或不利的拐点。

6. 不同域流量。对相同服务器分别运行 ShareGPT、HumanEval 和领域特定流量。测量各分布下的接受率。识别 draft 漂移发生的情形。

7. 第二个目标模型。在 Qwen3-Coder-30B MoE 上运行相同管线。由于 MoE 的路由噪声，draft 更具挑战性。进行报告。

8. K8s HPA。在 K8s 下部署，HPA 监控 `queue_wait_ms`。演示在负载增加 3 倍时的弹性伸缩。

9. 成本对比。计算 $/1M tokens，相比 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 在相同评估上的成本。并发布对比结果。

## 使用示例

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付物

`outputs/skill-inference-server.md` 描述了交付内容。交付应包含一个经测量的带投机性解码的服务栈、完整基准报告，以及一个 K8s 部署方案。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | 相对基线的实际加速 | 在两个模型上以匹配质量达到 2.5x+ 吞吐 |
| 20 | 在真实流量下的接受率 | 按分布的接受率报告 |
| 20 | P99 尾延迟规范 | 在 batch 1/8/32 下有无投机的 p99 报告 |
| 20 | 运维 | K8s 部署，基于 queue-wait 的 HPA，平滑滚动发布 |
| 15 | 报告与方法学 | 清晰说明做了哪些改变及其原因 |
| **100** | | |

## 练习

1. 测量当 draft 落后于目标一个版本时的接受率衰减（例如 Llama 3.3 -> 3.4 漂移）。构建一个监控告警。

2. 实现 ngram 回退：当 EAGLE-3 的接受率跌破阈值时切换到 ngram drafts。报告可靠性提升情况。

3. 运行受控的 MoE 实验：对相同的 Qwen3-Coder-30B 注入路由噪声与不注入时进行比较。测量 draft 接受率的敏感性。

4. 扩展到 H200（141 GB）。报告每个副本可用的模型尺寸余量，以及是否能在不量化的情况下服务 Llama 3.3 70B。

5. 在相同的 H100 硬件上基准 TensorRT-LLM 的投机性解码。报告其相对 vLLM 的优劣点。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Draft model | “Speculator” | 提出 N 个候选 token，供目标模型验证的小模型 |
| EAGLE-3 | “2026 draft architecture” | 在目标隐藏态上训练的 draft head；约 ~75% 的接受率 |
| P-EAGLE | “Parallel speculation” | 在一次目标验证传递中验证的 draft 分支树（并行投机） |
| Acceptance rate | “Hit rate” | 被接受且无需重采样的 drafted token 的比例 |
| Quantization | “FP8 / INT4” | 为在 GPU 内存中容纳更多模型而使用的低精度权重 |
| Queue wait | “HPA metric” | 请求在开始推理前在等待队列中的等待时间 |
| Speculators hub | “Aligned drafts” | Red Hat 上的对齐 EAGLE drafts 仓库 |

在上表中请注意：RAG 仍记作 RAG，Embeddings 翻译为 “嵌入”，Prompt engineering 为 “提示词工程”，few-shot 为 “少样本”，chain-of-thought 为 “思维链”，function calling 为 “函数调用”，speculative decoding 为 “投机性解码”，distributed training 为 “分布式训练”。

## 参考读物

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 参考的 serving 栈文档
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行投机性解码的论文与集成说明
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — draft-head 训练管线
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐 drafts 仓库
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — 厂商替代方案
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — 商业参考架构
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM repository](https://github.com/vllm-project/vllm) — 代码与基准资料