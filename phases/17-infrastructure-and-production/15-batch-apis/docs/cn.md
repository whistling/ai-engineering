# 批处理 API — 50% 折扣已成行业标准

> 每个主要供应商都提供带有 50% 折扣和 ~24 小时周转的异步批处理 API。OpenAI、Anthropic、Google 以及大多数推理平台（Fireworks 的批处理层、Together 的批处理）都实现了相同的模式。将批处理与提示词缓存和隔夜流水线叠加后，成本会降至同步未缓存费用的 ~10%。规则非常简单粗暴：如果不是交互式的，就应该放到批处理。内容生成流水线、文档分类、数据抽取、报告生成、大规模标注、目录打标签——任何能容忍 24 小时延迟的工作，未迁移到批处理就是在白白放钱在桌上。2026 年的生产模式是将每个新的 LLM 工作负载分流到三条通道：交互式（同步 + 缓存）、半交互式（异步队列 + 回退）、批处理（隔夜、缓存输入叠加）。那些假装是交互式但能容忍几分钟延迟的工作负载浪费最大。

**Type:** 学习  
**Languages:** Python (stdlib, 玩具级 batch-vs-sync 成本模拟器)  
**Prerequisites:** Phase 17 · 14（提示词与语义缓存）  
**Time:** ~45 分钟

## 学习目标

- 说出三大供应商的批处理 API（OpenAI、Anthropic、Google）以及常见的 50% 折扣 + 24 小时周转保证。
- 计算对一个隔夜分类工作负载叠加批处理 + 缓存输入后的成本，并与同步未缓存基线比较。
- 将一个工作负载分流为 interactive / semi-interactive / batch，并说明理由。
- 说出两个陷阱：部分交互性（用户期望比 24 小时快）和输出模式漂移（不同供应商的批处理文件格式不同）。

## 问题场景

你的团队发布了一个每晚运行的报告生成流水线。50,000 个文档，对每个文档做摘要，聚类这些摘要，草拟一份高层简报。同步运行需要 4 小时，成本为 $2,000/晚。你听说了批处理 API。

批处理给你 50% 的折扣。你还对系统提示词（在所有 50k 次调用中共享）启用了提示词缓存。叠加后，账单下降到 $180/晚——约为基线的 ~9%。同一流水线，仅需三项配置更改。

批处理是 LLM 成本工具箱中最便宜的杠杆，但却很少有人拉动。原因主要是组织层面：团队想到“实时”时，实际上 SLA 是“明天早上之前”。本课要点是不要把 90% 的账单放在桌上。

## 概念

### 三大批处理 API

**OpenAI Batch API**：上传 JSONL 文件，包含请求列表。承诺 24 小时周转（实际通常 ~2–8 小时）。对输入和输出 token 提供 50% 折扣。`/v1/batches` 端点。可被缓存的输入在此基础上还适用缓存输入定价。

**Anthropic Message Batches**：JSONL 上传。24 小时周转。50% 折扣。支持 `cache_control` —— 缓存写入为显式操作，批处理内会自动进行缓存读取。

**Google Vertex AI Batch Prediction**：BigQuery 或 GCS 输入。对 Gemini 提供类似的 50% 折扣。可与 Vertex pipelines 集成。

### 语义：异步不等于慢

批处理的意思是“我承诺在 24 小时内返回”——不是“这会花 24 小时”。典型的 P50 是 2–6 小时。供应商会在 GPU 利用率较低的空闲时段调度你的批处理作业。

### 与缓存叠加

一个 50k 文档的摘要任务，使用同一个 4K-token 的系统提示词：

- 同步未缓存：50000 ×（$input × 4000 + $output × 200）按全价计费。
- 同步已缓存：系统提示词在第一次写入后被缓存；剩余 49999 次的输入成本降为原来的 1/10（示例）。
- 批处理已缓存：在上述基础上，输入与输出均享受 50% 折扣。

叠加效果：批处理 + 缓存 ≈ 同步未缓存费用的 ~10%。任何在隔夜运行并且有共享系统提示词的工作负载都应该使用这个方案。

### 工作负载分流

**Interactive（交互式）** — 用户等待响应。首屏响应时间很重要。使用同步调用并结合提示词缓存。不能使用批处理。

**Semi-interactive（半交互式）** — 用户提交任务，几分钟后回来查看结果。使用异步队列，并在批处理不可用时回退到同步。适用于中等体量的 RAG 索引之类场景。

**Batch（批处理）** — 用户期望“明早”或“下个小时”看到结果。内容流水线、大规模分类、离线分析。总是使用批处理，总是叠加缓存。

常见错误：因为流水线处于生产环境就把所有东西归为交互式。生产并不是延迟规范——SLA 才是。

### 部分交互性陷阱

有些功能看起来是交互式的，但可以容忍 5–10 分钟延迟。例如：一个每晚运行的客户健康报告带有“刷新”按钮。用户点击刷新，等待 10 分钟是可以接受的。团队却将其作为同步功能发布。如果有 50 个并发刷新，这会比通过批处理打包并通过邮件分发贵 10 倍。

要问的问题是：“24 小时对这个用户意味着什么？”如果答案是“他们不会在意”，那就用批处理。

### 输出模式陷阱

批处理文件格式在不同供应商间各不相同：

- OpenAI：JSONL，每行一个请求。
- Anthropic：JSONL，每行一条消息；响应格式嵌入在内。
- Vertex：BigQuery 表或 GCS 前缀，使用 TFRecord。

写一个“通吃多家供应商的批处理客户端”意味着每个供应商都需要适配器代码。那些宣称支持多供应商批处理的网关（Portkey、LiteLLM 的部分层级）仍然只是对原生格式做了薄封装。

### 应记住的数据

- 各家供应商的批处理折扣：输入 + 输出统一 50% 折扣。
- 周转 SLA：保证 24 小时，典型 P50 为 2–6 小时。
- 批处理 + 缓存叠加：≈ 同步未缓存费用的 ~10%。
- 工作负载分流规则：如果可接受 24 小时延迟，则始终使用批处理。

## 使用方法

`code/main.py` 会计算 50k 文档工作负载在同步、同步+缓存、批处理、批处理+缓存 下的成本。并以 $ 和百分比报告节省量。

## 发布产物

本课会生成 `outputs/skill-batch-triager.md`。基于工作负载特性，将其分流为 interactive/semi/batch 并估算节省。

## 练习

1. 运行 `code/main.py`。对于一个 100k 文档、3K-token 系统提示词和 500-token 输出的流水线，计算全叠加（批处理 + 缓存）相对于同步基线的节省。
2. 从你熟悉的真实产品中挑三个功能。将每个功能分流为 interactive/semi/batch。
3. 一位用户抱怨他们的报告花了 3 小时。那是一次错误的批处理分流，还是合理的交互式？写出决策判据。
4. 你的批处理 API 返回 SLA 为 24 小时，但 P99 为 20 小时。你如何向用户沟通这一点——在极端情况下下游系统应如何表现？
5. 计算盈亏平衡点：当共享前缀长度达到多少时，批处理 + 缓存会比在你自己的保留 GPU 上通宵运行更便宜？

## 术语表

| 术语 | 常说的 | 实际含义 |
|------|--------|----------|
| Batch API | “异步折扣” | 输入 + 输出统一 50% 折扣，24 小时周转 |
| JSONL | “批处理格式” | 每行一个 JSON 请求；OpenAI/Anthropic 的标准 |
| Message Batches | “Anthropic 批处理” | Anthropic 的批处理 API 产品名称 |
| Batch prediction | “Vertex 批处理” | Vertex AI 的批处理预测产品 |
| Turnaround SLA | “24 小时承诺” | 保证响应时间，而非典型值；典型为 2–6 小时 |
| Workload triage | “交互性决策” | 将请求路由为 Interactive / Semi / Batch |
| Output schema | “响应格式” | 每个供应商的 JSONL 布局；不可直接移植 |
| Stacked discount | “批处理 + 缓存” | 当两者都生效时，约为未缓存同步账单的 ~10% |

## 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL 格式与 `/v1/batches` 语义说明。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — 批处理格式与 `cache_control` 交互说明。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini 的批处理语义。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)