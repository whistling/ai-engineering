# LLM 可观测性栈选择

> 2026 年的可观测性市场分为两类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估、提示词管理、会话重放打包在一起。网关/接入工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 的核心采用 MIT 许可证，具有良好的开源与商业平衡（云端每月免费 50K 事件）。Phoenix 是基于 OpenTelemetry 的，使用 Elastic License 2.0 —— 非常适合可漂移性/RAG 可视化，但并非作为持久的生产后端。Arize AX 利用零拷贝 Iceberg/Parquet 集成，宣称在规模上比单体可观测性便宜约 100 倍。LangSmith 在 LangChain/LangGraph 领域领先，$39/用户/月，自托管仅限企业版。Helicone 是基于代理的，15–30 分钟即可完成设置，免费 100K 请求/月，但在 agent 跟踪的深度上稍逊。常见的生产模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），通过 OpenTelemetry 进行粘合。

**Type:** 学习  
**Languages:** Python（stdlib，玩具追踪抽样模拟器）  
**Prerequisites:** Phase 17 · 08（推理指标），Phase 14（代理工程）  
**Time:** ~60 分钟

## 学习目标

- 区分开发平台（打包：评估 + 提示词管理 + 会话）与网关/遥测工具（仅跟踪 + 指标）。
- 将六大主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到它们的许可、定价和适用场景。
- 解释 OpenTelemetry 粘合模式，说明如何将网关工具与独立评估平台组合使用。
- 指出 2026 年的成本差异化因素（Arize AX 的零拷贝方法 vs 单体式摄取）并说明大约 100x 的倍数。

## 问题

你发布了一个 LLM 功能。功能能跑通，但你对提示词失败、工具循环、延迟回归、成本激增或提示缓存命中率没有可见性。你在 Google 上搜索 “LLM observability”，会看到八个工具都声称解决相同问题，但价格分成三档。

它们并没有解决相同的问题。LangSmith 回答“为什么这个 LangGraph 运行失败？”，Phoenix 回答“我的 RAG 管道是否在漂移？”，Helicone 回答“哪个应用在消耗令牌？”，Langfuse 回答“我能否自托管整个方案？”。不同的工具，不同的受众。

选择时要考虑四个维度：栈（LangChain？原生 SDK？多供应商？）、许可容忍度（仅 MIT？Elastic 可接受？商业许可没问题？）、预算（免费层？$100/月？$1000/月？）和自托管（必须？可选？绝不可？）。

## 概念

### 两类工具

**开发平台（Development platforms）**将可观测性与评估、提示管理、数据集版本控制、会话重放捆绑在一起。你可以运行实验，查看哪个提示有效，用数据集回归测试新提示对老赢家的影响。代表：LangSmith、Langfuse、Comet Opik。

**网关/遥测工具（Gateway/telemetry tools）**对推理调用进行打点——提示、响应、令牌、延迟、模型、成本。代表：Helicone、SigNoz、OpenLLMetry、Phoenix。功能精简。可以通过 OpenTelemetry 与独立评估工具结合使用。

### Langfuse — OSS 平衡

- 核心采用 Apache / MIT 许可；可通过 Docker 自托管。
- 云端免费层：每月 50K 事件。付费：团队版 $29/月。
- 支持评估、提示管理、跟踪、数据集。对四类开发平台功能都覆盖得较合理。
- 适合场景：需要 LangSmith 级功能但必须自托管或坚持 OSS 许可的团队。

### Phoenix（Arize）— 遥测优先、OpenTelemetry 原生

- Elastic License 2.0；自托管很容易。
- 在 RAG 和漂移可视化上表现出色。嵌入空间散点图作为一等公民提供。
- 并非为持久性生产后端设计——主要用于开发时的可观测性。
- 适合场景：RAG 管道开发、漂移调试，通常与单独的网关配合用于生产。

### Arize AX — 规模化方案

- 商业产品。通过 Iceberg/Parquet 实现零拷贝数据湖集成。
- 声称在规模上比单体可观测性（如 Datadog 级别）便宜 ~100 倍。计算逻辑：将追踪以 Parquet 存在你的 S3 上；Arize 直接读取。
- 适合场景：>10M 条追踪/天，已有数据湖，想要 LLM 专用仪表盘但不想承担 Datadog 式定价。

### LangSmith — 面向 LangChain/LangGraph

- 商业产品，$39/用户/月。自托管仅限企业版。
- 在 LangChain 和 LangGraph 生态中是最佳实践。如果你不使用它们，则吸引力较小。
- 适合场景：团队已承诺使用 LangChain，且愿意付费。

### Helicone — 代理式的最小可行方案

- 通过将 `OPENAI_API_BASE` 指向 Helicone 代理，15–30 分钟即可上手。
- MIT 许可；免费 100K 请求/月，付费 $20/月 起。
- 包含故障转移、缓存、速率限制 —— 同时作为网关使用。
- 在 agent / 多步追踪的深度上不如一些专用平台。
- 适合场景：快速启动、单栈应用，需要网关 + 可观测性二合一。

### Opik（Comet）— 开源开发平台

- Apache 2.0，完全开源。
- 功能集与 Langfuse 类似，继承 Comet 的功能背景。
- 适合场景：已经在使用 Comet 的 ML 团队，希望在同一面板中获得 LLM 可观测性。

### SigNoz — OpenTelemetry 为先的全栈 APM

- Apache 2.0。处理通用 APM，并通过 OpenTelemetry 支持 LLM。
- 适合场景：希望统一服务与 LLM 调用的可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年发布了 GenAI 语义约定（例如 `gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。消费 OTel 的工具可以互操作。出现的生产模式为：

1. 从每次 LLM 调用发出遵循 GenAI 约定的 OTel 数据。  
2. 路由到网关（Helicone / Portkey）用于日常监控。  
3. 双向发送到评估平台（Phoenix / Langfuse）以便回归排查。  
4. 存档到数据湖（Iceberg）用于长期分析，通过 Arize AX 或 DuckDB 进行查询。

### 陷阱：在错误层面打点

在你的 agent 框架内部（例如在 agent 框架里加入 LangSmith 跟踪）进行打点，会把你耦合到该框架。若在 HTTP/OpenAI-SDK 层（通过 OpenLLMetry 或你的网关）打点，则更具可移植性。

### 采样——你无法保存所有数据

当超过 1M 请求/天时，全量追踪的保留成本会超过 LLM 调用本身。按规则采样：错误 100%、高成本 100%、成功请求 5%。始终保留聚合值；对长尾保留原始样本。

### 需要记住的数字

- Langfuse 免费云：每月 50K 事件。  
- LangSmith：$39/用户/月。  
- Helicone 免费：每月 100K 请求。  
- Arize AX 宣称：在规模上约比单体式便宜 100x。  
- OpenTelemetry GenAI 约定：2025 年发布，2026 年广泛采用。

## 使用示例

`code/main.py` 模拟了一个 1M 条追踪/天的场景，采用不同的保留策略（100% 摄取、采样、采样 + 错误优先），报告存储成本以及在每种策略下会丢失哪些信息。

## 部署建议

本课件生成 `outputs/skill-observability-stack.md`。根据栈、规模、预算、许可姿态，推荐相应工具组合。

## 练习

1. 你的团队在使用 LangChain，想要开源自托管的可观测性。选择 Langfuse 或 Opik 并给出理由。  
2. 在 5M 追踪/天，Datadog 报价 $150K/月的情况下，计算 Arize AX 的盈亏平衡点。  
3. 设计一套你们组织应强制在每次 LLM 调用中包含的 OpenTelemetry GenAI 属性集合。  
4. 论证 Phoenix 单独用于生产是否足够。如果不足，在哪些情况下它不足？  
5. Helicone 带来 20ms 的代理开销。在 P99 TTFT（首字节到渲染时间）为 300 ms 时，这是否可接受？如果 SLA 是 100 ms，又如何？

## 关键术语

| 术语 | 人们怎么说 | 它实际上是什么意思 |
|------|------------|--------------------|
| OpenLLMetry | “OTel for LLMs” | 面向 LLM 的开源 OpenTelemetry 插件/工具 |
| GenAI conventions | “OTel attributes” | 用于 LLM 调用的标准 OTel 属性名称 |
| LangSmith | “LangChain observability” | 与 LangChain 生态绑定的商业平台 |
| Langfuse | “OSS LangSmith” | MIT 开源，具有类似功能集 |
| Phoenix | “Arize dev tool” | OpenTelemetry 原生的开发/评估平台 |
| Arize AX | “scale observability” | 商业化零拷贝 Iceberg/Parquet 可观测性方案 |
| Helicone | “proxy observability” | 收集 LLM 遥测的 HTTP 代理 + 网关功能 |
| Opik | “Comet LLM” | 来自 Comet 的 Apache 2.0 开源开发平台 |
| Session replay | “trace rerun” | 重放包含工具调用的完整 agent 会话 |
| Eval | “offline test” | 在标注数据集上运行候选模型/提示词的离线测试 |

## 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)  
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)  
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)  
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)  
- [Arize Phoenix docs](https://docs.arize.com/phoenix)  
- [Helicone docs](https://docs.helicone.ai/)