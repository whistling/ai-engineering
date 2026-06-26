# Capstone 11 — LLM 可观测性与评估仪表盘

> Langfuse 开源化。Arize Phoenix 发布了 2026 年 GenAI semconv 映射。Helicone 和 Braintrust 都在用户级成本归因上加大投入。Traceloop 的 OpenLLMetry 成为事实上的 SDK 自动埋点。生产形态是：ClickHouse 存储 traces，Postgres 存元数据，Next.js 作 UI，外加一小队评估作业（DeepEval、RAGAS、LLM-judge）在采样的 traces 上运行。构建一个自托管的系统，从至少四类 SDK 摄取，并演示在五分钟内捕获注入的回归。

**Type:** 结业项目  
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)  
**Prerequisites:** Phase 11 (LLM 工程), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)  
**Phases exercised:** P11 · P13 · P17 · P18  
**Time:** 25 小时

## 问题

到 2026 年，所有在生产环境运行流量的 AI 团队都保持一套与模型并行的可观测层。成本归因、幻觉检测、漂移监控、越狱信号、SLO 仪表盘、PII 泄露告警等。开源参考项目 —— Langfuse、Phoenix、OpenLLMetry —— 在摄取 schema 上趋于一致，采用 OpenTelemetry GenAI semantic conventions。现在你可以用一个 SDK 对 OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM 进行埋点并生成兼容的 spans。

你将构建一个自托管的仪表盘，从至少四类 SDK 摄取数据，在采样的 traces 上运行一小组评估作业，检测漂移并触发告警。衡量标准：在故意注入的回归（提示词开始产生 PII）情况下，仪表盘在五分钟内捕获并发出告警。

## 概念

摄取为 OTLP HTTP。SDK 产出符合 GenAI semconv 的 spans：`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`。spans 写入 ClickHouse 做列式分析；元数据（用户、会话、应用）写入 Postgres。

评估作为批处理作业在采样的 traces 上运行。DeepEval 评估保真性、毒性和答案相关性。RAGAS 在 trace 携带检索上下文时评估检索指标。自定义的 LLM-judge 运行领域特定检查（PII 泄露、越权响应）。评估结果作为 eval spans 写回同一 ClickHouse，并链接到父 trace。

漂移检测监控嵌入空间分布随时间的变化（对提示词嵌入做 PSI 或 KL 散度）以及评估分数趋势。告警通过 Prometheus Alertmanager 发送，然后推送到 Slack / PagerDuty。UI 使用 Next.js 15 与 Recharts。

## 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- 摄取：OpenTelemetry SDKs + GenAI semantic conventions；OTLP HTTP 传输  
- Collector：OpenTelemetry Collector，使用 tail-sampling 处理器（用于成本控制）  
- 存储：ClickHouse 用于 spans，Postgres 用于元数据，S3 用于原始事件归档  
- 评估：DeepEval、RAGAS 0.2、Arize Phoenix evaluator 包、自定义 LLM-judge  
- 漂移：对合并的提示词嵌入（使用 sentence-transformers）每周计算 PSI / KL  
- 告警：Prometheus Alertmanager -> Slack / PagerDuty  
- UI：Next.js 15 App Router + Recharts + server actions  
- 默认支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 实作步骤

1. Collector 配置。OpenTelemetry Collector，启用 OTLP HTTP 接收器，tail-sampler 策略：对错误 trace 保留 100%，对成功 trace 保留 10%，并导出到 ClickHouse 与 S3。

2. ClickHouse 模式。建表 `spans`，列映射 GenAI semconv：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，以及用于长负载的 JSON 包。按 `user_id` 和 `app_id` 建次级索引。

3. SDK 覆盖性测试。编写一个小型客户端应用，分别使用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）并启用 OpenLLMetry 自动埋点。验证每个 SDK 都能产生规范的 GenAI spans 并落盘到 ClickHouse。

4. 评估作业。定时作业读取最近 15 分钟的采样 traces，运行 DeepEval（保真性、毒性、答案相关性）。输出为与父 trace 关联的 eval spans。

5. 自定义 LLM-judge。实现一个 PII 泄露判定器：给定响应，调用一个守护 LLM（guard LLM）对 PII 泄露概率评分。高分响应进入人工分诊队列。

6. 漂移检测。周度作业计算本周合并提示词嵌入与过去 4 周基线的 PSI。如果 PSI 超阈值则告警。

7. 仪表盘。Next.js 15，页面包括：概览（spans/sec、每用户成本、p95 延迟）、traces（搜索 + 瀑布图）、评估（保真性趋势、毒性）、漂移（PSI 趋势）、告警。

8. 告警链路。Prometheus exporter 读取评估分数聚合与延迟分位数；Alertmanager 根据规则将警告路由到 Slack，将严重告警路由到 PagerDuty。

9. 回归探测。注入一个 bug：聊天机器人开始以 1% 概率泄露伪造的 SSN。衡量 MTTR：从 bug 部署到 Slack 告警的时间。

## 使用示例

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付

`outputs/skill-llm-observability.md` 为交付件。给定一个 LLM 应用，仪表盘能摄取其 traces、运行评估、对漂移告警，并在 Next.js 中展现每用户成本拆分。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema 覆盖率 | 产生规范 GenAI spans 的 SDK 家族数量（目标：6+） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数与人工标注集比对 |
| 20 | 仪表盘用户体验 | 注入回归的 MTTR（目标：<5 分钟） |
| 20 | 成本 / 扩展性 | 持续摄取 1k spans/sec 无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 全链路演练 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义埋点。验证规范的 spans（带有真实的 `gen_ai.*` 属性）落盘到 ClickHouse。  
2. 在相同 traces 上用 Phoenix evaluators 替换 DeepEval。比较两套评估引擎的分数漂移。  
3. 锐化漂移检测：按 app-id 计算 PSI 而非全局计算。展示每个应用的漂移曲线。  
4. 添加一个“用户影响”页面：每用户成本与每用户失败率并带小型折线图。  
5. 构建一个 tail-sampling 策略：对毒性 > 0.5 的 trace 保留 100%，其余按 10% 分层采样。评估引入的采样偏差。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| GenAI semconv | “OTel LLM attributes” | 2025 年 OpenTelemetry 对 LLM span 属性的规范（system、model、tokens） |
| Tail sampling | “后追踪采样” | Collector 在 trace 完结后决定保留或丢弃（可以查看错误） |
| PSI | “Population stability index” | 比较两个分布的漂移度量；通常 > 0.2 表示显著漂移 |
| LLM-judge | “以模评模” | 用一个 LLM 根据量表（保真性、毒性、PII）为另一个 LLM 的输出评分 |
| Tail-sampling policy | “保留规则” | 决定哪些 traces 被持久化或丢弃的规则；例如保留错误 + 按采样率保留 |
| Eval span | “关联的评估 span” | 一个子 span，携带评估分数并关联到原始 LLM 调用 span |
| Cost per user | “单位经济” | 在一个窗口期内按 user_id 归因的美元成本；关键产品指标 |

（注：术语翻译力求与常见 AI 工程用语一致，例如将 Prompt engineering 翻为 提示词工程、Embeddings 翻为 嵌入、Fine-tuning 翻为 微调、Context window 翻为 上下文窗口、few-shot 翻为 少样本、chain-of-thought 翻为 思维链、guardrails 翻为 护栏、function calling 翻为 函数调用、speculative decoding 翻为 投机性解码、positional embeddings 翻为 位置嵌入、self-attention 翻为 自注意力、instruction tuning 翻为 指令微调、distributed training 翻为 分布式训练。）

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考开源可观测平台  
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 支持强漂移检测的替代参考实现  
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 自动埋点 SDK 家族  
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 摄取 schema 规范  
- [Helicone](https://www.helicone.ai) — 另一个托管可观测方案  
- [Braintrust](https://www.braintrust.dev) — 侧重评估的替代平台  
- [ClickHouse documentation](https://clickhouse.com/docs) — 列式 spans 存储文档  
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估库