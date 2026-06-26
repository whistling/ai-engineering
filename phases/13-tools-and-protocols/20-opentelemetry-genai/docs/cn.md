# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个 agent 调用了五个工具、三个 MCP 服务器和两个子代理。你需要一个覆盖全部的 trace。OpenTelemetry GenAI 语义约定（v1.37 及以后版本的稳定属性）是 2026 年的标准，Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 原生支持。本课列出所需属性，梳理 span 层级（agent → LLM → tool），并提供一个可以插到任意 OTel exporter 的 stdlib span 发射器。

**Type:** 构建  
**Languages:** Python（stdlib，OTel span 发射器）  
**Prerequisites:** Phase 13 · 07（MCP 服务器），Phase 13 · 08（MCP 客户端）  
**Time:** ~75 分钟

## 学习目标

- 列出 LLM span 与工具执行 span 所需的 OTel GenAI 属性。
- 构建覆盖 agent 循环、LLM 调用、工具调用与 MCP 客户端派发的 trace 层级。
- 决定要捕获的内容（可选择启用）与要脱敏/省略的内容（默认）。
- 在不改写工具代码的情况下向本地 collector（Jaeger、Langfuse）发出 spans。

## 问题描述

2026 年 2 月的一次调试：用户报告 “我的 agent 有时响应需要 30 秒， 有时只需 3 秒。” 没有 traces。日志显示了 LLM 调用，但没有工具分发、没有 MCP 服务器往返、也没有子代理。你只能猜测。最终你发现：某个 MCP 服务器在冷启动时偶尔会挂起。

没有端到端追踪，你无法定位这个问题。OTel GenAI 解决了它。

这些约定由 2025–2026 年间 OpenTelemetry 语义约定小组统一确定。它们定义了稳定的属性名称，因此 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的 spans。只需一次 Instrumentation；即可发送到任意后端。

## 概念

### Span 层级

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整个过程在同一个 trace id 下嵌套。span id 表示父子关系。

### 必需属性

根据 2025–2026 年的 semconv：

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"` 等。
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"` 等。
- `gen_ai.request.model` — 请求的模型字符串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际提供服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 供应商响应 id，用于关联。

对于工具 spans：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 具体调用 id。
- `gen_ai.tool.description` — 工具描述（可选）。

对于 agent spans：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span 类型（Span kinds）

- 对跨进程边界的调用（LLM 供应商、MCP 服务器）使用 `SpanKind.CLIENT`。
- 对 agent 自身的循环步骤和工具执行使用 `SpanKind.INTERNAL`。

### 可选择启用的内容捕获（Opt-in content capture）

默认情况下，spans 只携带指标和时序 —— 不包含提示词或补全内容。大型载荷与 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并通过特定的内容捕获环境变量来包含这些内容。在生产环境启用前请谨慎审查。

### Span 上的事件

可以将 token 级别事件作为 span 事件添加：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

事件在 span 内按时间顺序排列，以支持详细重放。

### Exporter（导出目标）

OTel spans 可以导出到：

- Jaeger / Tempo：开源、内部部署。
- Langfuse：面向 LLM 可观测性的可视化，显示 token 使用情况。
- Arize Phoenix：评估（evals）与追踪结合。
- Datadog：商业服务；原生解析 `gen_ai.*` 属性。
- Honeycomb：列式数据存储，便于查询。

以上都使用 OTLP 作为传输格式。你的代码无需关心具体后端。

### 在 MCP 间的传播（Propagation across MCP）

当 MCP 客户端调用服务器时，应在请求中注入 W3C traceparent 头。可流式的 HTTP 支持标准头。stdio 原生不携带 HTTP 头；该规范的 2026 路线图讨论在 JSON-RPC 调用中增加 `_meta.traceparent` 字段。

在该功能发布之前：将 traceparent 手动包含到每个请求的 `_meta` 中。服务器记录该 trace id。

### 指标（Metrics）

除了 spans 外，GenAI semconv 还定义了指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

对不需要每次调用细节的仪表盘使用这些指标。

### AgentOps 层

AgentOps（成立于 2024 年）专注于 GenAI 可观测性。它封装了流行框架（LangGraph、Pydantic AI、CrewAI），以自动发出 OTel spans。如果你的栈使用受支持框架，可以使用它；否则请手动进行上面描述的 instrumentation。

## 使用方法

`code/main.py` 会以 OTLP-JSON 类似的格式向 stdout 发出符合 OTel 形状的 spans，模拟一个调用 LLM、派发两个工具并进行一次 MCP 往返的 agent。没有真实的 exporter —— 本课重点在于 span 的形状和属性集。你可以把输出粘到兼容 OTLP 的查看器里，或者直接阅读它。

需要关注的点：

- Trace id 在所有 spans 中共享。
- parent-child 链接通过 `parentSpanId` 编码。
- 所需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；有一种场景通过环境变量打开它。

## 交付物

本课会生成 `outputs/skill-otel-genai-instrumentation.md`。针对一个 agent 代码库，该 skill 会输出一个检测/埋点计划：在哪里添加 spans、应填充哪些属性、以及应面向哪些 exporter。

## 练习

1. 运行 `code/main.py`。统计 spans 数量并识别哪些是 CLIENT，哪些是 INTERNAL。

2. 打开内容捕获（环境变量），确认出现了 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件。注意 PII 的影响。

3. 为工具执行添加指标 `gen_ai.tool.execution.duration`，并在每次调用时发出一个直方图样本。

4. 将 traceparent 从父 agent span 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器能够看到相同的 trace id。

5. 阅读 OTel GenAI semconv 规范。识别本课代码未发出的一个规范中列出的属性并添加它。

## 术语表

| Term | 常说的说法 | 实际含义 |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | 用于 traces、metrics、logs 的开放标准 |
| GenAI semconv | "GenAI semantic conventions" | 针对 LLM / 工具 / agent spans 的稳定属性名称 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享的前缀 |
| Span | "定时的操作" | 拥有开始、结束与属性的工作单元 |
| Trace | "跨 span 的血统" | 共享 trace id 的 span 树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 关于 span 方向性的提示 |
| OTLP | "OpenTelemetry Line Protocol" | exporter 的传输格式 |
| Opt-in content | "提示词 / 补全捕获" | 默认关闭；通过环境变量启用 |
| traceparent | "W3C 头" | 在服务间传播 trace 上下文 |
| Exporter | "后端专用发送器" | 将 spans 发送到 Jaeger / Datadog / 等组件 |

## 相关阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI spans、指标与事件的权威约定  
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 与工具执行 span 属性清单  
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — agent 级别的 `invoke_agent` span  
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — 在 GitHub 上托管的规范源  
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产环境集成实战指南