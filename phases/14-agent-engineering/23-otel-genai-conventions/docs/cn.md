# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（于 2024 年 4 月启动）定义了智能体遥测的标准模式。跨度名称、属性和内容捕获规则在各厂商之间趋同，因此在 Datadog、Grafana、Jaeger 和 Honeycomb 中的智能体追踪具有相同含义。

**Type:** 学习 + 构建  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 14 · 13（LangGraph），Phase 14 · 24（可观测性平台）  
**Time:** ~60 分钟

## 学习目标

- 命名 GenAI 的 span 类别：model/client、agent、tool。  
- 区分 `invoke_agent` 的 CLIENT 与 INTERNAL spans 及其适用场景。  
- 列出顶层 GenAI 属性：provider 名称、request model、data-source ID。  
- 解释内容捕获合约：需显式 opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、推荐外部引用存储。

## 问题

每个厂商都会发明自己的 span 名称。运维团队最终需要为每个框架构建单独的仪表板。OpenTelemetry 的 GenAI SIG 通过定义整个生态系统共同遵循的标准来解决此问题。

## 概念

### Span 类别

1. **Model / client spans。** 覆盖原始 LLM 调用。由提供者 SDK（Anthropic、OpenAI、Bedrock）和框架的模型适配器发出。  
2. **Agent spans。** `create_agent`（当智能体被构建时）和 `invoke_agent`（当其运行时）。  
3. **Tool spans。** 每次工具调用各自对应一个 span；通过父子关系与 agent span 相连。

### Agent span 命名

- Span 名称：如果有名称则为 `invoke_agent {gen_ai.agent.name}`；否则回退为 `invoke_agent`。  
- Span 类型（kind）：  
  - **CLIENT** — 适用于远程智能体服务（OpenAI Assistants API、Bedrock Agents）。  
  - **INTERNAL** — 适用于进程内智能体框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。  
- `gen_ai.request.model` — 请求的模型 ID。  
- `gen_ai.response.model` — 实际解析出的模型（可能因路由策略与请求不同）。  
- `gen_ai.agent.name` — 智能体标识符。  
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。  
- `gen_ai.data_source.id` — 对于 RAG：被查询的语料库或存储的标识。

针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 存在技术特定的约定。

### 内容捕获

默认规则：instrumentations 不应默认捕获输入/输出。捕获需显式 opt-in，通过以下属性允许：

- `gen_ai.system_instructions`  
- `gen_ai.input.messages`  
- `gen_ai.output.messages`

推荐的生产模式：将内容外部存储（S3、你的日志存储），并在 spans 上记录引用（指针 ID，而非明文）。这是第 27 课中针对内容中毒的可观测性防护实践。

### 稳定性

截至 2026 年 3 月，多数约定仍为实验性。使用以下环境变量 opt-in 到稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 会将 GenAI 属性原生映射到其 LLM Observability 模式。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 此模式的常见错误

- **在 spans 中捕获完整提示（prompts）。** 导致 PII、密钥、客户数据出现在运维可读的追踪中。应外部存储。  
- **没有设置 `gen_ai.provider.name`。** 在缺失归属信息时，多提供商仪表板会失效。  
- **Spans 缺少父链接。** 导致工具 spans 成为孤儿。始终传播上下文。  
- **未设置稳定性 opt-in。** Collector 或后端升级时你的属性可能被重命名。

## 构建示例

`code/main.py` 实现了一个符合 GenAI 约定的标准库级别 span 发射器：

- 提供带有 GenAI 属性模式的 `Span`。  
- 提供带 `start_span`、嵌套上下文的 `Tracer`。  
- 一个脚本化的智能体运行，会发出：`create_agent`、`invoke_agent`（INTERNAL）、每个工具的 span，以及用于 LLM 调用的 `chat` spans。  
- 一个内容捕获模式，将提示外部存储并在 spans 上记录 ID。

运行命令：

```
python3 code/main.py
```

输出：包含所有必需 GenAI 属性的 span 树，以及显示 opt-in 内容引用的“外部存储”示例。

## 使用场景

- **Datadog LLM Observability**（v1.37+）会原生映射这些属性。  
- **Langfuse / Phoenix / Opik**（第 24 课）— 自动为生态系统打点。  
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel traces；可基于 GenAI 属性构建仪表盘。  
- **自托管** — 运行带有 GenAI 处理器的 OTel Collector。

## 部署指南

`outputs/skill-otel-genai.md` 将 OTel GenAI spans 接入现有智能体，默认启用内容捕获策略并使用外部引用存储。

## 练习

1. 对第 01 课的 ReAct 智能体循环进行打点，使用 `invoke_agent`（INTERNAL）+ 每个工具的 spans。将数据发送到 Jaeger 实例。  
2. 以“仅引用”模式增加内容捕获：将提示存入 SQLite，span 属性仅携带行 ID。  
3. 阅读 `gen_ai.data_source.id` 规范。将其接入第 09 课的 Mem0 检索。  
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并验证你的属性在 Collector 升级后不会被重命名。  
5. 构建仪表盘：仅基于 GenAI 属性回答“哪些工具错误与哪些模型相关联”。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| GenAI SIG | “OpenTelemetry GenAI 组” | OTel 工作组，定义该模式 |
| invoke_agent | “智能体 span” | 表示一次智能体运行的 span 名称 |
| CLIENT span | “远程调用” | 针对远程智能体服务的 span |
| INTERNAL span | “进程内” | 针对进程内智能体运行的 span |
| gen_ai.provider.name | “提供者” | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | “RAG 源” | 检索命中的语料库/存储标识 |
| Content capture | “提示日志” | 对消息的 opt-in 捕获；生产环境中应外部存储 |
| Stability opt-in | “预览模式” | 通过环境变量锁定实验性约定 |

（注：在本节中，术语已使用标准 AI 工程中文译法，例如“提示词工程”译为“提示词工程”，“agent loop”译为“智能体循环”，“stateful graphs”译为“有状态图”，以便与业界惯例一致。）

## 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范  
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认包含 GenAI spans  
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel spans  
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context 传播