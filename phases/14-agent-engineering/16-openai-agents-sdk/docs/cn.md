# OpenAI Agents SDK: Handoffs, Guardrails, Tracing

> OpenAI Agents SDK 是基于 Responses API 构建的轻量级多智能体框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。Handoff 表现为名为 `transfer_to_<agent>` 的工具。Guardrails 会在输入或输出上触发。Tracing 默认开启。

**Type:** 学习 + 构建  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 14 · 01（智能体循环）、Phase 14 · 06（工具使用）  
**Time:** ~75 分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个原语。
- 解释 handoffs：为何将其建模为工具、模型看到的名称形式、以及上下文如何传递。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` 与阻塞模式的区别。
- 在 stdlib 中实现带有 handoffs + guardrails + span 风格 tracing 的运行时。

## 问题背景

无法干净地委派的智能体最终会把所有内容塞进同一个提示词。没有护栏的智能体会泄露 PII、产生违反策略的输出，或陷入无限循环。OpenAI 的 SDK 将使多智能体可控的三个原语进行了规范化。

## 概念

### 五个原语

1. **Agent.** LLM + 指令 + 工具 + handoffs。
2. **Handoff.** 委派给另一个 agent。对模型呈现为名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail.** 在输入（仅第一个 agent）、输出（仅最后一个 agent）或工具调用（针对函数工具）上执行的验证。
4. **Session.** 自动的会话历史，跨轮次保存。
5. **Tracing.** LLM 生成、工具调用、handoff、guardrail 的内置 span。

### 将 Handoffs 视为工具

模型在它的工具列表中会看到 `transfer_to_billing_agent`。调用它会向运行时发出信号，从而：

1. 复制会话上下文（或通过 `nest_handoff_history` beta 将其折叠）。
2. 使用目标 agent 的指令初始化目标 agent。
3. 用目标 agent 继续运行。

这是监督者模式（第13课 / 第28课）的产品化实现。

### Guardrails（护栏）

三种类型：

- **输入护栏。** 在第一个 agent 的输入上运行。在任何 LLM 调用之前拒绝不安全或超出范围的请求。
- **输出护栏。** 在最后一个 agent 的输出上运行。捕获 PII 泄露、策略违规、或格式错误的响应。
- **工具护栏。** 针对每个函数工具运行。验证参数、检查权限、记录审计执行。

运行模式：

- **Parallel**（默认）。护栏 LLM 与主 LLM 并行运行。降低尾延迟。如果触发，主 LLM 的工作会被丢弃（浪费 token）。
- **Blocking**（`run_in_parallel=False`）。护栏 LLM 先运行。如果触发，则不会在主调用上浪费 token。

Tripwires 会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### Tracing（追踪）

默认开启。每次 LLM 生成、工具调用、handoff 和护栏都会发出一个 span。设置 `OPENAI_AGENTS_DISABLE_TRACING=1` 可选择退出。使用 `add_trace_processor(processor)` 可以将 span 同时发送到你自己的后端以及 OpenAI 的后端。

### Sessions（会话）

`Session` 将会话历史存储在后端（SQLite、Redis、自定义）。`Runner.run(agent, input, session=session)` 会自动加载并追加历史。

### 该模式可能出错的地方

- **Handoff 漂移（Handoff drift）。** Agent A 把任务交给 Agent B，Agent B 又交回给 Agent A。为此需要添加跳数计数器。
- **护栏绕过。** 工具护栏只在函数工具上触发；内置工具（文件读取、网页抓取）需要单独的策略。
- **过度追踪。** span 中包含敏感内容。请与 OTel GenAI 内容捕获规则（第23课）配合使用 — 将敏感内容外部存储，只在 span 中引用 ID。

## 构建它

`code/main.py` 在 stdlib 中实现了 SDK 的轮廓：

- `Agent`、`FunctionTool`、`Handoff`（作为具有转移语义的函数工具）。
- `Runner`，包含输入/输出/工具护栏、handoff 分发和跳数计数器。
- 一个简单的 span 发射器来展示 trace 的形态。
- 一个分诊（triage）agent，根据用户查询将任务转给 billing 或 support；一个输入护栏在某种输入上会触发。

运行：

```
python3 code/main.py
```

trace 会展示两个成功的 handoff、一次输入护栏触发，以及一个反映真实 SDK 发出的 span 树状结构。

## 使用场景

- 用于以 OpenAI 为主的产品时使用 **OpenAI Agents SDK**。
- 用于以 Claude 为主的产品时使用 **Claude Agent SDK**（第17课）。
- 当你需要显式状态和持久恢复时使用 **LangGraph**（第13课）。
- 当你需要精确控制（语音、多提供商、联邦部署）时使用 **Custom**。

## 部署指南（Ship It）

`outputs/skill-agents-sdk-scaffold.md` 为 Agents SDK 应用搭建脚手架，包含分诊 agent、handoffs、输入/输出/工具护栏、会话存储和 trace 处理器。

## 练习

1. 添加 handoff 跳数计数器：在 N 次转移后拒绝。追踪该行为。
2. 实现 `nest_handoff_history` 选项 — 在转移前将之前的消息折叠为一条摘要。
3. 编写一个阻塞式输出护栏。对会触发护栏的提示与通过的提示比较延迟。
4. 将 `add_trace_processor` 接上 JSON 日志器。每个 span 发出的形态是什么？
5. 阅读 SDK 文档。将你的 stdlib 玩具移植到 `openai-agents-python`。你在哪些地方建模错误？

## 关键词

| Term | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Agent | "LLM + instructions" | SDK 中的 Agent 类型；拥有工具和 handoffs |
| Handoff | "Transfer" | 模型调用以委派给另一个 agent 的工具 |
| Guardrail | "Policy check" | 在输入 / 输出 / 工具调用上的验证 |
| Tripwire | "Guardrail trip" | 护栏拒绝时抛出的异常 |
| Session | "History store" | 在运行之间持久化的会话记忆 |
| Tracing | "Spans" | 对 LLM + 工具 + handoff + guardrail 的内置可观测性 |
| Blocking guardrail | "Sequential check" | 护栏先运行；触发时不浪费 token |
| Parallel guardrail | "Concurrent check" | 护栏并行运行；延迟更低，但触发时会浪费 token |

## 深入阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 原语、handoffs、guardrails、tracing  
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对应实现  
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 何时使用 handoffs  
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK 的 span 对应的标准