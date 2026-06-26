# Function Calling Deep Dive — OpenAI, Anthropic, Gemini

> 这三家前沿提供商在 2024 年在同一个工具调用循环上达成了共识，然后在其它方面各自分裂。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 并通过唯一 ID 做关联。本课将并排对比三者，以便在将代码从一个提供商移植到另一个时不会破坏。

**Type:** 构建  
**Languages:** Python（stdlib、schema 转换器）  
**Prerequisites:** Phase 13 · 01（工具接口）  
**Time:** ~75 分钟

## 学习目标

- 说明 OpenAI、Anthropic 和 Gemini 在函数调用负载（声明、调用、结果）上的三种结构差异。
- 将一个工具声明翻译为三家提供商的格式，并预测严格模式约束在哪些地方会有所不同。
- 在每个提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解各提供商的硬性限制（工具数量、模式深度、参数长度）以及当超限时它们各自返回的错误特征。

## 问题陈述

函数调用请求的结构因提供商而异。以下是 2026 年生产栈中的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应在 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]` 中包含调用，其中 `arguments` 是一个需要你解析的 JSON 字符串。严格模式（`strict: true`）通过受限解码强制执行模式合规性。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 返回。`input` 已经被解析（是一个对象，而不是字符串）。你用包含 `{type: "tool_result", tool_use_id, content}` 块的新 `user` 消息回复。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 到达，其中在 Gemini 3 及以上版本中 `id` 是用于并行调用关联的唯一值。你回复 `{functionResponse: {name, id, response}}`。

相同的循环。不同的字段名、不同的嵌套、不同的字符串与对象约定、不同的关联机制。一个团队在 OpenAI 上写的天气 agent 移植到 Anthropic 需要两天的接线工作，再到 Gemini 又需要一天，仅仅是为了这些格式差异。

本课构建一个翻译器，将三种格式统一为一个规范化的工具声明并在边缘路由。Phase 13 · 17 会把相同模式泛化为一个 LLM 网关。

## 概念

### 通用结构

每个提供商都需要五样东西：

1. 工具列表。每个工具的名称、描述和输入模式（input schema）。
2. 工具选择。强制指定某个工具、禁止工具，或让模型决定。
3. 调用发出。结构化输出，指定工具和参数。
4. 调用 ID。将响应与正确的调用关联（对并行调用很重要）。
5. 结果注入。将结果与调用绑定回来的消息或块。

### 字段级别的结构差异

| 方面 | OpenAI | Anthropic | Gemini |
|------|--------|-----------|--------|
| 声明外壳 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| 模式字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | 字符串化的 JSON | 已解析的对象 | 已解析的对象 |
| ID 格式 | `call_...`（由 OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | 角色 `tool`，`tool_call_id` | 带 `tool_result` 的 `user`，`tool_use_id` | 带匹配 `id` 的 `functionResponse` |
| 强制指定工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格模式 | `strict: true` | schema 即契约（始终执行） | 请求级别的 `responseSchema` |

### 你实际上会碰到的限制

- **OpenAI。** 每次请求最多 128 个工具。模式深度 5。参数字符串 <= 8192 字节。严格模式要求没有 `$ref`，没有带重叠的 `oneOf`/`anyOf`/`allOf`，并且 `required` 中列出每个属性。
- **Anthropic。** 每次请求最多 64 个工具。模式深度理论上不受限，但实际建议限制在 10。没有严格模式开关；模式被视为契约，模型倾向于遵守。
- **Gemini。** 每次请求最多 64 个函数。模式类型为 OpenAPI 3.0 子集（与 JSON Schema 2020-12 有细微差异）。Gemini 3 对并行调用引入了唯一 ID。

### `tool_choice` 行为

三种模式大家都支持，名称不同。

- 自动（Auto）。模型选择工具或文本。默认。
- 要求 / Any。模型必须至少调用一个工具。
- 无（None）。模型不得调用工具。

此外，每个提供商还有一个独有模式：

- **OpenAI。** 按名称强制指定某个工具。
- **Anthropic。** 按名称强制指定某个工具；`disable_parallel_tool_use` 标志区分单次与并行。
- **Gemini。** `mode: "VALIDATED"` 会让每个响应通过模式验证器，无论模型意图如何。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）会在一次助手消息中发出多个调用。你需要全部运行它们，并用一个批量的 tool-role 消息回复，每个 `tool_call_id` 一条记录。Anthropic 历史上只做单次调用；`disable_parallel_tool_use: false`（Claude 3.5 的默认值）启用多次调用。Gemini 2 允许并行调用但没有稳定的 ID；Gemini 3 为并行调用引入了 UUID，从而能够正确关联乱序响应。

### 流式传输（Streaming）

三者都支持流式的工具调用，线格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量 delta 块会逐步到达。你需要累积直至 `finish_reason: "tool_calls"`。
- **Anthropic。** 使用 block-start / block-delta / block-stop 事件。`input_json_delta` 片段携带部分参数。
- **Gemini。** 在 Gemini 3 中新增的 `streamFunctionCallArguments` 发出带有 `functionCallId` 的片段，以便多个并行调用可以交错传输。

Phase 13 · 03 会深入讨论并行 + 流式重组。本课着重声明和单次调用的形状。

### 错误与修复

无效参数的错误表现也不同。

- **OpenAI（非严格模式）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入一条错误消息并重新调用。
- **OpenAI（严格模式）。** 在解码期间进行验证；无效 JSON 不可能出现，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；模式更像建议。请在服务器端验证。
- **Gemini。** OpenAPI 3.0 的怪癖：对象字段上的 `enum` 会被静默忽略；需自行验证。

### 翻译器模式

你的代码中，一个规范化的工具声明看起来像这样（你可以选定形状）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将它翻译为三家提供商的声明。`code/main.py` 中的测试装置正好做了这件事，然后将一个伪造的工具调用按每家提供商的响应形状往返一遍。无需网络——本课教的是形状，而不是 HTTP。

生产团队会把这个翻译器封装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。Phase 13 · 17 发布了一个在任意三者前面暴露 OpenAI 形状 API 的网关。

## 使用方法

`code/main.py` 定义了一个规范化的 `Tool` 数据类和三个翻译器，它们分别输出 OpenAI、Anthropic 和 Gemini 的声明 JSON。然后它把手工制作的每种提供商响应解析回相同的规范调用对象，演示在底层语义是一致的。运行它并并排比较三种声明。

观察点：

- 三个声明块仅在外壳和字段名上不同。
- 三个响应块在调用所在位置上不同（顶层的 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数能从三种响应形状中提取 `{id, name, args}`。

## 交付成果

本课产出 `outputs/skill-provider-portability-audit.md`。给定针对某个提供商的函数调用集成，技能将生成一份可移植性审计：它依赖哪些提供商限制、哪些字段需要重命名、移植到其它提供商时会有哪些中断。

## 练习

1. 运行 `code/main.py` 并验证三家提供商的声明 JSON 都序列化了相同的底层 `Tool` 对象。修改规范工具以添加一个枚举参数，并确认只有 Gemini 翻译器需要处理 OpenAPI 的怪癖。
2. 为每个提供商添加一个 `ListToolsResponse` 解析器，以提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 本身没有原生提供；注意这一不对称性。
3. 实现 `tool_choice` 转换：将规范的 `ToolChoice(mode="force", tool_name="x")` 映射为三家提供商的所有形状。然后映射 `mode="any"` 和 `mode="none"`。对照本课的差异表进行检查。
4. 选择三家提供商中的一个，通读其函数调用指南全文。找出该提供商的模式规范中其它两家不支持的一个字段。候选项：OpenAI 的 `strict`、Anthropic 的 `disable_parallel_tool_use`、Gemini 的 `function_calling_config.allowed_function_names`。
5. 编写一个测试向量：一个其参数违反声明模式的工具调用。将其通过每个提供商的验证器（第一课中的 stdlib 验证器可做代理）并记录哪些错误触发。记录你会出于严格性考虑在生产中选用哪个提供商。

## 关键术语

| 术语 | 常说法 | 实际含义 |
|------|--------|----------|
| Function calling | “Tool use” | 提供商级别的用于结构化工具调用发出的 API（函数调用） |
| Tool declaration | “Tool spec” | 名称 + 描述 + JSON Schema 输入负载 |
| `tool_choice` | “Force / forbid” | 自动 / 必需 / 无 / 指定名称 模式 |
| Strict mode | “Schema enforcement” | OpenAI 用于限制解码以匹配模式的开关 |
| `tool_use` 块 | “Anthropic 的调用形状” | 带 id、name、input 的内联 content 块 |
| `functionCall` 部分 | “Gemini 的调用形状” | 一个包含 name、args 和 id 的 `parts[]` 条目 |
| Arguments-as-string | “Stringified JSON” | OpenAI 返回的参数是字符串化的 JSON，而不是对象 |
| 并行工具调用 | “Fan-out in one turn” | 在一次助手消息中发出多个工具调用 |
| Refusal | “Model declines” | 严格模式下模型以拒绝块而不是调用响应 |
| OpenAPI 3.0 子集 | “Gemini 模式怪癖” | Gemini 使用与 JSON-Schema 类似但有小差异的方言 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 官方参考，包括严格模式和并行调用  
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 与 `tool_result` 块语义  
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 ID 与 OpenAPI 子集  
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业层面文档  
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式模式强制执行细节