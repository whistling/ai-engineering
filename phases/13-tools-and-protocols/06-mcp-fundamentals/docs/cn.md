# MCP 基础 — 原语、生命周期、JSON-RPC 基础

> 在 MCP 出现之前，每个集成都各自为政。Model Context Protocol（模型上下文协议），由 Anthropic 于 2024 年 11 月首次发布，并在 2025 年底由 Linux Foundation 的 Agentic AI Foundation 代管，将发现与调用标准化，使任何客户端都能与任何服务器通信。2025-11-25 规范定义了六个原语（三个服务器端、三个客户端）、一个三阶段生命周期，以及基于 JSON-RPC 2.0 的线格式。掌握这些内容，本阶段的 MCP 章节其余部分主要就是阅读了。

**Type:** 学习  
**Languages:** Python（stdlib，JSON-RPC 解析器）  
**Prerequisites:** Phase 13 · 01 到 05（工具接口和函数调用）  
**Time:** ~45 分钟

## 学习目标

- 说出所有六个 MCP 原语（服务器端：tools、resources、prompts；客户端：roots、sampling、elicitation），并为每个给出一个用例。
- 演练三阶段生命周期（initialize、operation、shutdown），并说明每个阶段谁发送哪些消息。
- 解析与生成 JSON-RPC 2.0 的请求、响应与通知封包。
- 解释 `initialize` 中的能力协商是什么以及如果没有它会发生什么问题。

## 问题背景

在 MCP 出现之前，每个使用工具的 agent 都有自己的协议。Cursor 有一个 MCP 形状但不兼容的工具系统。Claude Desktop 使用另一套。VS Code 的 Copilot 扩展又有第三套。一个团队为 “Postgres 查询” 写了同样的工具三次，每次针对不同主机的 API。要重用它就得复制代码。

结果是一次性集成的爆发式增长，生态系统的开发速度因此受限。

MCP 通过规范化线格式解决了这个问题。单个 MCP 服务器可以在任何 MCP 客户端上工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，以及到 2026 年 4 月的 300+ 客户端。每月 1.1 亿次 SDK 下载。10000+ 个公共服务器。Linux Foundation 在 2025 年 12 月将其代管到新的 Agentic AI Foundation。

本阶段使用的规范版本是 **2025-11-25**。该版本新增了异步任务（SEP-1686）、URL 模式征询（SEP-1036）、带工具的采样（SEP-1577）、增量范围同意（SEP-835），以及符合 OAuth 2.1 的 resource-indicator 语义。Phase 13 · 09 到 16 覆盖这些扩展。本课片段只到基础部分为止。

## 概念

### 三个服务器端原语

1. **Tools（工具）。** 可调用的操作。沿用 Phase 13 · 01 中的四步循环。
2. **Resources（资源）。** 暴露的数据。只读、可通过 URI 地址寻址：`file:///path`、`db://query/...`、以及自定义 scheme。
3. **Prompts（提示模板）。** 可复用的模板。在宿主 UI 中作为斜线命令；服务器提供模板，客户端填入参数。

### 三个客户端原语

4. **Roots（根）。** 服务器被允许访问的 URI 集合。由客户端声明；服务器必须遵守。
5. **Sampling（采样）。** 服务器请求客户端的模型执行一次补全。允许服务器端托管的 agent 循环在没有服务器端 API key 的情况下进行。
6. **Elicitation（征询）。** 服务器在交互中途向客户端的用户请求结构化输入。可以是表单或 URL（SEP-1036）。

MCP 中的每个能力恰好属于这六项之一。Phase 13 · 10 到 14 对每一项都有深入讨论。

### 线格式：JSON-RPC 2.0

每条消息都是一个包含下列字段的 JSON 对象：

- 请求（Requests）：`{jsonrpc: "2.0", id, method, params}`。
- 响应（Responses）：`{jsonrpc: "2.0", id, result | error}`。
- 通知（Notifications）：`{jsonrpc: "2.0", method, params}` — 没有 `id`，且不期望响应。

基础规范定义了 ~15 个方法，按原语分组。重要的方法包括：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：initialize。**

客户端发送 `initialize`，携带它的 `capabilities` 和 `clientInfo`。服务器以其自身的 `capabilities`、`serverInfo` 和它支持的规范版本响应。客户端在消化完响应后发送 `notifications/initialized`。从此之后，任一方都可以根据协商的能力发送请求。

**阶段 2：operation（运行）。**

双向通信。客户端调用 `tools/list` 进行发现，然后用 `tools/call` 调用工具。如果服务器声明了相应能力，它可以发送 `sampling/createMessage`。当工具集发生变化时，服务器可能发送 `notifications/tools/list_changed`。当用户更改根作用域时，客户端可能发送 `notifications/roots/list_changed`。

**阶段 3：shutdown（关闭）。**

任一方关闭传输。MCP 本身没有结构化的关闭方法；传输层（stdio 或 Streamable HTTP，见 Phase 13 · 09）承载连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 是合同。下面是服务器的示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发送 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自己的能力来达成一致：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明 `sampling`，服务器就不得调用 `sampling/createMessage`。对称地：如果服务器没有声明 `resources.subscribe`，客户端就不得尝试订阅。

这正是防止生态系统漂移的机制。一个不支持采样的客户端仍然是有效的 MCP 客户端；一个不调用采样的服务器仍然是有效的 MCP 服务器。它们只是不会一起使用该功能。

### 结构化内容与错误形状

`tools/call` 返回一个由类型化块组成的 `content` 数组：`text`、`image`、`resource`。Phase 13 · 14 将 MCP 应用（`ui://` 交互式 UI）加入该列表。

错误使用 JSON-RPC 错误代码。规范定义的扩展包括：`-32002` “Resource not found（资源未找到）”、`-32603` “Internal error（内部错误）”，以及作为 `error.data` 的 MCP 特定错误数据。

### 客户端能力 vs 工具调用细节

一个常见混淆点：`capabilities.tools` 表示客户端是否支持工具列表变更通知。客户端是否会在运行时调用特定工具是由其模型的实时决策驱动的，而不是能力标志。能力标志是规范级别的合同；模型的选择是正交的。

### 为什么选 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是一个轻量的双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（采样、通知），因此 JSON-RPC 以其对称的请求/响应形态自然契合。JSON-RPC 也能清晰地叠加在 stdio、WebSocket 或 Streamable HTTP 之上，而无需重做 HTTP 的请求形式。

```figure
mcp-tool-call
```

## 使用指南

`code/main.py` 附带了一个最小的 JSON-RPC 2.0 解析器和生成器，然后手动演示 `initialize` → `tools/list` → `tools/call` → `shutdown` 的序列，并打印每条消息。没有真实传输；只是消息形状。可对照“进一步阅读”中的规范核验每个封包。

注意事项：

- `initialize` 双向声明能力；响应包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应 `content` 是 `{type, text}` 块的数组。

## 交付物

本课输出 `outputs/skill-mcp-handshake-tracer.md`。给定一份类似 pcap 的 MCP 客户端-服务器交互记录，该 skill 注释每条消息属于哪个原语、哪个生命周期阶段，以及它依赖哪个能力。

## 练习

1. 运行 `code/main.py`。找出能力协商发生的那一行，并描述如果服务器没有声明 `tools.listChanged` 会发生什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形状：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在一个长时间运行的 `tools/call` 过程中发出该消息，并确认客户端的处理器会显示进度条。

3. 自头到尾阅读 MCP 2025-11-25 规范 —— 整个文档约 80 页。找出大多数服务器不需要的那个能力标志。提示：它与资源订阅有关。

4. 在纸上草拟一个假设性的 “cron job” 功能应属于哪个原语。（提示：服务器希望客户端在计划时间调用它。今天六个原语中没有一项完全匹配。）MCP 的 2026 路线图已有该功能的 SEP 草案。

5. 从 GitHub 上的一个开放 MCP 服务器解析一段会话日志。统计 request、response、notification 的数量。计算生命周期流量与运行时流量各占多少比例。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MCP | "Model Context Protocol" | 模型上下文协议——用于模型与工具发现与调用的开放协议 |
| Server primitive | "What a server exposes" | 服务器暴露的原语：tools（操作）、resources（数据）、prompts（模板） |
| Client primitive | "What a client lets servers use" | 客户端允许服务器使用的原语：roots（作用域）、sampling（LLM 回调/采样）、elicitation（征询用户输入） |
| JSON-RPC 2.0 | "The wire format" | 对称的请求/响应/通知封包 |
| `initialize` handshake | "Capability negotiation" | 首次消息对；服务器和客户端声明它们支持的特性 |
| `tools/list` | "Discovery" | 客户端查询服务器当前的工具集合 |
| `tools/call` | "Invocation" | 客户端请求服务器以参数执行某个工具 |
| `notifications/*_changed` | "Mutation events" | 服务器告知客户端其原语列表已发生变化 |
| Content block | "Typed result" | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "Spec Evolution Proposal" | 规范演进提案（例如 SEP-1686 表示异步任务） |

## 进一步阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 规范原文  
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语的心智模型  
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月的发布文章  
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾与 2025-11-25 规范变更  
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — 对 SEP-1686、1036、1577、835 与 1724 的总结