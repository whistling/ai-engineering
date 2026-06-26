# MCP 资源与提示 — 超越工具的上下文暴露

> 工具获得了 MCP 90% 的注意力。其余两个服务器原语解决不同的问题。Resources（资源）用于以只读方式暴露数据；prompts（提示）将可重用的模板作为斜线命令暴露。许多服务器应该使用资源而不是将读取封装为工具，也应该使用提示而不是在客户端提示中硬编码工作流。本课命名了决策规则，并演示 `resources/*` 与 `prompts/*` 消息。

**Type:** 构建  
**Languages:** Python (stdlib, resource + prompt handler)  
**Prerequisites:** Phase 13 · 07（MCP server）  
**Time:** ~45 分钟

## 学习目标

- 在给定领域内，为某个能力在工具、资源或提示之间做出决策。
- 实现 `resources/list`、`resources/read`、`resources/subscribe` 并处理 `notifications/resources/updated`。
- 实现带有参数模板的 `prompts/list` 和 `prompts/get`。
- 识别主机何时将提示作为斜线命令展示，何时自动注入为上下文。

## 问题

一个简单的笔记应用 MCP 服务器将所有东西都暴露为工具：`notes_read`、`notes_list`、`notes_search`。这会把每次数据访问都包装成模型驱动的工具调用。后果：

- 模型必须决定是否在每个可能受益于上下文的查询中调用 `notes_read`。
- 只读内容无法被订阅或推流到主机的侧边栏。
- 客户端 UI（如 Claude Desktop 的资源附件面板、Cursor 的“包含文件”选择器）无法展示这些数据。

正确的划分：将数据作为资源暴露，将会改变或需要计算的操作作为工具暴露，将可重用的多步骤工作流作为提示暴露。每个原语都有其 UX 可得性和访问模式。

## 概念

### Tools vs resources vs prompts — 决策规则

| Capability | Primitive |
|------------|-----------|
| User wants to search, filter, or transform data | tool |
| User wants the host to include this data as context | resource |
| User wants a templated workflow they can re-run | prompt |

指导原则：如果模型在每个相关查询中调用该能力会受益，那它就是一个工具。如果用户会受益于将其附加到对话中，它就是一个资源。如果整个多步骤工作流是用户想要重用的单位，那它就是一个提示。

### 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接受 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的标识：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义 scheme）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 同时支持文本和二进制。二进制使用 `blob`（base64 编码字符串）并且有 `mimeType`。

### 资源订阅

在 capability 中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。当资源更改时，服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

用例：一个将资源映射为磁盘文件的笔记服务器；文件监视器触发更新通知；当外部编辑文件时，Claude Desktop 会重新拉取该文件进对话上下文。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 允许你暴露参数化的 URI 模式：`notes://{id}`，并将 `id` 作为补全目标。客户端可以在资源选择器中自动补全 id。

### 提示（Prompts）

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接受 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

提示是一个模板，填充后生成主机发送给模型的消息列表。例如，`code_review` 提示接受 `file_path` 参数并返回三条消息序列：一条 system 消息、一条包含文件主体的 user 消息，以及一条带有推理模板的 assistant 启动消息。

### 主机与提示

Claude Desktop、VS Code、Cursor 会在聊天 UI 中将提示作为斜线命令曝光。用户输入 `/code_review` 并从表单中选择参数。服务器的提示是“用户快捷方式”与“发送给模型的完整提示”之间的契约。

并非所有客户端都支持提示 — 请检查能力协商。声明了提示能力但客户端不支持提示的服务器，其斜线命令对该客户端将不可见。

### “列表已更改” 通知

当集合发生变动时，资源和提示都会发出 `notifications/list_changed`。例如，一个刚导入了 20 条新笔记的笔记服务器会发出 `notifications/resources/list_changed`；客户端会重新调用 `resources/list` 以获取新增项。

### 内容类型约定

文本示例：`mimeType: "text/plain"`、`text/markdown`、`application/json`。  
二进制示例：`image/png`、`application/pdf`，并带有 `blob` 字段。  
MCP 应用（Lesson 14）：在 `ui://` URI 中使用 `text/html;profile=mcp-app`。

### 动态资源

资源 URI 不必对应静态文件。`notes://recent` 可以在每次读取时返回最近五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可以自由计算内容。

规则：如果客户端可以按 URI 缓存，则该 URI 必须稳定。如果计算是一次性的，URI 应包含时间戳或 nonce，以防止客户端缓存失效导致陈旧数据。

### 订阅与轮询

支持订阅的客户端通过 `notifications/resources/updated` 接收服务器推送。未预先订阅的客户端或不支持订阅的主机则通过重新读取轮询。两者都符合规范。服务器在 capability 声明中告知客户端其支持哪种方式。

订阅的成本：服务器端每个会话都要记录订阅状态（谁订阅了什么）。保持订阅集合有边界；断开连接的客户端应超时清理。

### 提示与系统提示（system prompts）

MCP 中的提示不是系统提示。主机的系统提示（其自身的操作说明）和 MCP 的提示（服务器提供、由用户调用的模板）并存。一个良好的客户端绝不会让服务器提示覆盖自身的系统提示；二者应层叠应用。

## 使用示例

`code/main.py` 将第 07 课的笔记服务器扩展为：

- 每条笔记作为单独资源（`notes://note-1` 等），并支持 `resources/subscribe`。
- 一个会渲染为三条消息模板的 `review_note` 提示。
- 一个文件监视器模拟器，当笔记被修改时发出 `notifications/resources/updated`。
- 一个始终返回最近五条笔记的动态资源 `notes://recent`。

运行示例以观察完整流程。

## 发布成果

本课产出 `outputs/skill-primitive-splitter.md`。对于一个拟议的 MCP 服务器，该技能会将每个能力分类为 tool / resource / prompt，并给出理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发一次笔记编辑并验证 `notifications/resources/updated` 事件是否触发。

2. 添加一个 `resources/list_changed` 触发器：当创建新笔记时发送该通知，使客户端重新发现新资源。

3. 为 GitHub MCP 服务器设计三个提示：`summarize_pr`、`triage_issue`、`release_notes`。每个提示应有参数模式。提示主体应可在无需进一步编辑的情况下直接运行。

4. 从第 07 课服务器中选择一个现有工具，判断它应保留为工具还是拆分为资源加工具对。用一句话说明理由。

5. 阅读规范中 `server/resources` 与 `server/prompts` 部分。识别 `resources/read` 中很少使用但在规范中支持的那一个字段。提示：查看资源内容上的 `_meta`。

## 术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Resource | "Exposed data" | 可通过 URI 访问的主机可读取的内容 |
| Resource URI | "Pointer to data" | 带 scheme 前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | "Watch for changes" | 客户端选择的针对特定 URI 的服务器推送更新 |
| `notifications/resources/updated` | "Resource changed" | 向客户端发出所订阅资源已有新内容的信号 |
| Resource template | "Parameterized URI" | 带有补全提示的 URI 模式 |
| Prompt | "Slash-command template" | 带参数槽位的命名多消息模板 |
| Prompt arguments | "Template inputs" | 主机在渲染前收集的有类型参数 |
| `prompts/get` | "Render template" | 服务器返回填充后的消息列表 |
| Content block | "Typed chunk" | `{type: text | image | resource | ui_resource}` |
| Slash-command UX | "User shortcut" | 主机将提示作为以 `/` 开头的命令呈现 |

## 延伸阅读

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — 资源 URI、订阅与模板  
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — 提示模板与斜线命令集成  
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整的 `resources/*` 消息参考  
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整的 `prompts/*` 消息参考  
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — 社区指南，扩展官方文档