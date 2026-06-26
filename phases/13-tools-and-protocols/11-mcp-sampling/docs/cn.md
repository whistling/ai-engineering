# MCP 采样 — 服务器请求的 LLM 完成与代理循环

> 大多数 MCP 服务器只是简单的执行者：接收参数、运行代码、返回内容。采样让服务器可以反向发起请求：它让客户端的 LLM 做出决策。这使得服务器托管的代理循环成为可能，而服务器本身无需持有任何模型凭据。SEP-1577（于 2025-11-25 合并）在采样请求中加入了工具，使循环可以包含更深入的推理。漂移风险说明：SEP-1577 所描述的工具在采样中的形态在 2026 年第一季度仍处于实验阶段，且 SDK API 可能仍在调整。

**Type:** 构建
**Languages:** Python（标准库，采样 harness）
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决了什么问题（服务器托管循环但无需服务器端 API 密钥）。
- 实现一个服务器，它请求客户端对多轮提示进行采样并返回完成结果。
- 使用 `modelPreferences`（成本 / 速度 / 智能 优先级）来引导客户端的模型选择。
- 构建一个 `summarize_repo` 工具，该工具内部通过采样迭代而不是硬编码行为。

## 问题背景

一个用于代码摘要工作流的有用 MCP 服务器需要：遍历文件树、选择要读取的文件、综合生成摘要并返回。LLM 的推理应当在哪里发生？

选项 A：服务器调用自己的 LLM。需要 API 密钥，在服务器端计费，对每个用户都很昂贵。

选项 B：服务器返回原始内容；客户端的 agent 做推理。可行，但会把服务器逻辑移动到客户端提示中，这很脆弱。

选项 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（例如哪些文件要读、要做几轮），而客户端保留计费和模型选择。服务器完全不持有凭据。

采样就是选项 C。它是可信服务器在不作为完整 LLM 托管方的情况下托管代理循环的机制。

## 概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个浮点数，和为 1.0：

- `costPriority`：偏好更便宜的模型。
- `speedPriority`：偏好更快速的模型。
- `intelligencePriority`：偏好更高能力的模型。

以及 `hints`：服务器偏好的命名模型。客户端可能采纳或忽略 hints；客户端的用户配置始终优先。

### `includeContext`

三个取值：

- `"none"` — 仅包含服务器提供的消息。默认值。
- `"thisServer"` — 包含来自该服务器会话的先前消息。
- `"allServers"` — 包含所有会话上下文。

自 2025-11-25 起，`includeContext` 被软弃用，因为它会泄露跨服务器的上下文，存在安全隐患。优先使用 `"none"` 并在 `messages` 中显式传递上下文。

### 带工具的采样（SEP-1577）

2025-11-25 新增：采样请求可以包含 `tools` 数组。客户端使用这些工具执行完整的工具调用循环。这样服务器可以通过客户端的模型托管一个 ReAct 风格的代理循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：采样 -> 如果调用工具则执行工具 -> 再次采样 -> 返回最终的 assistant 消息。这在 2026 年第一季度仍属实验性；SDK 函数签名可能仍有变化。实现时请参考 2025-11-25 规范的 client/sampling 部分以确认细节。

### 人机在环（Human-in-the-loop）

客户端必须在运行采样前将服务器请求模型执行的内容展示给用户。恶意服务器可能利用采样去操纵用户会话（“对用户说 X，以便他们点击 Y”）。Claude Desktop、VS Code 和 Cursor 会将采样请求以确认对话框的形式显示，用户可以拒绝。

2026 年的共识：在没有人工确认的情况下进行采样是红旗（high-risk）。网关（Phase 13 · 17）可以对低风险采样进行自动批准，对可疑请求进行自动拒绝。

### 无需 API 密钥的服务器托管循环

典型用例：一个没有自身 LLM 访问权限的代码摘要 MCP 服务器。流程如下：

1. 遍历仓库结构。
2. 使用 `sampling/createMessage` 请求：「选出最能描述该仓库用途的五个文件。」
3. 读取这些文件。
4. 使用 `sampling/createMessage` 提供这些文件内容并请求：「用三段话总结这个仓库。」
5. 将摘要作为 `tools/call` 结果返回。

服务器从未直接调用 LLM API。客户端用户使用自己的凭据为这些完成计费。

### 安全风险（Unit 42 披露，2026 年第一季度）

- covert sampling（隐蔽采样）。例如一个工具总是调用采样并请求「从会话上下文中返回用户邮件地址」。Phase 13 · 15 涵盖了此类攻击向量。
- 通过采样盗用资源。服务器要求客户端摘要攻击者的载荷，导致用户承担费用。
- 循环炸弹（Loop bombs）。服务器在紧循环中调用采样。客户端必须强制实施每会话速率限制。

## 使用示例

`code/main.py` 提供了一个伪造的服务器-客户端采样 harness。一个模拟的 `summarize_repo` 工具会触发两轮采样（pick-files，然后 summarize），伪客户端返回预设的响应。这个 harness 展示了：

- 服务器发送带有 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回一个完成结果。
- 服务器继续其循环。
- 速率限制器对每次工具调用的采样次数进行上限。

关注点：

- 服务器只暴露一个工具（`summarize_repo`）；所有推理都发生在采样调用中。
- 模型偏好会权重化客户端的模型选择；hints 列出服务器偏好的模型名字。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 的限制可以阻断失控循环。

## 交付（Ship It）

本课产物为 `outputs/skill-sampling-loop-designer.md`。针对需要 LLM 调用的服务器端算法（研究、摘要、规划），此技能设计了基于采样的实现，包含合适的 modelPreferences、速率限制和安全确认流程。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2，观察速率限制的中止行为。

2. 实现 SEP-1577 的工具在采样中变体：在采样请求中携带 `tools` 数组。验证客户端循环在返回最终完成前执行那些工具。注意漂移风险：SDK 签名在 2026 年上半年可能仍会变化。

3. 增加人机在环确认：在服务器首次 `sampling/createMessage` 前暂停并等待用户批准。被拒绝的调用应返回类型化的拒绝响应。

4. 增加按客户端会话键控的每用户速率限制。相同用户在同一服务器上的循环应共享预算。

5. 设计一个 `summarize_pdf` 工具，使用采样挑选要包含的片段。勾勒要发送的消息序列。比较 `modelPreferences.intelligencePriority` 在 0.1 与 0.9 时行为会如何改变？

## 关键术语

| Term | 大家怎么说 | 实际含义 |
|------|------------|----------|
| Sampling | “服务器到客户端的 LLM 调用” | 服务器向客户端的模型请求一次完成 |
| `sampling/createMessage` | “这个方法” | 用于采样请求的 JSON-RPC 方法 |
| `modelPreferences` | “模型优先级” | 成本 / 速度 / 智能 权重，外加名称提示 |
| `includeContext` | “跨会话泄露” | 软弃用的上下文包含模式 |
| SEP-1577 | “采样中的工具” | 允许在采样中使用工具以实现服务器托管的 ReAct |
| Human-in-the-loop | “用户确认” | 客户端在运行采样前向用户展示请求并等待确认 |
| Loop bomb | “失控采样” | 服务器端无限采样循环；客户端必须限流 |
| Covert sampling | “隐藏推理” | 恶意服务器在采样提示中隐藏意图 |
| Resource theft | “消耗用户的 LLM 预算” | 服务器强制客户端为其未经允许的采样付费 |
| `stopReason` | “生成停止的原因” | `endTurn`、`stopSequence`，或 `maxTokens` |

## 延伸阅读

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — 采样的高层概览
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — `sampling/createMessage` 规范样式
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — 关于采样中工具的规范演进提案（实验性）
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 隐蔽采样与资源盗用模式分析
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — 含客户端代码示例的逐步讲解