# A2A — Agent-to-Agent 协议

> MCP 是 agent-to-tool。A2A (Agent2Agent) 是 agent-to-agent——一个让基于不同框架的不可见（opaque）代理相互协作的开放协议。由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux 基金会，2026 年 4 月达到 v1.0，拥有包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow 在内的 150+ 支持者。它吸收了 IBM 的 ACP 并加入了 AP2 支付扩展。本课程讲解 Agent Card、Task 生命周期以及两种传输绑定。

**Type:** 构建  
**Languages:** Python（stdlib，Agent Card + Task harness）  
**Prerequisites:** Phase 13 · 06 (MCP 基础), Phase 13 · 08 (MCP 客户端)  
**Time:** ~75 分钟

## 学习目标

- 区分 agent-to-tool（MCP）与 agent-to-agent（A2A）的使用场景。  
- 在 `/.well-known/agent.json` 发布包含技能和端点元数据的 Agent Card。  
- 演练 Task 生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。  
- 使用包含 Parts（text、file、data）的 Messages 和作为输出的 Artifacts。

## 问题背景

一个客服代理需要将报告撰写任务委派给一个专门的写作代理。A2A 出现之前的选项：

- 自定义 REST API。可行，但每次配对都是一次性实现。  
- 共享代码库。要求两个代理运行相同的框架。  
- MCP。并不契合：MCP 用于调用工具，而不是在保留每个代理内部推理不可见的同时，让两个代理协作。

A2A 弥补了这一空白。它将交互建模为一个代理向另一个代理发送 Task，带有生命周期、消息和工件。被调用代理的内部状态保持不可见——调用方只看到任务状态的变迁和最终输出。

A2A 是“让跨框架的代理互相通信”的协议。它并不替代 MCP；两者互补。

## 概念

### Agent Card

每个遵循 A2A 的代理在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现（Discovery）基于 URL：获取卡片，得知 A2A 端点的 URL，枚举其技能。

### 签名的 Agent Cards（AP2）

AP2 扩展（2025 年 9 月）为 Agent Card 添加了加密签名。发布者使用 JWT 对自己的卡片签名；消费者进行验证。防止冒充。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

客户端通过 `tasks/send` 发起。被调用代理在不同状态间切换；客户端通过 SSE 订阅状态更新或轮询。

### Messages 与 Parts

一个 message 携带一个或多个 Parts：

- `text` — 纯文本内容。  
- `file` — base64 二进制块并带有 mimeType。  
- `data` — 类型化的 JSON 载荷（为被调用代理提供结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

输出是 Artifacts，而不是原始字符串。Artifact 是有名称、有类型的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以作为分块流式传输。调用者负责累积这些分块。

### 两种传输绑定

1. **JSON-RPC over HTTP。** `/a2a` 端点，POST 用于请求，可选的 SSE 用于流式传输。默认绑定。  
2. **gRPC。** 适用于以 gRPC 为原生的企业环境。

两种绑定承载相同的逻辑消息结构。

### 不可见性（Opacity）保持

设计原则之一：被调用代理的内部状态保持不可见。调用方只看到任务状态和 Artifacts。被调用代理的思维链（chain-of-thought）、工具调用、子代理委派——这些全部对调用方不可见。这与 MCP 不同，MCP 中工具调用是透明的。

其理由：A2A 允许竞争对手在不暴露内部实现的情况下协作。你可以说“调用这个客服代理”，而不用让调用方了解该代理如何实现服务。

### 时间线

- **2025-04-09。** Google 宣布 A2A。  
- **2025-06-23。** 捐赠给 Linux 基金会。  
- **2025-08。** 吸收 IBM 的 ACP。  
- **2025-09。** AP2 扩展（Agent Payments）发布。  
- **2026-04。** v1.0 发布，拥有 150+ 支持组织。

### 与 MCP 的关系

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想调用一个特定工具时使用 MCP。当你想把一个完整任务委派给另一个代理时使用 A2A。许多生产系统同时使用两者：代理在工具层使用 MCP，在协作层使用 A2A。

## 使用示例

`code/main.py` 实现了一个最小的 A2A 框架：一个 research agent 发布它的卡片，一个 writer agent 接收带有 PDF 和文本指令的 `tasks/send`，状态按序从 working → input_required → working → completed，并返回一个文本类型的 artifact。全部使用 stdlib；使用内存传输以便关注消息格式。

需要关注的点：

- Agent Card 的 JSON 结构。  
- Task id 分配与状态转换。  
- 混合类型 parts 的 Messages。  
- 任务中途进入 input-required 分支。  
- 完成时返回 Artifact。

## 交付物

本课程会生成 `outputs/skill-a2a-agent-spec.md`。给定一个应被其他代理调用的新代理，该技能会生成 Agent Card JSON、技能模式和端点蓝图。

## 练习

1. 运行 `code/main.py`。跟踪完整的 Task 生命周期，包括被调用代理在 input-required 暂停时提出澄清请求的情形。  

2. 添加一个签名的 Agent Card。对卡片的规范化 JSON 使用 HMAC 签名。编写一个验证器并确认在卡片被篡改时验证失败。  

3. 实现任务流式传输：writer agent 在 SSE 上分三次发出递增的 artifact 块，调用方负责累积这些块。  

4. 设计一个将 MCP 服务器包装为 A2A 代理的方案。将每个 MCP 工具映射为一个 A2A skill。注意权衡——会丢失哪些不可见性？  

5. 阅读 A2A v1.0 的公告，找出截至 2026 年 4 月仍未被任何框架实现的那个功能。（提示：与多跳任务委派有关。）

## 术语要点

| 术语 | 大家怎么说 | 它实际意味着 |
|------|-----------|--------------|
| A2A | "Agent-to-Agent protocol" | 用于不可见代理协作的开放协议 |
| Agent Card | "`.well-known/agent.json`" | 描述代理技能和端点的已发布元数据 |
| Skill | "A callable unit" | 代理支持的命名操作（类似于 MCP 的工具） |
| Task | "Unit of delegation" | 带有生命周期和最终 artifact 的工作项 |
| Message | "Task input" | 携带 Parts（text、file、data） |
| Part | "Typed chunk" | message 中的 `text` / `file` / `data` 元素 |
| Artifact | "Task output" | 在完成时返回的命名、类型化输出 |
| AP2 | "Agent Payments Protocol" | 用于信任和支付的签名 Agent Cards 扩展 |
| Opacity | "Black-box collaboration" | 被调用代理的内部实现对调用方隐藏 |
| Input-required | "Task pause" | 当代理需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 权威的 A2A 规范  
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现与 SDK  
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月的治理转移新闻稿  
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图与合作伙伴动向  
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布说明与向后兼容指导