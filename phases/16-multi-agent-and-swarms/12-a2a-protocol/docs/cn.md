# A2A — Agent-to-Agent 协议

> Google 在 2025 年 4 月宣布 A2A；到 2026 年 4 月，该规范位于 https://a2a-protocol.org/latest/specification/，并有 150+ 个组织支持。A2A 是对 MCP（第13课）的横向补充：当 MCP 是垂直的（agent ↔ tools）时，A2A 是点对点的（agent ↔ agent）。它定义了 Agent Card（发现）、带有工件（文本、结构化数据、视频）的任务、不透明的任务生命周期以及认证。生产系统越来越多地将 MCP 与 A2A 配对使用。Google Cloud 在 2025–2026 年期间将 A2A 支持集成到 Vertex AI Agent Builder 中。

**Type:** 学习 + 构建  
**Languages:** Python（标准库，`http.server`，`json`）  
**Prerequisites:** Phase 16 · 04（原始模型）  
**Time:** ~75 分钟

## 问题

你的 agent 需要调用运行在另一台系统上的另一个 agent。怎么做？你可以暴露一个 HTTP 端点，定义一个定制的 JSON 模式，然后指望对端能理解。每一对 agent 都变成了定制集成。

A2A 是这种调用的通用线协议。标准发现、标准任务模型、标准传输、标准工件。就像为 agent 这一一级的第一类公民设计的 HTTP + REST。

## 概念

### 四个要素

**Agent Card。** 在 `/.well-known/agent.json` 上的 JSON 文档，描述 agent：名称、技能、端点、支持的模态、认证要求。通过读取 Agent Card 进行发现。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**任务（Task）。** 工作单元。一个异步、有状态的对象，具有生命周期：`submitted → working → completed / failed / canceled`。客户端发送任务，轮询或订阅以获取更新。

**工件（Artifact）。** 任务产生的结果类型。文本、结构化 JSON、图像、视频、音频。工件是有类型的，使不同模态成为一等公民。

**不透明生命周期。** A2A 不规定远程 agent 如何解决任务。客户端看到状态转换和工件；实现可以自由使用任何框架。

### MCP / A2A 的划分

- **MCP**（第13课）：agent ↔ tool。agent 通过 JSON-RPC 与工具服务器读/写。默认是无状态的。
- **A2A**：agent ↔ agent。点对点协议；双方都是各自有独立推理的 agent。

生产级多 agent 系统同时使用两者。一个 A2A 节点会在其一侧调用 MCP 工具。划分让两类关注点保持清晰。

### 发现流程

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或者使用流式：通过 SSE 订阅 `/tasks/{id}/events` 接收推送更新。

### 认证（Auth）

A2A 支持三种常见模式：

- **Bearer token** — 使用 OAuth2 或不透明令牌。
- **mTLS** — 双向 TLS；组织间相互验证身份。
- **Signed requests** — 对负载进行 HMAC 签名的请求。

认证在 Agent Card 中声明；客户端发现并遵循。

### 到 2026 年 4 月的 150+ 个组织

企业采用推动了 A2A 的规模化。要点是：A2A 成为了企业 agent 系统跨信任边界互通的方式。Google Cloud 发布了对 Vertex AI Agent Builder 的 A2A 支持；Microsoft Agent Framework 支持 A2A；大多数主流框架（LangGraph、CrewAI、AutoGen）都提供 A2A 适配器。

### A2A 的优势

- **跨组织调用。** 公司 A 的 agent 调用公司 B 的 agent。没有 A2A 时，每一对都是定制契约。
- **异构框架互通。** LangGraph 的 agent 调用 CrewAI 的 agent，再到自定义的 Python agent。A2A 实现了规范化。
- **类型化工件。** 视频结果、结构化 JSON、音频——都是一等媒体。
- **长时任务。** 不透明生命周期 + 轮询使得耗时数小时的任务易于处理。

### A2A 的局限

- **对延迟敏感的微调用。** A2A 的生命周期是异步的。亚毫秒级的 agent 间直连不适合；应使用直接 RPC。
- **紧耦合的进程内 agents。** 如果两个 agent 运行在同一 Python 进程内，A2A 的 HTTP 往返代价过高。
- **小团队。** 规范的开销是真实存在的；仅在内部使用的 agents 可能不需要这种形式化。

### A2A 与 ACP、ANP、NLIP 的比较

2024–2026 年间出现了几个相关规范：

- **ACP**（IBM/Linux 基金会）— A2A 的前身，范围更窄。
- **ANP**（Agent Network Protocol）— 强调对端发现、去中心化优先。
- **NLIP**（Ecma 自然语言交互协议，2025 年 12 月标准化）— 自然语言内容类型。

截至 2026 年 4 月，A2A 是采用率最高的对等协议。参见 arXiv:2505.02279（Liu 等，"A Survey of Agent Interoperability Protocols"）了解比较。

## 实现它

`code/main.py` 实现了一个 A2A 最小服务器和客户端，使用 `http.server` 和 JSON。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 返回工件。

客户端：

- 获取 Agent Card，
- 提交任务，
- 轮询直到完成，
- 读取工件。

运行：

```
python3 code/main.py
```

脚本在后台线程启动服务器，然后针对该服务器运行客户端。你将看到完整流程：发现、提交、轮询、工件。

## 使用它

`outputs/skill-a2a-integrator.md` 设计了一个 A2A 集成：Agent Card 内容、任务模式、认证选择、流式与轮询的取舍。

## 部署清单（Ship It）

检查项：

- **固定规范版本。** A2A 仍在演进；Agent Card 应声明协议版本。
- **幂等的任务创建。** 重复提交（网络重试）应生成同一个任务。
- **工件模式。** 声明 agent 返回的结构；消费者应进行验证。
- **限流 + 认证。** A2A 面向外部；应用标准的 Web 安全措施。
- **失败任务的死信队列。** 长期检查失败模式，定位重复出现的失败类型。

## 练习

1. 运行 `code/main.py`。确认客户端发现服务器并接收正确的工件。  
2. 给服务器添加第二个 skill（例如 "summarize"）。更新 Agent Card。编写一个客户端，根据任务类型选择 skill。  
3. 实现一个 SSE 流式端点：`/tasks/{id}/events`，发送状态变更。客户端需要做哪些不同的处理？  
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。识别规范要求但此演示未实现的三项内容。  
5. 将 A2A（Agent Card 发现）与 MCP（通过 `listTools` 的服务器端能力列举）进行比较。自描述 agent 与能力探测之间的权衡是什么？

## 关键术语

| 术语 | 人们怎么说 | 它实际上是什么意思 |
|------|-----------|--------------------|
| A2A | "Agent-to-agent" | 跨系统调用其他 agent 的点对点协议。Google 2025。 |
| Agent Card | "The agent's business card" | 位于 `/.well-known/agent.json` 的 JSON，描述技能、端点、认证。 |
| Task | "The unit of work" | 异步、有状态的对象，带生命周期；完成时产生工件。 |
| Artifact | "The result" | 类型化输出：文本、结构化 JSON、图像、视频、音频。作为一等媒体。 |
| Opaque lifecycle | "How it's solved is the agent's business" | 客户端看到状态转换；服务器可自由选择框架/工具。 |
| Discovery | "Finding the agent" | `GET /.well-known/agent.json` 返回 Agent Card。 |
| MCP vs A2A | "Tools vs peers" | MCP：垂直的 agent ↔ tool。A2A：横向的 agent ↔ agent。 |
| ACP / ANP / NLIP | "Sibling protocols" | 相关规范；截至 2026 年 A2A 是采用度最高的。 |

（注：文中涉及的术语如 Prompt engineering -> 提示词工程、RAG -> RAG、Embeddings -> 嵌入、Fine-tuning -> 微调、Context window -> 上下文窗口、few-shot -> 少样本、chain-of-thought -> 思维链、guardrails -> 护栏、function calling -> 函数调用、speculative decoding -> 投机性解码、positional embeddings -> 位置嵌入、self-attention -> 自注意力、instruction tuning -> 指令微调、distributed training -> 分布式训练、Model Context Protocol -> 模型上下文协议 等均可参考通用 AI 工程术语翻译。）

## 延伸阅读

- [A2A specification](https://a2a-protocol.org/latest/specification/) — 官方规范  
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月的发布文章  
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — 参考实现和 SDK  
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 的比较