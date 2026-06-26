# Building an MCP Client — Discovery, Invocation, Session Management

> 大多数 MCP 内容发布的是服务器端教程，并对客户端只带过场一句。客户端代码才是困难的编排所在：进程生成、能力协商、跨多个服务器的工具列表合并、采样回调、重连以及命名空间冲突解决。本课构建一个多服务器客户端，将三个不同的 MCP 服务器提升为对模型暴露的一个平坦工具命名空间。

**Type:** 构建  
**Languages:** Python（stdlib，多服务器 MCP 客户端）  
**Prerequisites:** Phase 13 · 07（构建 MCP 服务器）  
**Time:** ~75 分钟

## Learning Objectives

- 以子进程方式生成 MCP 服务器，完成 `initialize` 并发送 `notifications/initialized`。
- 维护每个服务器的会话状态（能力、工具列表、最后看到的通知 id）。
- 将来自多个服务器的工具列表合并为一个命名空间并处理冲突。
- 将工具调用路由到拥有该工具的服务器并重新组合响应。

## The Problem

一个真实的 agent 主机（Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个 MCP 服务器。用户可能同时运行一个文件系统服务器、一个 Postgres 服务器和一个 GitHub 服务器。客户端的工作是：

1. 生成每个服务器进程。
2. 与每个服务器分别完成握手。
3. 对每个服务器调用 `tools/list` 并将结果扁平化。
4. 当模型发出 `notes_search` 时，在合并命名空间中查找并路由到正确的服务器。
5. 处理来自任意服务器的通知（例如 `tools/list_changed`），且不阻塞。
6. 在传输失败时重连。

把这些功能手工实现，才把“玩具”与“可用服务”区分开来。官方 SDK 封装了这些，但心智模型需要你自己理解。

## The Concept

### Child-process spawning

使用 `subprocess.Popen`，参数 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1` 并使用文本模式以便逐行读取。每个服务器是一个进程；客户端为每个服务器保存一个 `Popen` 句柄。

### Per-server session state

每个服务器一个 `Session` 对象包含：

- `process` — Popen 句柄。
- `capabilities` — 在 `initialize` 时服务器声明的能力。
- `tools` — 最近一次的 `tools/list` 结果。
- `pending` — 请求 id 到等待响应的 promise/future 的映射。

请求本质上是异步的；在服务器 B 正在处理中时向服务器 A 发送 `tools/call` 不应阻塞。可以使用带队列的线程或 asyncio。

### Merged namespace

当客户端看到聚合后的工具列表时，名称可能会冲突。两个服务器可能都暴露 `search`。客户端有三种选择：

1. **按服务器名加前缀。** `notes/search`、`files/search`。清晰但丑陋。
2. **沉默的先到者优先。** 后来的 `search` 覆盖早先的。风险：隐藏冲突。
3. **冲突拒绝。** 拒绝加载第二个服务器；通知用户。对安全敏感的主机最安全。

Claude Desktop 使用按服务器前缀方式。Cursor 使用冲突拒绝并给出清晰错误。VS Code MCP 也采用按服务器前缀。

### Routing

合并后，调度表将映射 `tool_name -> session`。模型按名称发出调用；客户端查找对应 session，将 `tools/call` 消息写入该服务器的 stdin，然后等待响应。

### Sampling callback

如果服务器在 `initialize` 时声明了 `sampling` 能力，则可能发送 `sampling/createMessage` 请求客户端运行其 LLM。客户端必须：

1. 在采样完成之前阻止对该服务器的进一步请求，或者如果实现支持并行化则进行流水线处理。
2. 调用其 LLM 提供器。
3. 将响应发送回服务器。

第 11 课覆盖了采样的端到端流程。本课为完整性做了桩实现（stub）。

### Notification handling

`notifications/tools/list_changed` 意味着重新调用 `tools/list`。`notifications/resources/updated` 意味着如果该资源正被使用，则重新读取。通知不能产生响应——不要尝试去 ack 它们。

一个常见的客户端错误：在 `tools/call` 上阻塞读取循环，而流中还有通知。使用后台读取线程将每条消息推到队列；主线程从队列出列并分发。

### Reconnection

传输可能失败：服务器崩溃、操作系统终止了进程、stdio 管道断开。客户端检测到 stdout 的 EOF 就将该会话视为已死。选项包括：

- 静默重启服务器并重新握手。适用于纯只读服务器。
- 向用户展示失败。适用于有用户可见会话状态的有状态服务器。

Phase 13 · 09 覆盖 Streamable HTTP 的重连语义；stdio 更简单。

### Keepalive and session id

Streamable HTTP 使用 `Mcp-Session-Id` 头。stdio 没有会话 id —— 进程身份即是会话。保活 ping 可选；stdio 管道在空闲时不会断开。

## Use It

`code/main.py` 以子进程方式生成三个模拟的 MCP 服务器，与每个服务器握手，合并它们的工具列表，并将工具调用路由到正确的服务器。这里的“服务器”实际上是运行简单响应器的其他 Python 进程（没有真实的 LLM）。运行它可以看到：

- 三次初始化，每个都有自己的能力集合。
- 三个 `tools/list` 结果合并为一个包含 7 个工具的命名空间。
- 基于工具名称的路由决策。
- 通过命名空间前缀避免冲突。

注意观察：

- `Session` dataclass 清晰地保存了每个服务器的状态。
- 后台读取线程逐行出队 stdout，不会阻塞主线程。
- 调度表是一个简单的 `dict[str, Session]`。
- 冲突处理是明确的：当两个服务器声明相同名称时，后者会被重命名并加上前缀。

## Ship It

本课产出 `outputs/skill-mcp-client-harness.md`。给定声明式的 MCP 服务器列表（名称、命令、参数），该 skill 会生成一个启动它们、合并工具列表并提供带冲突解决的路由函数的 harness。

## Exercises

1. 运行 `code/main.py` 并观察服务器生成日志。对其中一个模拟服务器进程发送 SIGTERM 并观察客户端如何检测到 EOF 并将该会话标记为已死。

2. 实现命名空间前缀。当两个服务器都暴露 `search` 时，将第二个重命名为 `<server>/search`。更新调度表并验证工具调用正确路由。

3. 为服务器重启添加连接池式回退：对连续失败使用指数回退，最大为 30 秒，在三次失败后向用户发出通知。

4. 设计一个支持 100 个并发 MCP 服务器的客户端。哪个数据结构可以替代简单的调度字典？（提示：用于前缀命名空间的 trie，加上衡量每个服务器工具数量的指标。）

5. 将客户端移植到官方 MCP Python SDK。SDK 封装了 `stdio_client` 和 `ClientSession`。代码应从约 ~200 行缩减到 ~40 行，同时保留多服务器路由功能。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MCP client | "The agent host" | 生成服务器并协调工具调用的进程 |
| Session | "Per-server state" | 能力、工具列表和挂起请求的记账 |
| Merged namespace | "One tool list" | 所有激活服务器的平坦工具名称集合 |
| Namespace collision | "Two servers same tool" | 客户端必须对重复项进行前缀、拒绝或先到者策略 |
| Routing | "Who gets this call?" | 从工具名到所属服务器的调度 |
| Background reader | "Non-blocking stdout" | 将服务器 stdout 清空到队列的线程或任务 |
| Sampling callback | "LLM-as-a-service" | 处理来自服务器的 `sampling/createMessage` 的客户端处理器 |
| `notifications/*_changed` | "Primitive mutated" | 表示客户端必须重新发现或重新读取 |
| Reconnection policy | "When server dies" | 传输失败时的重启语义 |
| Stdio session | "Process = session" | 无会话 id；子进程生命周期即是会话 |

## Further Reading

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — 规范化的客户端行为  
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的入门客户端教程  
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — 参考 `ClientSession` 和 `stdio_client`  
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — TypeScript 并行实现  
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个编辑器宿主内多路复用多个 MCP 服务器