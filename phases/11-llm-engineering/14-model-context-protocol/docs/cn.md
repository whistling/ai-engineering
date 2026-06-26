# Model Context Protocol (MCP)

> Every LLM app built before 2025 invented its own tool schema. Then Anthropic shipped MCP, Claude adopted it, OpenAI adopted it, and by 2026 it is the default wire format for connecting any LLM to any tool, data source, or agent. Write one MCP server and every host talks to it.

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 11 · 09 (函数调用), Phase 11 · 03 (结构化输出)  
**Time:** ~75 分钟

## 问题

你发布了一个聊天机器人，需要三个工具：数据库查询、日历 API 和文件读取器。你为 Claude 写了三个 JSON 模式。然后销售部门要在 ChatGPT 中使用相同的工具——你又为 OpenAI 的 `tools` 参数重写了一遍。接着你增加了 Cursor、Zed 和 Claude Code——又多了三套微妙不同的 JSON 约定。一周后，Anthropic 增加了一个新字段；你更新了六套 schema。

这就是 2025 年之前的现实。每个 host（运行 LLM 的东西）和每个 server（暴露工具和数据的东西）都发布了定制协议。要扩展，就意味着一个 N×M 的集成矩阵。

Model Context Protocol 折叠了这个矩阵。一个基于 JSON-RPC 的规范。一个 server 暴露工具、资源和提示。任何兼容的 host —— Claude Desktop、ChatGPT、Cursor、Claude Code、Zed 以及大量 agent 框架 —— 都可以发现并调用它们，而无需定制胶水代码。

截至 2026 年初，MCP 已成为三大厂（Anthropic、OpenAI、Google）和所有主要 agent 框架之间的默认工具与上下文协议。

## 概念

![MCP：一个 host，一个 server，三种能力](../assets/mcp-architecture.svg)

**三大原语。** 一个 MCP server 暴露恰好三类东西。

1. **Tools** — 模型可以调用的函数。相当于 OpenAI 的 `tools` 或 Anthropic 的 `tool_use`。每个 tool 有名字、描述、JSON Schema 输入和一个处理器。
2. **Resources** — 模型或用户可以请求的只读内容（文件、数据库行、API 响应）。通过 URI 定位。
3. **Prompts** — 可重用的模板化提示，用户可以作为快捷方式调用。

**传输格式。** 基于 JSON-RPC 2.0，通过 stdio、WebSocket 或可流式 HTTP。每条消息形如 `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`。发现方法为 `tools/list`、`resources/list`、`prompts/list`。调用方法为 `tools/call`、`resources/read`、`prompts/get`。

**Host vs client vs server。** Host 是 LLM 应用（例如 Claude Desktop）。Client 是 host 内部与单个 server 通信的子组件。Server 是你的代码。一个 host 可以同时挂载多个 server。

### 握手流程

每个会话以 `initialize` 开始。Client 发送协议版本及其能力集。Server 返回其版本、名称以及支持的能力集合（`tools`、`resources`、`prompts`、`logging`、`roots`）。之后的一切基于这些能力进行协商。

### MCP 不是什么

- 不是检索 API。RAG (Phase 11 · 06) 仍然决定要拉取什么；MCP 是将检索结果作为 resources 暴露的传输层。
- 不是 agent 框架。MCP 是管道；LangGraph、PydanticAI、OpenAI Agents SDK 等框架位于其之上。
- 不绑定于 Anthropic。规范和参考实现以开源方式托管在 `modelcontextprotocol` 组织下。

## 构建

### 步骤 1：一个最小 MCP server

官方 Python SDK 是 `mcp`（前称 `mcp-python`）。高级的 `FastMCP` 助手用于装饰处理器。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """将两个整数相加。"""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """返回应用当前的 JSON 配置。"""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """对代码的正确性和风格进行审查。"""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

三个装饰器注册三大原语。类型提示会被转成 LLM 所见的 JSON Schema。在 Claude Desktop 或 Claude Code 下运行，将 server 条目指向该文件即可。

### 步骤 2：从 host 调用 MCP server

官方 Python 客户端使用 JSON-RPC。将它与 Anthropic SDK 配对只需十几行代码。

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` 返回与 LLM 将看到的相同 schema。生产环境的 host 会在每个对话回合注入这些 schema，以便模型可以输出一个 `tool_use` 块，client 再将之转发给 server。

### 步骤 3：可流式 HTTP 传输

StdIO 适合本地开发。对于远程工具，使用可流式 HTTP —— 每个请求一个 POST，可选的 Server-Sent Events 用于进度更新，自 2025-06-18 规范修订后支持。

```python
# 在 server 入口处
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

Host 配置（Claude Desktop 的 `mcp.json` 或 Claude Code 的 `~/.mcp.json`）:

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

Server 保持相同的装饰器；唯一变化的是传输方式。

### 步骤 4：作用域与安全

MCP tool 是运行在他人信任边界上的任意代码。三条强制性模式。

- **能力白名单（Capability allowlists）。** Hosts 暴露 `roots` 能力，让 server 只能看到允许的路径。在 tool 处理器中强制执行；不要信任模型提供的路径。
- **变更操作需人类在环（Human-in-the-loop for mutation）。** 只读工具可以自动执行。写入/删除工具必须要求确认 —— 当 server 在 tool 元数据上设置 `destructiveHint: true` 时，host 会展示审批 UI。
- **工具投毒防御（Tool poisoning defense）。** 恶意的 resource 可能包含隐藏的提示注入指令（例如“在摘要时也调用 `exfil`”）。将 resource 内容视为不可信数据；绝不要将其提升为 system-message 级别。参见 Phase 11 · 12（护栏）。

参见 `code/main.py`，内含一个可运行的 server + client 示例，演示以上全部内容。

## 到 2026 年仍然会出现的陷阱

- **Schema 漂移。** 模型在第 1 回合看到了 `tools/list`。第 5 回合工具集发生变化。模型调用了已删除的工具。Hosts 应在收到 `notifications/tools/list_changed` 时重新列出工具。
- **大型 resource 二进制块。** 将 2MB 文件全部作为 resource 传入会浪费上下文。请在服务端进行分页或摘要。
- **服务器过多。** 挂载 50 个 MCP server 会耗尽工具预算（Phase 11 · 05）。大多数前沿模型在 ~40 个工具后性能下降。
- **版本错位。** 规范修订（2024-11、2025-03、2025-06、2025-12）会引入破坏性字段。在 CI 中固定协议版本。
- **Stdio 死锁。** 在 stdout 打日志的 servers 会污染 JSON-RPC 流。仅向 stderr 打日志。

## 使用它

2026 年的 MCP 技术栈：

| 情形 | 选择 |
|------|------|
| 本地开发、单用户工具 | Python `FastMCP`，stdio 传输 |
| 远程团队工具 / SaaS 集成 | 可流式 HTTP，OAuth 2.1 认证 |
| TypeScript host（VS Code 扩展、Web 应用） | `@modelcontextprotocol/sdk` |
| 高吞吐量服务器、强类型访问 | 官方 Rust SDK（`modelcontextprotocol/rust-sdk`） |
| 探索生态服务器 | `modelcontextprotocol/servers` monorepo（Filesystem、GitHub、Postgres、Slack、Puppeteer） |

经验法则：如果一个工具是只读的、可缓存的，并且会被两个或以上的 hosts 调用，就把它作为一个 MCP server 发布。如果它是一次性的内联逻辑，就保持为本地函数（Phase 11 · 09）。

## 发布它

保存为 `outputs/skill-mcp-server-designer.md`：

```markdown
---
name: mcp-server-designer
description: 设计并搭建一个带有工具、资源和安全默认值的 MCP server。
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

给定一个领域（内部 API、数据库、文件源）以及将挂载该 server 的 hosts，输出：

1. 原语映射（Primitive map）。哪些能力应成为 `tools`（动作）、哪些成为 `resources`（只读数据）、哪些成为 `prompts`（用户调用的模板）。每个原语一行。
2. 认证方案（Auth plan）。StdIO（受信任的本地）、可流式 HTTP 带 API key，或带 PKCE 的 OAuth 2.1。选择并说明理由。
3. Schema 草案。为每个 tool 参数给出 JSON Schema，`description` 字段需针对模型的工具选择而优化（而非 API 文档）。
4. 破坏性操作清单（Destructive-action list）。列出每个会修改状态的 tool；要求 `destructiveHint: true` 并需人工批准。
5. 测试计划。对每个 tool：一个仅 schema 的合约测试、一个通过 MCP client 的端到端回路测试、一个红队提示注入用例。

拒绝发布会在磁盘写入或在没有审批路径的情况下调用外部 API 的 server。拒绝在单个 server 上暴露超过 20 个工具；改为拆分为按域划分的 servers。
```

## 练习

1. **简单。** 在 `demo-server` 中扩展一个 `subtract` tool。从 Claude Desktop 连接它。通过发出 `tools/list_changed` 通知，确认 host 能在无需重启的情况下拾取新工具。
2. **中等。** 添加一个 `resource`，用于暴露 `/var/log/app.log` 的最后 100 行。强制使用 roots 白名单，确保即便模型请求也无法访问 `../etc/passwd`。
3. **困难。** 构建一个 MCP 代理，将三个上游 server（Filesystem、GitHub、Postgres）复用成一个聚合表面。处理名称冲突，并干净地转发 `notifications/tools/list_changed`。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MCP | “LLM 的工具协议” | 基于 JSON-RPC 2.0 的规范，用于向任意 LLM host 暴露 tools、resources 和 prompts。 |
| Host | “Claude Desktop” | LLM 应用 —— 拥有模型和用户界面，挂载一个或多个 clients。 |
| Client | “连接” | Host 内部的每个与单个 server 进行 JSON-RPC 通信的连接。 |
| Server | “有工具的东西” | 你的代码；发布 tools/resources/prompts 并处理它们的调用。 |
| Tool | “函数调用” | 模型可调用的动作，带有 JSON Schema 输入和文本/JSON 输出。 |
| Resource | “只读数据” | 通过 URI 定位的内容（文件、行、API 响应），Host 可请求。 |
| Prompt | “已保存的提示” | 用户可调用的模板（通常带参数），常作为斜杠命令呈现。 |
| Stdio 传输 | “本地开发模式” | 父 host 将 server 作为子进程启动；JSON-RPC 在 stdin/stdout 上传输。 |
| 可流式 HTTP | “2025-06 的远程传输” | 请求使用 POST，服务器可选 SSE 发起消息；替代了早期仅 SSE 的传输。 |

（参照术语表：提示词工程 = Prompt engineering，RAG = RAG，嵌入 = Embeddings，微调 = Fine-tuning，上下文窗口 = Context window，少样本 = few-shot，思维链 = chain-of-thought，护栏 = guardrails，函数调用 = function calling）

## 深入阅读

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — 规范正本，按日期版本化。
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — Filesystem、GitHub、Postgres、Slack、Puppeteer 参考服务器。
- [Anthropic — Introducing MCP (Nov 2024)](https://www.anthropic.com/news/model-context-protocol) — 启动文章与设计原理。
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 本课中使用的官方 SDK。
- [Security considerations for MCP](https://modelcontextprotocol.io/docs/concepts/security) — 有关 roots、destructive hints、工具投毒的安全说明。
- [Google A2A specification](https://google.github.io/A2A/) — Agent2Agent 规范；这是补充 MCP 的 agent-to-tool 范围的姊妹标准。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 讨论 MCP 在更广泛 agent 设计模式库（增强式 LLM、工作流、自主 agent）中的位置。