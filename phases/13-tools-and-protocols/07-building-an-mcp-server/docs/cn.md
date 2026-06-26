# Building an MCP Server — Python + TypeScript SDKs

> Most MCP tutorials show only stdio hello-worlds. A real server exposes tools plus resources plus prompts, handles capability negotiation, emits structured errors, and works the same across SDKs. This lesson builds a notes server end-to-end: stdlib stdio transport, JSON-RPC dispatch, the three server primitives, and a pure-function style that drops into either the Python SDK's FastMCP or the TypeScript SDK when you graduate.

**Type:** 构建  
**Languages:** Python（stdlib，stdio MCP 服务器）  
**Prerequisites:** Phase 13 · 06（MCP 基础）  
**Time:** ~75 分钟

## Learning Objectives

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个从 stdin 读取 JSON-RPC 消息并将响应写入 stdout 的分发循环（dispatch loop）。
- 按照 JSON-RPC 2.0 规范及 MCP 的额外错误码发出结构化错误响应。
- 将 stdlib 实现升级到 FastMCP（Python SDK）或 TypeScript SDK，而无需重写工具逻辑。

## The Problem

在你可以使用远程传输（Phase 13 · 09）或认证层（Phase 13 · 16）之前，你需要一个干净的本地服务器。所谓本地即 stdio：服务器由客户端作为子进程启动，消息通过 stdin/stdout 按行流动。

2025-11-25 规范规定 stdio 消息以 JSON 对象编码并以显式的换行符 `\n` 分隔。这里不使用 SSE；SSE 是旧的远程模式，并将在 2026 年中期移除（Atlassian 的 Rovo MCP 服务器在 2026 年 6 月 30 日弃用；Keboola 在 2026 年 4 月 1 日弃用）。对于 stdio，线协议就是每行一个 JSON 对象。

notes 服务器是一个很好的例子，因为它涵盖了所有三个服务器原语。工具用于变更（`notes_create`）。资源用于暴露数据（`notes://{id}`）。提示词用于提供模板（`review_note`）。本课的结构可推广到任何领域。

## The Concept

### Dispatch loop

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何非 JSON-RPC 信封的内容。调试日志应写入 stderr。
- 每个请求必须对应一个带有相同 `id` 的响应。
- 通知（notification）不得被响应。

### Implementing `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只申报你支持的功能。客户端依赖能力集来决定可用特性。

### Implementing `tools/list` and `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个条目包含 `name`、`description`、`inputSchema`。`tools/call` 接受 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块是有类型的。最常见的示例：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形式。协议级错误（未知方法、参数错误）作为 JSON-RPC 错误返回。工具级错误（调用合法但工具执行失败）作为 `{content: [...], isError: true}` 返回。这让模型能在上下文中看到失败信息。

### Implementing resources

资源本质上是只读的。`resources/list` 返回清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...`，或自定义 scheme，比如 `notes://`。

当你把数据作为资源暴露而不是工具时：

- 模型不会“调用”它；客户端可以在用户请求时将其注入到上下文中。
- 订阅允许服务器在资源变化时推送更新（Phase 13 · 10）。
- Phase 13 · 14 在此基础上扩展了 `ui://` 用于交互式资源。

### Implementing prompts

提示词（prompts）是带命名参数的模板。主机将其呈现为斜杠命令（slash-commands）。一个 `review_note` 提示可能接受 `note_id` 参数并生成多消息的提示模板，供客户端传给其模型。

### Stdio transport subtleties

- 换行分隔的 JSON。没有长度前缀的分帧。
- 不要做输出缓冲。在每次写入后调用 `sys.stdout.flush()`。
- 客户端控制进程生命周期。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE；记录并退出。

### Annotations

每个工具可以携带描述安全属性的 `annotations`：

- `readOnlyHint: true` — 纯读取，可安全重试。
- `destructiveHint: true` — 不可逆的副作用；客户端应提示确认。
- `idempotentHint: true` — 相同输入产生相同输出。
- `openWorldHint: true` — 与外部系统交互。

客户端使用这些信息决定 UX（确认对话、状态指示）以及路由策略（Phase 13 · 17）。

### Graduation path

stdlib 实现位于 `code/main.py`，大约 180 行。FastMCP（Python）将相同逻辑压缩为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 也有等价的形式。当你准备好时可直接替换；概念（capabilities、dispatch、content blocks）均相同。

## Use It

`code/main.py` 是一个完整的基于 stdio 的 notes MCP 服务器，仅使用 stdlib。它处理 `initialize`、`tools/list`、`tools/call`（三个工具：`notes_list`、`notes_search`、`notes_create`）、每条笔记的 `resources/list` 与 `resources/read`，以及一个 `review_note` 提示。你可以通过管道 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

关注点：

- 分发器是以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器都返回内容块列表，而不是裸字符串。
- 当执行器抛出异常时，应设置 `isError: true`。

## Ship It

本课产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该脚手架会生成一个 MCP 服务器，正确划分工具 / 资源 / 提示并给出 SDK 升级路径。

## Exercises

1. 运行 `code/main.py` 并使用手工构造的 JSON-RPC 消息驱动它。执行 `notes_create`，然后用 `resources/read` 检索新笔记。

2. 添加一个带有 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端会弹出确认对话（这需要真实宿主；Claude Desktop 可用）。

3. 实现 `resources/subscribe`，使服务器在笔记被修改时推送 `notifications/resources/updated`。增加一个保活任务。

4. 将服务器移植到 FastMCP。Python 文件应缩减至 80 行以下。线端行为必须保持一致；用相同的 JSON-RPC 测试工具验证。

5. 阅读规范中 `server/tools` 节并找出本课服务器未实现的工具定义字段之一。（提示：有几个；选择一个并添加它。）

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MCP server | "The thing that exposes tools" | 以 stdio 或 HTTP 讲 MCP JSON-RPC 的进程 |
| stdio transport | "Child process model" | 服务器被客户端以子进程启动；通过 stdin/stdout 通信 |
| Dispatcher | "Method router" | 将 JSON-RPC 方法名映射到处理函数的映射表 |
| Content block | "Tool result chunk" | 工具响应中 `content` 数组的有类型元素 |
| `isError` | "Tool-level failure" | 表示工具失败；区别于 JSON-RPC 错误 |
| Annotations | "Safety hints" | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | "Python SDK" | 基于装饰器的高级 MCP 框架（Python） |
| Resource URI | "Addressable data" | 标识资源的 `file://`、`db://` 或自定义 scheme |
| Prompt template | "Slash-command brief" | 服务器提供的带参数槽的模板，供宿主 UI 使用 |
| Capability declaration | "Feature toggle" | 在 `initialize` 中为每个原语声明的功能开关 |

## Further Reading

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参考的 Python 实现（模型上下文协议）
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 平行的 TypeScript 实现（模型上下文协议）
- [FastMCP — server framework](https://gofastmcp.com/) — 基于装饰器的 Python MCP 服务器框架
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* 消息的完整参考文档