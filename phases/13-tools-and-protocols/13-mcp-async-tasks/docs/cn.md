# Async Tasks (SEP-1686) — Call-Now, Fetch-Later for Long-Running Work

> 实际的 agent 工作需要数分钟到数小时：CI 运行、深度研究汇总、批量导出。同步工具调用会断开连接、超时或阻塞 UI。SEP-1686（于 2025-11-25 合入）增加了 Tasks 原语：任何请求都可以增强为任务，结果可以稍后获取或通过状态通知流式传输。漂移风险提醒：Tasks 在 2026 年上半年仍为实验性；SDK 接口还在根据规范设计中。

**Type:** 构建  
**Languages:** Python（stdlib，异步任务状态机）  
**Prerequisites:** Phase 13 · 07 (MCP 服务器), Phase 13 · 09 (传输层)  
**Time:** ~75 分钟

## 学习目标

- 识别何时将工具从同步提升为支持任务增强（服务器端工作超过 ~30 秒）。
- 理解任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态以防止崩溃丢失进行中的工作。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## 问题场景

一个 `generate_report` 工具运行多分钟的抽取流水线。在同步模型下的选项：

1. 保持连接三分钟。远端传输会中断；客户端超时；UI 冻结。
2. 立即返回占位符；要求客户端轮询自定义端点。破坏 MCP 的统一性。
3. fire-and-forget；没有结果。

都不理想。SEP-1686 增加了第四种：任务增强。任何请求（通常是 `tools/call`）都可以被标记为任务。服务器立即返回任务 id。客户端轮询 `tasks/status` 并在完成后获取 `tasks/result`。服务器端状态在重启时仍然保留。

## 概念

### 任务增强

通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定），请求即可变为任务。服务器会立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器承诺保留状态的时间；超过 ttl 后任务结果会被丢弃。

### 每个工具的可选加入

工具注解可以声明任务支持：

- `taskSupport: "forbidden"` — 该工具始终同步运行。适用于快速工具。
- `taskSupport: "optional"` — 客户端可请求任务增强。
- `taskSupport: "required"` — 客户端必须使用任务增强。

`generate_report` 工具应为 `required`。`notes_search` 工具应为 `forbidden`。

### 状态

```
working  -> input_required -> working  (通过询问循环)
working  -> completed
working  -> failed
working  -> cancelled
```

状态机为追加式：一旦进入 `completed`、`failed` 或 `cancelled`，任务即为终态。

### 方法

- `tasks/status {taskId}` — 返回当前状态和进度提示。
- `tasks/result {taskId}` — 若未完成则阻塞或返回 404。
- `tasks/cancel {taskId}` — 幂等；对终态无效。
- `tasks/list` — 可选；列举活动和最近完成的任务。

### 流式状态变更

当服务器支持时，客户端可以订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

使用流式订阅而非轮询的客户端能获得更好的用户体验。轮询始终作为最小化的支持面存在。

### 持久化状态

规范要求声明任务支持的服务器必须持久化状态。崩溃不应丢失在 ttl 内的已完成结果。存储可以是 SQLite、Redis 或文件系统。第 13 课的 harness 使用文件系统。

### 取消语义

`tasks/cancel` 为幂等操作。如果任务正在执行，服务器会尝试停止（检查执行器是否支持协作式取消）。如果已处于终态，请求为无操作。

### 崩溃恢复

服务器进程重启时：

1. 加载所有持久化的任务状态。
2. 将任何仍标记为 `working` 且其进程已终止的任务标记为带错误 `CRASH_RECOVERY` 的 `failed`。
3. 在 ttl 内保留 `completed` / `failed` / `cancelled` 状态。

### 异步任务加上采样

任务本身可以调用 `sampling/createMessage`。这就是长时间运行的研究任务的工作方式：服务器的任务线程根据需要对客户端模型进行采样，同时客户端 UI 将任务显示为 `working` 并周期性地更新进度。

### 为什么这是实验性的

SEP-1686 于 2025-11-25 发布，但更广的路线图指出三个未决问题：耐久订阅原语、子任务（父子任务关系），以及结果 TTL 的标准化。预计规范将在 2026 年持续演进。生产代码应仅将 Tasks 视为常见用例的稳定实现，并为将来 SDK 在子任务等方面的变化做好防范。

## 使用方法

`code/main.py` 实现了一个持久化任务存储（基于文件系统）和在后台线程运行的 `generate_report` 工具。客户端调用该工具，立即获得任务 id；在 worker 更新进度时轮询 `tasks/status`，并在完成时获取 `tasks/result`。取消功能可用；崩溃恢复通过杀死 worker 线程并重新加载状态来模拟。

关注点：

- 任务状态 JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- worker 线程更新 `progress` 字段；轮询会看到其推进。
- 客户端发出的取消会设置一个事件；worker 会检查并提前退出。
- 状态重载（“崩溃”）会将进行中的任务标记为带 `CRASH_RECOVERY` 的 `failed`。

## 交付物

本课产生 `outputs/skill-task-store-designer.md`。针对长时运行的工具（研究、构建、导出），该技能会设计任务存储（状态形状、ttl、持久性），选择合适的 `taskSupport` 标志，并草拟进度通知方案。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` 任务，轮询状态，然后获取结果。

2. 在运行中间加入 `tasks/cancel` 调用。验证 worker 是否响应并且状态变为 `cancelled`。

3. 模拟崩溃恢复：杀死 worker 线程，重启加载器，观察 `CRASH_RECOVERY` 的失败模式。

4. 将存储扩展为 SQLite。持久性优势不变；查询选项打开（例如列出来自会话 X 的所有任务）。

5. 阅读 MCP 2026 年路线图文章。识别最可能在下一年影响 SDK API 设计的与 Tasks 相关的未决问题。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Task | "Long-running tool call" | 请求被增强为带 `_meta.task` 的异步执行 |
| SEP-1686 | "Tasks spec" | 在 2025-11-25 增加 Tasks 的规范演进提案 |
| `_meta.task` | "Task envelope" | 每个请求的元数据，包含 id、state、ttl |
| taskSupport | "Tool flag" | 每个工具的 `forbidden` / `optional` / `required` 标记 |
| `tasks/status` | "Poll method" | 获取当前状态和可选的进度提示 |
| `tasks/result` | "Fetch result" | 返回已完成的负载，若未就绪返回 404 |
| `tasks/cancel` | "Stop it" | 幂等的取消请求 |
| ttl | "Retention budget" | 服务器承诺保留任务状态的毫秒数 |
| `notifications/tasks/updated` | "State push" | 服务器发起的状态变更事件 |
| Durable store | "Crash-safe state" | 文件系统 / SQLite / Redis 的持久化层 |

## 延伸阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 发起提案与完整讨论  
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 设计演练与原因说明  
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — 机制与状态机  
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDK 级任务实现模式  
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 未决问题与 2026 年重点（包括子任务）