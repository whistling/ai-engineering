# Claude Agent SDK: 子智能体 和 会话存储

> Claude Agent SDK 是 Claude Code harness 的库形式。内置工具、用于上下文隔离的子智能体、钩子、W3C 跟踪传播、会话存储接口一致性。Claude Managed Agents 是用于长期异步工作的托管替代方案。

**Type:** 学习 + 构建
**Languages:** Python（stdlib）
**Prerequisites:** Phase 14 · 01 (智能体循环), Phase 14 · 10 (技能库)
**Time:** ~75 分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（harness 形态）之间的区别。
- 描述子智能体（并行化与上下文隔离）以及何时使用它们。
- 说出 Python SDK 的会话存储表面（`append`, `load`, `list_sessions`, `delete`, `list_subkeys`）以及 `--session-mirror` 的作用。
- 实现一个基于 stdlib 的 harness，包含内置工具、带隔离上下文的子智能体生成、生命周期钩子和会话存储。

## 问题背景

原始的 LLM API 只能给你一次往返。生产级智能体需要工具执行、MCP 服务器、生命周期钩子、子智能体生成、会话持久化、追踪传播。Claude Agent SDK 将这种 harness 形态作为库发布——即 Claude Code 使用的相同 harness，以便为自定义智能体提供支持。

## 概念

### Client SDK vs Agent SDK

- Client SDK (`anthropic`)。原始 Messages API。你负责循环、工具和状态管理。
- Agent SDK (`claude-agent-sdk`)。内置工具执行、MCP 连接、钩子、子智能体生成、会话存储。把 Claude Code 的循环作为库提供。

### 内置工具

SDK 开箱提供 10+ 个工具：读写文件、shell、grep、glob、网页抓取等。自定义工具通过标准的 tool-schema 接口注册。

### 子智能体

Anthropic 文档列出的两种用途：

1. 并行化。并发运行独立工作。“为这 20 个模块各找对应的测试文件”就是 20 个并行子智能体任务。
2. 上下文隔离。子智能体使用各自的上下文窗口；仅把结果返回给编排者。编排者的预算得以保全。

Python SDK 最近的新增：`list_subagents()`、`get_subagent_messages()`，用于读取子智能体的对话记录。

### 会话存储

与 TypeScript 保持协议一致：

- `append(session_id, message)` — 添加一个回合。
- `load(session_id)` — 恢复对话。
- `list_sessions()` — 枚举会话。
- `delete(session_id)` — 支持级联删除子智能体的会话。
- `list_subkeys(session_id)` — 列出子智能体键。

`--session-mirror`（CLI 标志）在流式写入时把记录镜像到外部文件，用于调试。

### 钩子

可注册的生命周期钩子：

- `PreToolUse`, `PostToolUse` — 用于拦截或审计工具调用。
- `SessionStart`, `SessionEnd` — 设置与清理。
- `UserPromptSubmit` — 在模型看到用户输入之前对其进行处理。
- `PreCompact` — 在上下文压缩之前运行。
- `Stop` — 智能体退出时的清理。
- `Notification` — 侧通道通知。

钩子是 pro-workflow（参见 Phase 14 课程）等系统添加横切行为的方式。

### W3C 跟踪上下文

调用者上激活的 OTel span 会通过 W3C 跟踪上下文头传播到 CLI 子进程。整个多进程追踪在你的后端显示为一个完整的 trace。

### Claude Managed Agents

托管替代方案（beta 版本 header `managed-agents-2026-04-01`）。用于长期异步工作，内置提示缓存和内置压缩。以可控性换取托管基础设施。

### 该模式的误区

- 子智能体过度生成。为 100 个小任务生成 100 个子智能体，开销会盖过收益。应当批处理。
- 钩子泛滥。每个团队都加钩子会让启动时间暴涨。应季度审查钩子。
- 会话膨胀。会话堆积导致大小增长。使用 `list_sessions` + 过期策略。

## 实作

`code/main.py` 在 stdlib 中实现了 SDK 形态：

- `Tool`, `ToolRegistry`，包含内置的 `read_file`、`write_file`、`list_dir`。
- `Subagent` — 私有上下文，隔离运行，返回结果。
- `SessionStore` — 实现 `append`, `load`, `list`, `delete`, `list_subkeys`。
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`。
- 一个演示：主智能体并行生成 3 个子智能体（各自隔离），聚合结果并持久化会话。

运行它：

```
python3 code/main.py
```

追踪显示了子智能体的上下文隔离（编排者上下文大小保持有界）、钩子执行以及会话持久化。

## 使用场景

- 对于以 Claude 为首要目标、希望获得 Claude Code harness 形态的产品，使用 Claude Agent SDK。
- 对于托管的长期异步工作，使用 Claude Managed Agents。
- 对于以 OpenAI 为首的对应方案，参考 OpenAI Agents SDK（第 16 课）。
- 如果你想要图形化的状态机和自定义工具，可以选 LangGraph + 自定义工具。

## 上线交付

`outputs/skill-claude-agent-scaffold.md` 为 Claude Agent SDK 应用搭建脚手架，包含子智能体、钩子、会话存储、MCP 服务器附加和 W3C 跟踪传播。

## 练习

1. 添加一个子智能体生成器，将 20 个任务分批为每组 5 个并行子智能体。比较编排者上下文大小与为每个任务生成一个子智能体的差异。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用进行速率限制（每个会话每分钟 5 次）。追踪其行为。
3. 将 `list_subkeys` 连接到子智能体树的渲染。深度嵌套会是什么样子？
4. 将该玩具示例移植到真实的 `claude-agent-sdk` Python 包。工具注册有什么变化？
5. 阅读 Claude Managed Agents 文档。什么时候你会从自托管切换到托管方案？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| Agent SDK | "Claude Code as a library" | Harness 形态：工具、MCP、钩子、子智能体、会话存储 |
| Subagent | "Child agent" | 单独的上下文，自有预算；结果向上冒泡 |
| Session store | "Conversation DB" | 持久化、恢复、列举、删除回合，并支持对子智能体的级联删除 |
| Hook | "Lifecycle callback" | 预/后 工具、会话、提示提交、压缩、停止 等回调 |
| W3C trace context | "Cross-process trace" | 父 span 传播到 CLI 子进程，形成跨进程追踪 |
| Managed Agents | "Hosted harness" | Anthropic 托管的长期异步工作 |
| `--session-mirror` | "Transcript mirror" | 在流式写入时把会话回合写入外部文件 |
| MCP server | "Tool surface" | 附加到智能体的外部工具/资源接口 |

## 延伸阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产实践
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对应方案