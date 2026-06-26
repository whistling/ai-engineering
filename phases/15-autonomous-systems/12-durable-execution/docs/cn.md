# Long-Running Background Agents: Durable Execution

> 生产环境中的长时程代理不会运行在 `while True` 中。每一次 LLM 调用都被视为一个带有检查点、重试和重放的活动。Temporal 在 2026 年 3 月将 OpenAI Agents SDK 的集成推向 GA。Claude Code Routines（Anthropic）能够按计划运行 Claude Code 调用，而无需持久化的本地进程。会话在等待人工输入时会暂停，能在部署间存活，并从以 `thread_id` 为键的最新检查点恢复。新的可用性体验背后是一个古老模式——工作流编排——加上一个新输入：将 LLM 调用视为必须在恢复时确定性重放的非确定性活动。

**Type:** 学习  
**Languages:** Python (stdlib，最小持久化执行状态机)  
**Prerequisites:** Phase 15 · 10（权限模式），Phase 15 · 01（长时程代理）  
**Time:** ~60 分钟

## 问题

考虑一个运行四小时的代理。它调用了三个工具、两次提示用户，并进行了四十次 LLM 调用。运行到一半时，它所在的主机重启了。会发生什么？

- 在天真的 `while True` 循环里：一切都会丢失。运行从头开始。三个具有真实副作用的工具调用会再次执行。已经批准的内容会再次提示用户。四十次 LLM 调用会被重复计费。
- 使用持久化执行：运行从最近的检查点恢复。已经完成的活动不会重复执行；其结果从持久化日志中重放。用户不会再次为已批准的事项重复批准。已完成的 LLM 调用不会被重复计费。

这与工作流引擎过去十年的做法相同（Temporal、Cadence、Uber 的 Cherami）。新的地方在于现在 LLM 调用成为了一类活动——非确定性、昂贵、可能有副作用——并且它们干净地符合这个模式。

本课的主旨：长时程可靠性会衰减（METR 观察到“约 35 分钟的退化”——成功率随时间大致呈二次下降）。持久化执行使得运行可以超出可靠性曲线所支持的时长，这在设计得当时是一种安全的失败方式，在设计不当时则会很危险。

## 概念

### 活动、工作流与重放

- **工作流（Workflow）**：确定性的编排代码。定义活动的顺序、分支与等待。必须是确定性的，以便能从事件日志重放而不产生意外偏差。
- **活动（Activity）**：非确定性、可能失败的工作单元。LLM 调用、工具调用、文件写入、HTTP 请求。每个活动都会记录其输入，并在完成后记录输出。
- **事件日志（Event log）**：持久化的后端存储。记录每个活动的开始、完成、失败、重试，以及每个工作流决策。
- **重放（Replay）**：在恢复时，工作流代码从头重新运行；所有已完成的活动返回其记录的结果而不重新执行。只有尚未完成的活动会实际运行。

这与 React 针对虚拟 DOM 的重新渲染，或 Git 从提交重建工作树的形态相同。编排器的确定性使得持久化成本低廉。

### 为什么 LLM 调用符合此模式

LLM 调用具有以下属性：
- 非确定性（temperature > 0；即使 temperature=0 在模型版本更新时也会漂移）。
- 昂贵（金钱与延迟）。
- 可能失败（速率限制、超时）。
- 可能产生副作用（如果它们调用工具）。

这正好符合“活动”的特征。将每次 LLM 调用封装为活动，可以获得带指数退避的重试、跨重启的检查点，以及用于调试的可重放轨迹。

### 以 `thread_id` 为键的检查点

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 都趋向于相同的 API 形态：一个 `thread_id`（或等价物）标识会话；每次状态转移持久化到后端（默认 PostgreSQL，开发时可用 SQLite，Redis 用作缓存）；恢复时读取最新检查点。

后端的选择很重要：

- **PostgreSQL**：持久、可查询，能在部署间存活。LangGraph 的默认选择。
- **SQLite**：仅用于本地开发；跨主机会丢失数据。
- **Redis**：速度快但非持久，除非配置了 AOF/快照。
- **Cloudflare Durable Objects**：透明分布式；由唯一键范围化；可存活数小时到数周。

### 将人类输入作为一等状态

Propose-then-commit（第 15 课）需要一个持久化的“等待人工输入”状态。工作流暂停，外部队列保存待处理请求，一旦批准，从完全相同的点恢复。没有持久化时这是尽力而为；有了持久化，过夜的批准可以到达，工作流在第二天早上继续执行。

### 35 分钟退化

METR 观察到每个被测的代理类别在约 35 分钟之后均出现可靠性下降。将任务时长翻倍，故障率大约成平方级增长。持久化执行并不能修复这个问题；它允许你运行超出可靠性轮廓支持的时长。安全的模式是将持久化与需要在重新进入时进行新鲜 HITL 的检查点策略结合，并配合预算终止开关（第 13 课），无论墙钟时间如何都限制总计算量。

### 何时持久化执行不是合适的答案

- 运行时间短于几分钟且没有人工输入时。开销大于收益。
- 严格的只读信息检索场景。
- 正确性需要在单个上下文窗口内端到端完成的任务（某些推理任务；某些单次生成）。

```figure
memory-consolidation
```

## 使用方法

`code/main.py` 实现了一个基于 stdlib 的最小持久化执行引擎。它支持：

- `@activity` 装饰器，将输入和输出记录到 JSON 事件日志。
- 一个将活动串联起来的工作流函数。
- 一个 `run_or_replay(workflow, event_log)` 函数，在不重新执行已完成活动的情况下重放它们。

驱动示例模拟了一个由三项活动组成的工作流，在中途崩溃，并展示了 (a) 天真重试会重新执行所有东西 与 (b) 重放只运行缺失活动 的差异。

## 上线

`outputs/skill-durable-execution-review.md` 审查了一项拟议的长时程代理部署是否满足正确的持久化执行形式：活动、确定性、检查点后端、人类输入状态和恢复时的 HITL 策略。

## 练习

1. 运行 `code/main.py`。观察天真重试与重放在活动执行计数上的差异。更改崩溃点，展示重放计数如何相应变化。

2. 将玩具引擎修改为显式使用 `thread_id`。模拟两个并发会话共享该引擎，并确认它们的事件日志不会冲突。

3. 选取玩具引擎中的一个活动。在工作流决策中引入一个非确定性项（例如 wall-clock 时间戳）。演示重放时的分歧。解释真实引擎如何处理此类情况（副作用注册、`Workflow.now()` API 等）。

4. 阅读 LangChain 的《Runtime behind production deep agents》一文。列出运行时持久化的每个状态，并说明每个状态覆盖了哪种故障模式。

5. 为一次 6 小时的自主编码任务设计检查点策略。你在哪些点设置检查点？崩溃后如何恢复？哪些情形需要重新进行 HITL？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---:|---|
| Workflow | “代理的脚本” | 确定性的编排代码；可从事件日志重放 |
| Activity | “一个步骤” | 非确定性的工作单元（LLM 调用、工具调用）；在前后记录 |
| Event log | “后端存储” | 每次状态转移的持久化记录 |
| Replay | “恢复” | 重新运行工作流；已完成的活动返回记录结果，不会重新执行 |
| Checkpoint | “存档点” | 由 `thread_id` 键控的持久化状态；恢复时以最新为准 |
| thread_id | “会话键” | 作用域持久化状态的标识符 |
| 35-minute degradation | “可靠性衰减” | METR：成功率随任务时长大约呈二次下降 |
| Non-determinism | “重放漂移” | 时钟、随机、LLM 输出；必须注册为副作用 |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 有关预算、回合与恢复语义。  
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent 的形状与实践。  
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体的运行时需求。  
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — 将 LLM 调用作为活动的形态说明。  
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 关于 35 分钟退化的参考资料。