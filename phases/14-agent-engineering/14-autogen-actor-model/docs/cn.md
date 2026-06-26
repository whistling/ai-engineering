# AutoGen v0.4：参与者模型与智能体框架

> AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕参与者模型重新设计了智能体编排。异步消息交换、事件驱动智能体、故障隔离、天然并发。该框架现处于维护模式，Microsoft Agent Framework（公测 2025 年 10 月）成为继任者。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** Phase 14 · 01（Agent Loop），Phase 14 · 12（Workflow Patterns）
**Time:** ~75 分钟

## 学习目标

- 描述参与者模型：智能体作为参与者、消息为唯一的进程间通信（IPC）、故障按参与者隔离。
- 说出 AutoGen v0.4 的三层 API —— Core、AgentChat、Extensions —— 以及每层的用途。
- 解释为什么将消息传递与处理解耦可以带来故障隔离与自然并发。
- 在 Python 中实现一个 stdlib 参与者运行时，并将一个两智能体的代码审查流程移植到它上面。

## 问题背景

大多数智能体框架是同步的：一个智能体产生消息，另一个智能体消费消息，处于调用栈中。失败会导致栈崩溃。并发是后来强行加入的。分布式部署需要重写大量代码。

AutoGen v0.4 的答案：参与者模型。每个智能体都是一个参与者，拥有私有的收件箱。消息是唯一的交互方式。运行时把投递与处理解耦。失败被隔离到单个参与者。并发成为原生特性。分布式只是不同的传输层。

## 概念

### 参与者

一个参与者具有：

- 私有状态（外部不可直接访问）。
- 收件箱（消息队列）。
- 处理器：`receive(message) -> effects`，其中 effects 可以是“回复（reply）”、“发送给其他参与者（send to other actor）”、“生成新参与者（spawn new actor）”、“更新状态（update state）”、“自我停止（stop self）”。

两个参与者不能共享内存。它们只能发送消息。

### AutoGen v0.4 的三层 API

1. **Core。** 低级参与者框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。支持异步消息交换、事件驱动。
2. **AgentChat。** 以任务为驱动的高级 API（替代 v0.2 的 ConversableAgent）。提供 `AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成——OpenAI、Anthropic、Azure、工具、记忆等。

### 为什么解耦很重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a，直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的收件箱并立即返回。运行时稍后负责投递。带来三个后果：

- **故障隔离。** 参与者 B 崩溃不会导致参与者 A 崩溃——运行时在 B 的处理器中捕获故障并决定后续（记录、重试、进入死信队列）。
- **自然并发。** 大量消息可以同时在路上；参与者并发地处理各自的收件箱。
- **面向分布式。** 收件箱 + 传输是同一抽象，不论参与者是在进程内还是在另一台主机上。

### 拓扑结构

- **RoundRobinGroupChat。** 智能体按固定轮转顺序轮流发言。
- **SelectorGroupChat。** 一个选择器智能体基于对话上下文决定下一步由谁处理。
- **Magentic-One。** 用于网页浏览、代码执行、文件处理的参考多智能体团队。基于 AgentChat 构建。

### 可观察性

内置 OpenTelemetry 支持。每条消息都会产生一个 span；工具调用携带符合 2026 年 OTel GenAI 语义约定的 `gen_ai.*` 属性（见 Lesson 23）。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 已稳定，适用于研究与原型开发。Microsoft 已把主动开发转移到 Microsoft Agent Framework（公测 2025 年 10 月 1 日；目标在 2026 年第一季度末达到 1.0 GA）。AutoGen 的模式可以顺利向前迁移——参与者模型是持久的核心思想。

## 构建实现

`code/main.py` 实现了一个 stdlib 参与者运行时：

- `Message` — 带类型的负载，具有 `sender`、`recipient`、`topic`、`body`。
- `Actor` — 抽象类，带有 `receive(message, runtime)`。
- `Runtime` — 事件循环，带共享队列、投递与故障隔离。
- 一个两智能体演示：`ReviewerAgent` 进行代码审查，`ChecklistAgent` 执行清单；它们交换消息直到达成一致意见。

运行方式：

```
python3 code/main.py
```

跟踪输出会显示消息投递、一个参与者的模拟失败（不会导致另一个参与者崩溃），以及最终对共享结论的收敛。

## 使用场景

- **AutoGen v0.4/v0.7**（维护中）——适用于研究、原型和多智能体模式实现。
- **Microsoft Agent Framework**（公测）——前进路径；在焕新的 API 中保留相同的参与者模型思想。
- **LangGraph swarm 拓扑**（Lesson 13）——通过共享工具交接实现的类似模式。
- **自定义参与者运行时**——当你需要特定传输（NATS、RabbitMQ、gRPC）时。

## 发布产物

`outputs/skill-actor-runtime.md` 生成一个最小参与者运行时以及针对给定多智能体任务的团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加死信队列：当处理器抛出异常时，将失败的消息存放以便人工检查。在你的玩具实现中，死信队列被命中的频率是多少？
2. 实现 `SelectorGroupChat`：一个选择器参与者基于会话状态选择下一个处理消息的人。
3. 添加分布式传输：将进程内队列替换为基于 JSON-over-HTTP 的服务器，使参与者能够在不同进程中运行。
4. 为每条消息关联一个 OTel span（或实现一个空操作替代）。发出 `gen_ai.agent.name`、`gen_ai.operation.name`，参见 Lesson 23。
5. 阅读 AutoGen v0.4 的架构文章。将你的玩具移植到真实的 `autogen_core` API。你跳过了哪些生产环境中重要的部分？

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Actor | "Agent" | 私有状态 + 收件箱 + 处理器；无共享内存 |
| Message | "Event" | 带类型的负载；参与者之间交互的唯一方式 |
| Inbox | "Mailbox" | 每个参与者的待处理消息队列 |
| Runtime | "Agent host" | 路由消息并隔离故障的事件循环 |
| Topic | "Channel" | 参与者间具名的发布-订阅通道 |
| Fault isolation | "Let it crash" | 一个参与者失败不会导致其他参与者崩溃 |
| RoundRobinGroupChat | "Fixed-rotation team" | 智能体按顺序轮流处理 |
| SelectorGroupChat | "Context-routed team" | 选择器决定下一步由谁处理 |
| Magentic-One | "Reference team" | 用于网页 + 代码 + 文件的多智能体小队 |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 重设计文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 图形化替代方案概览
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发出的 spans 的语义约定