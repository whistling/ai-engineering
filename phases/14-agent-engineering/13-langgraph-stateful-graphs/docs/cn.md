# LangGraph: 有状态图与持久化执行

> LangGraph 是 2026 年面向低层有状态编排的参考实现。智能体（Agent）是一个状态机；节点是函数；边是转移；状态是不可变的并在每步后进行检查点保存。发生失败后可以从精确停止的地方恢复。

**Type:** 学习 + 构建  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 14 · 01 (Agent Loop) , Phase 14 · 12 (Workflow Patterns)  
**Time:** ~75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：带不可变状态的状态机、函数节点、条件边以及每步后的检查点保存。  
- 列出文档强调的四项能力：持久化执行、流式输出、人工参与、全面的记忆。  
- 说明 LangGraph 支持的三种编排拓扑：监督者、点对点（群体/Swarm）、分层（嵌套子图）。  
- 实现一个 stdlib 状态图，包含不可变状态、条件边以及检查点/恢复循环。

## 问题背景

智能体和工作流面临同一个问题：当一个 40 步的运行在第 38 步失败时，你希望从第 38 步继续，而不是从头开始。二等状态模型会让运维人员在一个假设每次都是全新运行的库上手动补丁重试。

LangGraph 的设计答案：将状态作为一等的类型化对象，变更需要显式表示，并在每个节点后做检查点。恢复就是一个 `load_state(session_id)` 调用。

## 概念

### 图（graph）

一个图由以下部分定义：

- **State type。** 一个类型化的 dict（或 Pydantic 模型），每个节点都读取并可变更该状态。  
- **Nodes。** 纯函数 `(state) -> state_update`。返回的更新会在返回后合并到状态中。  
- **Edges。** 节点间的条件或直接转移。  
- **Entry and exit。** `START` 和 `END` 哨兵节点标记边界。

示例：一个包含 `classify`、`refund`、`bug`、`sales`、`done` 节点的智能体 —— 作为路由工作流的图。

### 持久化执行（Durable execution）

在每个节点返回后，运行时会序列化状态并将其写入检查点器（SQLite、Postgres、Redis 或自定义后端）。在步骤 N 发生失败时，运行时可以 `resume(session_id)` 并从第 N+1 步开始，状态完全一致。

LangGraph 文档明确指出在生产环境中此项很重要的用户：Klarna、Uber、J.P. Morgan。关键不是图的形状本身；而是图的形状加上检查点使得恢复代价很低。

### 流式（Streaming）

每个节点都可以产生部分输出。图会将每个节点的增量事件流式传输给调用者，从而使 UI 在图运行时可以实时更新。

### 人工参与（Human-in-the-loop）

在节点之间检查并修改状态。实现方式：在关键节点前暂停，将状态呈现给人工，接受修改后继续。因为状态已经被检查点序列化，这一过程非常容易实现。

### 记忆（Memory）

短期记忆（在一次运行内 —— 状态中的对话历史）和长期记忆（跨运行持久化 —— 由检查点器加上独立的长期存储实现）。LangGraph 通过工具与外部记忆系统（如 Mem0 或自定义存储）集成。

### 三种拓扑

1. **Supervisor（监督者）。** 中央路由 LLM 分派到专门子智能体。通过 `create_supervisor()` 位于 `langgraph-supervisor`（不过到 2026 年 LangChain 团队建议通过直接的工具调用来实现，以便获得更多上下文控制）。  
2. **Swarm / peer-to-peer（群体 / 点对点）。** 智能体通过共享的工具表面直接交接。无中央路由。  
3. **Hierarchical（分层）。** 监督者管理子监督者，作为嵌套子图实现。

### 该模式的局限与误区

- **检查点不够全面。** 仅检查点对话回合会导致工具状态和记忆写入不可恢复。必须序列化完整状态。  
- **非确定性节点。** 恢复假设节点输入会产生相同的状态更新。随机种子、墙钟时间、外部 API 必须被捕获或记录。  
- **条件边的过度使用。** 每条边都带条件的图会变成难以推理的状态机。优先采用线性链并在必要时做分支。

## 实战构建

`code/main.py` 实现了一个 stdlib 有状态图：

- `State` — 一个类型化的 dict，包含 `messages`、`step`、`route`、`output`、`human_approval`。  
- `Node` — 可调用对象，接收 state 并返回更新字典。  
- `StateGraph` — 节点 + 边 + 条件边 + run + resume。  
- `SQLiteCheckpointer`（内存示例） — 在每个节点后序列化状态；`load(session_id)` 可恢复。  
- 一个演示图：classify -> branch(refund / bug / sales) -> human gate -> send。

运行：

```
python3 code/main.py
```

运行跟踪会显示第一次在人工门点失败、持久化，然后恢复并生成最终输出。

## 使用场景

- **LangGraph** — 参考实现，适合生产就绪。使用 `create_react_agent`、`create_supervisor`，或自己构建图。  
- **AutoGen v0.4**（Lesson 14）— 在高并发场景下的参与者模型（actor model）替代方案。  
- **Claude Agent SDK**（Lesson 17）— 带内置会话存储的托管框架与子智能体支持。  
- **自定义** — 当你需要对状态形状或检查点后端进行精确控制时。

## 部署产出

`outputs/skill-state-graph.md` 会生成一个适配任何目标运行时的 LangGraph 形状状态图，并接入检查点与恢复。

## 练习

1. 在 `classify` 到 `end` 之间添加一个条件边，当分类置信度低于阈值时直接转到 `end`。在人为手动设置 `route` 后恢复运行。  
2. 将 SQLite 样例替换为真实的 SQLite 检查点器。测量每步序列化的开销。  
3. 实现并行边：同时运行两个节点，通过自定义 reducer 合并结果。不可变状态在这里带来了哪些好处？  
4. 阅读 `langgraph-supervisor` 参考文档。将示例移植到 `create_supervisor`。比较运行轨迹的形态。  
5. 添加流式：每个节点在运行时产出部分状态并 yield。按到达顺序打印增量。

## 术语表

| Term | 大众说法 | 实际含义 |
|------|----------------|------------------------|
| State graph | "Agent as state machine" | 类型化状态 + 节点 + 边 + reducer |
| Checkpointer | "Persistence backend" | 在每个节点后序列化状态；支持恢复 |
| Reducer | "State merger" | 将当前状态与节点返回的更新合并的函数 |
| Conditional edge | "Branch" | 由状态决定的边选择函数 |
| Subgraph | "Nested graph" | 在另一个图中作为节点使用的图 |
| Durable execution | "Resume from failure" | 从最后一个成功的节点并带有精确状态重启 |
| Supervisor | "Router LLM" | 用于分派到专门子智能体的中央调度者 |
| Swarm | "P2P agents" | 智能体通过共享工具进行交接；无中央路由 |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 参考文档  
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — 监督者模式 API  
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 参与者模型替代方案  
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — 会话存储与子智能体支持