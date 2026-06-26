# LangGraph — State Machines for Agents

> 手写的 ReAct 循环是一个 `while True`。用 LangGraph 写的 ReAct 循环是一个你可以检查点、打断、分支并进行时间旅行的图。Agent 本身没有变。包裹它的外壳变了。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (函数调用), Phase 11 · 14 (模型上下文协议)
**Time:** ~75 分钟

## 问题

你交付了一个支持函数调用的 agent。它能工作三回合，然后出问题：模型调用了一个返回 500 的工具，用户在任务中途反悔，或者 agent 决定退款而没有人工签署。`while True:` 循环没有挂载点。你不能暂停它，不能回溯，也不能分支到“如果模型选了另一个工具会怎样”。一旦你把这种东西交付给演示后，agent 就变成了一个黑盒，要么能工作要么不能。

下一步一目了然。agent 本身已经是一个状态机 —— 系统提示 + 消息历史 + 待处理的工具调用 + 下一个动作。把状态机显式化：为“模型思考”、“工具运行”、“人工批准”建立节点，为它们之间的条件转移建立边。一旦图是显式的，外壳就能免费获得四样东西：检查点（在步骤之间保存状态）、中断（为人工暂停）、流式（流式传输 token 和中间事件）、以及时间旅行（回到先前状态，尝试不同分支）。

LangGraph 是提供该抽象的库。它不是 LangChain 意义上的 agent 框架（“这里有个 AgentExecutor，祝你好运”）。它是一个具有一流状态、一流持久化和一流中断能力的图运行时。agent 循环是你画出来的，而不是你手写的。

## 概念

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 有三项要素。

1. 状态（State）。一个经过类型标注的字典（TypedDict 或 Pydantic model），在图中流动。每个节点接收完整状态并返回部分更新，LangGraph 使用每个字段的 *reducer* 合并更新 —— 对于应累积的列表使用 `operator.add`，默认是覆盖。
2. 节点（Nodes）。Python 函数 `state -> partial_state`。每个节点是一个离散步骤：比如“调用模型”、“运行工具”、“摘要”。
3. 边（Edges）。节点之间的转移。静态边指向固定节点。条件边接受一个路由函数 `state -> next_node_name`，以便图能基于模型输出进行分支。

编译（compile）图。编译会绑定拓扑结构，附加一个检查点器（可选但在生产中必需），并返回一个可运行对象。你用初始状态和一个 `thread_id` 调用它。每一步执行都会持久化一个以 `(thread_id, checkpoint_id)` 为键的检查点。

### 四项超能力

**Checkpointing（检查点）。** 每次节点转移都会将新状态写入存储（测试时内存存储，生产时 Postgres/Redis/SQLite）。通过使用相同的 `thread_id` 再次调用图来恢复。图会从暂停处继续。

**Interrupts（中断）。** 给节点标记 `interrupt_before=["human_review"]`，执行将在该节点运行前停止。状态会持久化。你的 API 向用户响应“等待审批”。稍后针对相同 `thread_id` 的 `Command(resume=...)` 请求会恢复执行。

**Streaming（流式）。** `graph.stream(state, mode="updates")` 在发生时产出状态增量。`mode="messages"` 在模型节点内部流式产出 LLM token。`mode="values"` 产出完整快照。你可以选择在 UI 中呈现什么。

**Time-travel（时间旅行）。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。将任一先前的 `checkpoint_id` 传给 `graph.invoke`，即可从该点分叉。非常适合调试（“如果模型选择了工具 B 会怎样？”）以及用于回放生产痕迹的回归测试。

### Reducer 是重点

每个状态字段都有一个 reducer。大多数默认是可以的 —— 新值覆盖旧值。但消息列表需要 `operator.add`，这样新消息会追加而不是替换。并行边通过 reducer 合并它们的更新。如果两个节点都更新 `messages`，而你忘了用 `Annotated[list, add_messages]`，第二个会默默获胜，你会丢失半回合对话。reducer 是库中唯一微妙的地方；把它设置正确，其余部分会自然而然地组合。

### 四个节点表示的 ReAct 图

一个生产级的 ReAct agent 是四个节点和两条边：

1. `agent` — 用当前消息历史调用 LLM。返回 assistant 消息（可能包含 tool_calls）。
2. `tools` — 执行最后一条 assistant 消息中的任何 tool_calls，并把工具结果附加为工具消息。
3. 从 `agent` 出发的条件边：如果最后一条消息有 tool_calls 则路由到 `tools`，否则到 `END`。
4. 从 `tools` 回到 `agent` 的静态边。

就是这么简单。你能得到完整的 ReAct 循环（Thought → Action → Observation → Thought → …），并带有检查点、中断和流式能力，代码量大约 40 行。

### StateGraph 与 Send（扇出）

`Send(node_name, state)` 允许一个节点分发并行子图。示例：agent 决定同时查询三个检索器。每个 `Send` 会为目标节点生成一个并行执行；它们的输出通过状态 reducer 合并。这就是 LangGraph 在不使用线程原语的情况下表达 orchestrator-workers 模式的方式。

### 子图（Subgraphs）

已编译的图可以作为另一个图中的节点。外层图看到的是一个单一节点；内层图有其自己的状态和检查点。这就是团队构建监督-工作者 agent 的方式：监督图将用户意图路由到每个域的工作子图。

## 构建它

### 步骤 1：状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是使消息列表累积而不是覆盖的 reducer。忘记它是使用 LangGraph 最常见的错误。

### 步骤 2：用 thread 运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每个更新都是一个字典 `{node_name: state_delta}`。你的前端可以将这些流式传输到 UI，让用户看到“agent 正在思考……调用 search_web……获得结果……在回答”。

### 步骤 3：加入人工流程中的中断

在节点上标记，使执行在运行该节点之前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 在每次工具调用之前暂停
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] 已被设置。检查拟议的工具调用。
# 如果批准：
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 如果拒绝：写入一条拒绝消息并恢复
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点和线程都会在中断之间保持。执行期间之外没有任何东西留在内存中。

### 步骤 4：用于调试的时间旅行

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 从一个先前的检查点分叉
target = history[3].config  # 回到三步之前
for event in app.stream(None, target, stream_mode="values"):
    pass  # 从该点开始回放
```

向 `invoke` 传入 `None` 会从给定检查点回放；传入一个值会将其作为对该检查点状态的更新后再继续。这就是在不重新运行整个对话的情况下复现一次糟糕 agent 运行的方法。

### 步骤 5：为生产替换检查点器

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都已提供。`MemorySaver` 适用于测试。任何需要跨重启持久化的场景都需要一个真实的存储。

## 技巧

> 你把 agent 构建成图，而不是 `while True` 循环。

在拿起 LangGraph 之前，做一个 60 秒的设计：

1. 给节点命名。每个离散的决策或有副作用的动作都是一个节点。“Agent 思考”、“工具运行”、“审阅者批准”、“响应流式输出”。如果你不能把它们列出来，任务还不是 agent 形状的。
2. 声明状态。使用最小的 TypedDict，并为每个列表字段声明 reducer。不要把所有东西塞进 `messages`；把任务特定字段提升到顶层（一个工作中的 `plan`、一个 `budget` 计数器、一个 `retrieved_docs` 列表）。
3. 画出边。默认是静态，除非下一步依赖于模型输出。每条条件边都需要一个带命名分支的路由函数。
4. 事先选择检查点器。测试用 `MemorySaver`，其余使用 Postgres/Redis/SQLite。不带检查点器不要上线 —— 没有检查点器就没有恢复、没有中断、没有时间旅行。
5. 决定在工具运行之前进行中断，而不是之后。把批准放在进入有副作用节点的边上，这样你可以在造成伤害之前取消；把验证放在模型输出之后的边上，这样你可以廉价地拒绝糟糕调用。
6. 默认使用流式。`mode="updates"` 适合 UI，`mode="messages"` 适合在模型节点内进行 token 级流式，`mode="values"` 适合评估期间的完整快照。

拒绝交付没有检查点器的 LangGraph agent。拒绝交付在副作用之后才中断的 agent。拒绝交付一个没有 `add_messages` reducer 的 `messages` 字段。

## 练习

1. 简单。实现上面四节点的 ReAct 图，带一个计算器工具和一个网页搜索工具。验证对于两回合对话，`list(app.get_state_history(config))` 至少返回四个检查点。
2. 中等。添加一个在 `agent` 之前运行的 `planner` 节点并在状态中写入结构化的 `plan: list[str]`。让 `agent` 标记计划步骤为已完成。如果在检查点恢复后 `plan` 丢失（reducer 错误），测试失败。
3. 困难。构建一个监督图，使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图有自己的状态和检查点器。在外层图上添加 `interrupt_before=["writer"]` 以便人工审批研究摘要。确认从先前检查点进行时间旅行只会重新运行被分叉的分支。

## 术语表

| Term | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| StateGraph | "The LangGraph graph" | 你在编译前添加节点和边的构建器对象。 |
| Reducer | "How the field merges" | 一个函数 `(old, new) -> merged`，当节点返回该字段的更新时应用；默认是覆盖，`add_messages` 会追加。 |
| Thread | "A conversation ID" | 一个 `thread_id` 字符串，用于限定某个会话的所有检查点。 |
| Checkpoint | "A paused state" | 节点转移后持久化的完整图状态快照，键为 `(thread_id, checkpoint_id)`。 |
| Interrupt | "Pause for a human" | `interrupt_before` / `interrupt_after` 在节点边界停止执行；使用 `Command(resume=...)` 恢复。 |
| Time-travel | "Fork from a prior step" | `graph.invoke(None, config_with_old_checkpoint_id)` 会从该检查点开始回放。 |
| Send | "Parallel subgraph dispatch" | 节点可以返回的一个构造体，用于生成目标节点的 N 个并行执行。 |
| Subgraph | "A compiled graph as a node" | 已编译的 StateGraph，被用作另一个图中的节点；保留自己的状态作用域。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — 关于 StateGraph、reducers、checkpointers 和 interrupts 的权威参考。
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本课程采用的心智模型，来自源码。
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — 关于 Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的详细说明。
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)` 和 编辑状态模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个 LangGraph agent 都在实现的模式；阅读它以理解推理轨迹的理由。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 哪些图形结构（chain、router、orchestrator-workers、evaluator-optimizer）在何时更合适。
- Phase 11 · 09 (函数调用) — 每个 LangGraph agent 节点重用的工具调用原语。
- Phase 11 · 14 (模型上下文协议) — 可插入到 LangGraph `ToolNode` 的外部工具发现机制（通过 MCP 适配器）。
- Phase 11 · 17 (Agent 框架权衡) — 何时选择 LangGraph 而不是 CrewAI、AutoGen 或 Agno。