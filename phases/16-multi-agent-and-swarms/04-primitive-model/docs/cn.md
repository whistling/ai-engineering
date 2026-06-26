# The Multi-Agent Primitive Model

> Every multi-agent framework shipping in 2026 — AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Microsoft Agent Framework — is a point in a four-dimensional design space. Four primitives, nothing more: the agent, the handoff, the shared state, the orchestrator. This lesson builds them from zero, runs a toy system on all four, then maps every major framework onto the same axes so you can read any new release in one paragraph.

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 阶段 14 (Agent Engineering), 阶段 16 · 01 (Why Multi-Agent)
**Time:** ~60 分钟

## Problem

每隔六个月就会有一个新的多智能体框架发布。AutoGen 在 2023 年。CrewAI 在 2024 年。LangGraph 和 OpenAI Swarm 在 2024 年。Google ADK 在 2025 年 4 月。Microsoft Agent Framework RC 在 2026 年 2 月。每条新闻稿都声称自己是“正确的抽象”。

如果你试图逐个学习它们，你会被折磨得精疲力尽。API 看起来各不相同。文档对“agent”是什么意思也各执一词。一个框架把它的共享内存称为“blackboard”，另一个称为“message pool”，第三个称为“StateGraph”。你开始怀疑这个领域只是在不断翻新表面。

事实并非如此。在市场宣传之下，四个原语是稳定的。学会它们一次，就能用一句话读懂每一个新框架。

## Concept

### The four primitives

1. **Agent** — 一个 system prompt 加上一组 tools。无状态；每次运行都从它的 system prompt 和当前消息历史开始。
2. **Handoff** — 从一个 agent 到另一个 agent 的结构化控制转移。机械上是一个返回新 agent 的工具调用，或是一条遵循条件的图边。
3. **Shared state** — 任意一个被多个 agent 读取（有时写入）的数据结构。消息池、黑板、键值存储、向量记忆。
4. **Orchestrator** — 决定谁下一个发言的实体。可选项：显式图（确定性）、LLM 说话者选择器（软路由）、上一个说话者的 handoff 调用（OpenAI Swarm），或基于队列的调度器（swarm 架构）。

这就是整个设计空间。每个框架在每个轴上选择默认值；剩下的只是表面语法。

### How every 2026 framework maps to it

| Framework | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | tool returns Agent | 调用者的问题 | LLM 的下一个 handoff 调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | speaker-selector on GroupChat | message pool | selector 函数（LLM 或 轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 任务输出链式保存 | manager LLM 或 静态顺序 |
| LangGraph | node function | graph edge + condition | `StateGraph` reducer | 图（确定性） |
| Microsoft Agent Framework | agent + orchestration patterns | pattern-specific | 线程 / 上下文 | pattern-specific |
| Google ADK | agent + A2A card | A2A task | A2A artifacts | 主机决定 |

表面差异看起来很大。底层：相同的四个旋钮。

### Why this matters

一旦你看清了这些原语，框架比较就变成一个简短的核对清单：

- 编排器是否信任 LLM 来路由（Swarm），还是把路由固定在代码里（LangGraph）？
- 共享状态是完整历史（GroupChat）还是投影式（StateGraph reducer）？
- agents 能否修改彼此的提示词（CrewAI 的 manager）还是只能通过 handoff 交接（Swarm）？

这三个问题回答了 80% 的“哪个框架适合某个问题”的判定。你就不再去盲目寻找“最好的多智能体框架”，而是开始针对你真正关心的轴来设计。

### The stateless insight

除了共享状态外每个原语都是无状态的。Agent 是 (prompt, tools) 的函数。Handoff 是一个函数调用。Orchestrator 是一个调度器。**系统中唯一有状态的东西是共享状态。** 这里是所有有趣 bug 出现的地方：记忆污染（Lesson 15）、消息排序、版本管理、写竞争。

隐藏共享状态的框架（Swarm）将问题推给调用者。把它集中化的框架（LangGraph 检查点、AutoGen 池）使其可检查，但把协调成本转移到共享状态实现上。

### Anatomy of a single primitive

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

无内存。无状态。两个具有相同 system prompt 和 tools 的 agent 是可互换的。所有看起来像是每个 agent 的状态实际上都在共享状态或 handoff 协议中。

#### Handoff

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占主导：

- **函数返回** — 工具返回下一个 agent。这是 OpenAI Swarm 的模式。agents 在它们的工具 schema 中携带路由信息。
- **图边** — LangGraph。边是声明式的。LLM 产生一个值；条件选择下一个节点。
- **说话者选择** — AutoGen GroupChat。一个选择器函数（有时本身是一次 LLM 调用）读取池并选择谁下一个发言。

#### Shared state

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

最少是一个消息列表。通常还有更多：结构化工件（CrewAI 任务输出）、类型化上下文（LangGraph reducer）、外部记忆（MCP、向量数据库）。

两种拓扑：**完整池（full pool）**（每个 agent 看到每条消息）和**投影式（projected）**（agent 看到基于角色的视图）。完整池简单但可扩展性差。投影式可扩展但需要预先的 schema 设计。

#### Orchestrator

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种风格：

- **静态** — 图在构建时固定（LangGraph 确定性、CrewAI 顺序式）。
- **LLM 选择型** — LLM 读取池并选择下一个说话者（AutoGen、CrewAI 层级式）。
- **Handoff 驱动** — 当前 agent 通过调用 handoff 工具来决定（Swarm）。
- **队列驱动** — 工作者从共享队列拉取工作；没有显式的下一个说话者（swarm 架构，Matrix）。

### What changes between frameworks

一旦原语被固定，剩下的设计决策是：

- **内存策略** — 临时 vs 持久检查点（LangGraph checkpointer）。
- **安全边界** — 谁可以批准 handoff（人类在环）。
- **成本核算** — 每个 agent 的 token 预算。
- **可观测性** — 跟踪 handoff、持久化状态以便重放。

这些都可以在原语之上实现。它们都不是新的原语。

## Build It

`code/main.py` 在大约 150 行标准库 Python 中实现了四个原语。没有真实的 LLM —— 每个 agent 都是脚本化的策略，这样焦点就能保持在协调结构上。

该文件导出：

- `Agent` — 一个包含 name、system prompt、tools、policy 函数的数据类。
- `Handoff` — 一个返回新 agent 的函数。
- `SharedState` — 一个线程安全的消息池。
- `Orchestrator` — 三个变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示运行相同的三 agent 流水线（research → write → review），通过三种编排器类型运行并在结束时打印消息池。你可以看到输出仅在“谁选择下一个”上有所不同；agents 和共享状态在各次运行中是相同的。

运行它：

```
python3 code/main.py
```

预期输出：三个编排器运行，每种模式一次。每个运行打印最终的消息池。如果研究者决定提前结束，handoff 驱动的运行会到达更少的 agents —— 这就是 LLM 路由的小型示例权衡。

## Use It

`outputs/skill-primitive-mapper.md` 是一个 skill，它读取任意多智能体代码库或框架文档并返回四原语映射。在新框架发布时运行它，可以在深入阅读文档前获得一句话的理解。

## Ship It

在采纳新框架之前，为其写下原语映射。如果你做不到，说明文档不完整或者框架在发明第五个原语（罕见——检查是否有一种你没见过的共享状态变体）。

把映射固定在你的架构文档中。当新成员加入时，在发给他们 API 文档之前先发映射。当框架版本变化时，比对映射，而不是变更日志。

## Exercises

1. 运行 `code/main.py` 三次，使用不同的 agent 策略。观察编排器的选择如何改变运行的 agent 顺序。
2. 实现第四种编排器类型：基于队列的，agent 在其中轮询共享状态以获取工作。会发生什么死锁，你如何检测它？
3. 取 LangGraph 快速入门 (https://docs.langchain.com/oss/python/langgraph/workflows-agents) 并把它重写成四原语。LangGraph 的哪些抽象是一一映射，哪些是便捷封装？
4. 阅读 OpenAI Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents)。找出 Swarm 在四个原语中哪个最易用，哪个被推给调用者。
5. 在上表中找出一个完全隐藏共享状态的框架。解释当 agents 需要在 handoff 之间跨越并协调而不重新读取历史时，会发生什么问题。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "An LLM with tools" | 一个 `(system_prompt, tools, model)` 三元组。无状态。 |
| Handoff | "Transfer of control" | 一个结构化调用，命名下一个 agent 并可携带可选负载。三种实现：函数返回、图边、说话者选择。 |
| Shared state | "Memory" / "context" | 多智能体系统中唯一有状态的部分。消息池或黑板。 |
| Orchestrator | "Coordinator" | 决定谁下一个运行的实体。静态图、LLM 选择、handoff 驱动或队列驱动。 |
| Primitive | "Abstraction" | 四个轴之一，每个框架都会参数化。不是一个框架特性。 |
| Message pool | "Shared chat history" | 完整历史的共享状态。易于理解，但可扩展性差。 |
| Projected state | "Scoped view" | 针对角色的共享状态视图。可扩展，但需要 schema 设计。 |
| Speaker selection | "Who talks next" | 一种编排器模式，其中一个函数（通常是 LLM）从一组中挑选下一个 agent。 |

## Further Reading

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 对 handoff 驱动式编排的最清晰阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat + 说话者选择是 LLM 选择型编排的参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 图边编排和基于 reducer 的共享状态
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — role-goal-backstory 的 agents，Sequential / Hierarchical 过程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — Microsoft 将 v0.4 转入维护后，live 的 AutoGen v0.2 分支