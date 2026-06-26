# 智能体框架权衡——LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都推销相同的演示（研究智能体生成报告），并隐藏相同的缺陷（状态模式与编排层冲突）。选择其抽象与你的问题形态相匹配的框架；否则，你将重复编写相同的胶水代码。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**Time:** ~45 分钟

## 问题

你有一个任务需要不止一次 LLM 调用。也许它是一个研究工作流（规划、搜索、总结、引用）。也许它是一个代码审查流水线（解析差异、评论、打补丁、验证）。也许它是一个预订航班、撰写电子邮件和提交费用报告的多轮助手。你选择了一个框架。

三天后，你发现该框架的抽象泄漏了。CrewAI 提供了角色，但当“研究员”需要将结构化计划交给“撰稿人”时，它会让你感到困扰。AutoGen 提供了智能体之间的聊天功能，但没有一流的状态管理，因此你的检查点只是对话日志的 pickle 序列化。LangGraph 提供了状态图，但在你了解智能体将做什么之前，它会强制你命名每个转换。Agno 提供了一个单智能体抽象，当你尝试扇出到三个并发工作器时，它会发出警告。

解决方案不是“选择最好的框架”。而是将框架的核心抽象与你的问题形态相匹配。本课程将绘制这张地图。

## 概念

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

四个框架主导着 2026 年的格局。它们的核心抽象并不相同。

| 框架 | 核心抽象 | 最佳适用场景 | 最差适用场景 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — 类型化状态、节点、条件边、检查点器。 | 具有显式状态和人机协作中断的工作流；需要时间旅行调试的生产智能体。 | 松散的、角色驱动的头脑风暴，拓扑结构未知。 |
| **CrewAI** | `Crew` — 角色（目标、背景故事）、任务、流程（顺序或层级）。 | 角色扮演或基于角色的工作流，具有简短的线性/层级计划。 | 超出团队轮次历史的任何有状态内容；复杂分支。 |
| **AutoGen** | `ConversableAgent` 对 — 两个或更多智能体轮流发言，直到满足退出条件。 | 多智能体*对话*（师生、提议者-评论者、行动者-审阅者），思考从聊天中浮现。 | 具有已知 DAG 的确定性工作流；任何需要跨重启持久状态的场景。 |
| **Agno** | `Agent` — 单个 LLM + 工具 + 内存，可组合成团队。 | 快速构建的单个智能体和轻量级团队；强大的多模态能力和内置存储驱动。 | 深度、显式分支的图，带有自定义归约器。 |

### “抽象”的实际含义

框架的核心抽象是你推销架构时在白板上绘制的东西。

- **LangGraph** → 你绘制一个图。节点是步骤，边是转换，每个点的状态对象都是类型化的。心智模型是状态机。
- **CrewAI** → 你绘制一个组织结构图。每个角色都有职位描述，经理负责路由任务。心智模型是一个小型专家团队。
- **AutoGen** → 你绘制一个 Slack 私信。两个智能体互相发送消息；如果需要主持人，第三个智能体加入。心智模型是聊天。
- **Agno** → 你绘制一个带有工具的单个框。将多个框并排放置以组成团队。心智模型是“自带电池的智能体”。

### 状态问题

状态是大多数框架选择在生产中崩溃的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型）、按字段归约器、一流检查点器（SQLite/Postgres/Redis）。恢复、中断和时间旅行是免费的。*(参见 Phase 11 · 16。)*
- **CrewAI。** 状态通过 `context` 字段以字符串形式在任务间流动，或通过 `output_pydantic` 进行结构化。开箱即用不提供持久的团队存储；如果团队必须在重启后存活，你需要自行附加。
- **AutoGen。** 状态是聊天历史和任何用户定义的 `context`。对话记录持久化；任意工作流状态不会持久化，除非你编写适配器。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）通过 `storage=` 附加到 `Agent` — 对话会话和用户记忆自动持久化。它不是一个完整的图检查点器；而是一个会话存储。

### 分支问题

每个非平凡的智能体都会分支。谁决定分支很重要。

- **LangGraph** — 你通过条件边决定。路由是一个带有命名分支的 Python 函数。分支在编译后的图中是一流的；检查点器记录了采取了哪个分支。
- **CrewAI** — 经理在层级模式下决定；在顺序模式下，你在构建时决定。路由隐含在任务列表中；在经理的提示之外没有一流的“if”语句。
- **AutoGen** — 智能体通过聊天决定。分支从谁接下来发言中浮现。`GroupChatManager` 选择下一个发言者；你可以手写一个 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno** — 智能体通过调用哪个工具来决定。团队具有协调器/路由器/协作器模式；超出此范围的分支是开发者的责任。

### 可观测性问题

- **LangGraph** — 通过 LangSmith 或任何 OTel 导出器实现 OpenTelemetry。每个节点转换都是一个跟踪跨度；检查点兼作可重放的跟踪。LangSmith 是第一方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI** — 自 2025 年末起提供一流的 OpenTelemetry；与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen** — 通过 `autogen-core` 集成 OpenTelemetry；AgentOps 和 Opik 有连接器。跟踪粒度是按智能体消息，而非按节点。
- **Agno** — 内置 `monitoring=True` 标志加上 OpenTelemetry 导出器；与 Langfuse 紧密集成以进行会话跟踪。

### 成本和延迟

所有四个框架都会增加每次调用的开销（框架逻辑、验证、序列化）。开销大致按递增顺序排列：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要取决于框架执行了多少额外的 LLM 路由。CrewAI 的层级管理器花费 token 决定谁接下来发言；AutoGen 的 `GroupChatManager` 也是如此。LangGraph 仅在你编写 `llm.invoke` 的地方花费 token。Agno 的单智能体路径很薄。

当每次运行的成本很重要时，优先选择显式路由（LangGraph 边、AutoGen `speaker_selection_method`），而不是 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。一流的 MCP 适配器（工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具都可以适配。通过 `allow_delegation=True` 实现团队到团队的委托。
- **AutoGen** → `FunctionTool` 包装任何 Python 可调用对象；提供 MCP 适配器。与 AG2 生态系统紧密耦合，用于智能体到智能体模式。
- **Agno** → `@tool` 装饰器或 `BaseTool` 子类；MCP 适配器；工具可以在智能体和团队之间共享。

## 技能

> 你可以用一句话解释为什么某个框架适合某个智能体问题。

预构建清单：

1.  **绘制形态。** 这是一个图（类型化状态、命名转换）吗？一个角色扮演（专家交接工作）吗？一个聊天（智能体对话直到完成）吗？一个带有工具的单个智能体吗？
2.  **决定谁来分支。** 开发者决定分支 → LangGraph。经理智能体决定 → CrewAI 层级模式。聊天中浮现 → AutoGen。工具调用决定 → Agno。
3.  **检查状态预算。** 你需要从检查点恢复吗？时间旅行吗？运行中途需要人工中断吗？如果需要，LangGraph 是默认选择；Agno 会话涵盖会话范围内的状态。
4.  **检查成本预算。** LLM 选择的路由在每个回合都会产生额外的 token 成本。如果智能体每天运行数千次，请优先选择显式路由。
5.  **预算框架开销。** 每个框架都是另一个依赖项。如果任务是两次 LLM 调用和一个工具，编写 30 行纯 Python 代码；没有框架比没有框架更便宜。

在你能够绘制图、组织结构图、聊天或智能体框之前，不要急于选择框架。不要选择一个让你为了实现所需功能而与它的状态模型作斗争的框架。

## 决策矩阵

| 问题形态 | 首选框架 | 原因 |
|---------------|---------------------|-----|
| 带有类型化状态、人工审批、长时间运行的工作流 DAG | LangGraph | 一流状态、检查点器、中断、时间旅行。 |
| 具有明确角色的研究/写作流水线 | CrewAI（顺序）或 LangGraph 子图 | CrewAI 中表达按任务分配角色成本低廉；当分支变得复杂时，使用 LangGraph 进行扩展。 |
| 提议者-评论者或师生对话 | AutoGen | 双智能体聊天是其原生形态。 |
| 带有工具、会话、内存的单个智能体 | Agno | 最精简的设置，内置存储和内存。 |
| 数千个带有归约器的并行扇出 | LangGraph + `Send` | 唯一一个具有一流并行分派 API 的框架。 |
| 快速原型，无框架承诺 | 纯 Python + 提供商 SDK | 没有框架就是最快的框架。 |

## 练习

1.  **简单。** 接受相同的任务——“研究 Anthropic 总部，撰写一份 200 字的简报，并引用来源”——并使用 LangGraph（四个节点：规划、搜索、撰写、引用）和 CrewAI（三个角色：研究员、撰稿人、编辑）实现它。报告每次运行的 token 成本和代码行数。
2.  **中等。** 使用 AutoGen（研究员 ↔ 撰稿人聊天，编辑通过 `GroupChat` 加入）和 Agno（一个带有 `search_tools` 和 `write_tools` 的单个智能体，加上一个会话存储）构建相同的任务。对四种实现进行排名，依据 (a) 每次运行的成本，(b) 崩溃后恢复的能力，(c) 在写入步骤前注入人工审批的能力。
3.  **困难。** 构建一个决策树脚本 `pick_framework.py`，它接受一个简短的问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），并返回一个带有一句话理由的建议。在你自行设计的六个案例上验证它。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Orchestration | “智能体如何协调” | 决定哪个节点/角色/智能体接下来运行的层。 |
| Durable state | “重启后恢复” | 在进程死亡后仍能存活的状态，通常附加到检查点或会话存储。 |
| LLM-selected routing | “让模型决定” | 一个规划器 LLM 在每个回合选择下一步；灵活但每次决策都需要花费 token。 |
| Explicit routing | “开发者决定” | 一个 Python 函数或静态边选择下一步；成本低且可审计。 |
| Crew | “一个 CrewAI 团队” | 角色 + 任务 + 流程（顺序或层级）绑定到一个可运行的实体中。 |
| GroupChat | “AutoGen 的多智能体聊天” | N 个智能体之间由发言者选择器管理的对话。 |
| Team (Agno) | “多智能体 Agno” | 在一组智能体上的路由/协调/协作模式。 |
| StateGraph | “LangGraph 的图” | 类型化状态、节点、条件边、检查点器抽象。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — `StateGraph`、检查点器、中断、时间旅行。
- [CrewAI 文档](https://docs.crewai.com/) — `Crew`、流程、`Agent`、任务、过程。
- [AutoGen 文档](https://microsoft.github.io/autogen/) — `ConversableAgent`、`GroupChat`、团队、工具。
- [Agno 文档](https://docs.agno.com/) — `Agent`、`Team`、工作流、存储、内存。
- [Anthropic — 构建高效智能体 (2024 年 12 月)](https://www.anthropic.com/research/building-effective-agents) — 模式库（提示词链、路由、并行化、编排器-工作器、评估器-优化器），与框架无关。
- [Yao 等人，“ReAct: Synergizing Reasoning and Acting” (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个框架都包装起来的循环。
- [Wu 等人，“AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation” (2023)](https://arxiv.org/abs/2308.08155) — AutoGen 的设计论文。
- [Park 等人，“Generative Agents: Interactive Simulacra of Human Behavior” (UIST 2023)](https://arxiv.org/abs/2304.03442) — CrewAI 风格角色堆栈所基于的角色扮演基础。
- Phase 11 · 16 (LangGraph) — 本课程所参照的框架。
- Phase 11 · 19 (Reflexion) — 一种与 LangGraph 完美契合但与 CrewAI 格格不入的模式。
- Phase 11 · 22 (生产可观测性) — 如何对你选择的任何框架进行检测。