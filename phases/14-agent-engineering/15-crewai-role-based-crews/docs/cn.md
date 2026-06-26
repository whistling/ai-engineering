# CrewAI: 基于角色的 Crews 和 Flows

> CrewAI 是 2026 年的基于角色的多智能体框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crews（自治、基于角色的协作）和 Flows（事件驱动、确定性）。文档直言不讳：“对于任何可投入生产的应用，请从 Flow 开始。”

**Type:** 学习 + 构建  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 14 · 12（工作流模式），Phase 14 · 14（参与者模型）  
**Time:** ~75 分钟

## 学习目标

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process）以及每个原语的职责范围。  
- 区分 Sequential、Hierarchical 和计划中的 Consensus 流程；为不同工作负载选择合适的一种。  
- 区分 Crews（自治、基于角色）与 Flows（事件驱动、确定性），并解释文档中为何建议用于生产。  
- 使用 `@tool` 装饰器和 `BaseTool` 子类接入工具；权衡结构化输出与自由文本。  
- 说出 CrewAI 的四种内存类型及各自适用场景。  
- 实现一个标准库级别的三智能体 crew（researcher、writer、editor），生成一份简报。  
- 识别三种 CrewAI 失败模式：提示词膨胀（prompt-bloat）、管理者 LLM 成本（manager-LLM tax）、脆弱的交接（brittle handoffs）。

## 问题

采用多智能体框架的团队会遇到相同的难题。“自治协作”在演示中很吸引人。但随后客户提交了一个错误，你需要确定性回放；或财务想知道按次运行的 LLM 路由成本；或值班想知道凌晨三点哪个 agent 卡住了。

自由形式的 LLM 路由 crew 无法清晰回答这些问题。纯 DAG 可以回答所有问题，但又失去了头脑风暴类智能体所需的探索性。

CrewAI 的分离在于诚实地表明了权衡。Crews 适用于协作、基于角色、探索性工作。Flows 适用于事件驱动、代码拥有、可审计的生产。相同框架、两种形态，根据场景选择。

## 概念

### 四个原语

CrewAI 的表面很小。记住这些，其它都是配置。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。backstory（背景故事）是承重的。它影响语气、判断以及智能体何时停止。tools 是智能体可以调用的函数（见下文）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。可重用的工作单元。`expected_output` 是契约。`context` 列出上游任务，其输出会传入当前任务。`output_pydantic` 强制结构化输出。
- **Crew。** 容器。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory` + `verbose` + `manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensus（计划中）。决定运行的形态。

Agents 之间不会直接相互可见。Tasks 指定 agent。Crew 将任务排序。Process 决定谁挑下一个任务。这就是全部心智模型。

> **Validated against** CrewAI 0.86 (2026-05)。更新版本可能重命名或合并某些 process 类型；在依赖特定形态前，请查阅 [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential。** 任务按声明顺序运行。任务 N 的输出可作为 `context` 提供给任务 N+1。成本最低。最可预测。适用于顺序固定的场景。  
- **Hierarchical。** 一个 manager Agent（额外的 LLM 调用）在专家间路由。CrewAI 会根据你的 `manager_llm` 配置或默认值生成该 manager。manager 每轮选择下一个任务，并可拒绝或重路由。当你有四个或更多专家且执行顺序确实依赖于先前输出时使用。  
- **Consensus。** 计划中但当前公共 API 未实现。文档保留该名称给将来的投票式流程。今天不要依赖它。

Hierarchical 在每轮专家调用之外增加一次 manager 的 LLM 调用。在五步运行上，token 成本可能翻倍或三倍。只有在需要路由时才采用它。

### Crews vs Flows

这是 2026 年文档开篇的框架。

- **Crew。** 由 LLM 驱动的自治。框架在运行时选择形态。适用于：研究、头脑风暴、草稿、路径本身是答案的一部分的场景。难以回放，难以测试。原型化成本低。  
- **Flow。** 事件驱动、代码拥有的图。`@start` 标记入口。`@listen(topic)` 标记当另一步发出该 topic 时触发的步骤。每一步都是普通的 Python（内部可以调用一个 Crew）。适用于：生产。可观测、可测试、确定性。

文档 2026 年的生产推荐：从 Flow 开始。在 Flow 步骤内部以 `Crew.kickoff()` 的形式折叠 Crews，当自治带来的价值高于其成本时再启用。Flow 提供审计轨迹，Crew 提供探索。组合使用，而不是二选一。

### 工具集成

给 Agent 提供工具有三种方式。选择最简单且适配的。

1. **`@tool` 装饰器。** 纯函数成为工具。签名即为 schema；docstring 是 LLM 可见的描述。适合一次性的小工具。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """返回查询的顶级结果。"""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 基于类的工具，带显式参数 schema、异步支持、重试。用于工具有状态（客户端、缓存）或需要结构化参数时。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "搜索网络并返回顶级结果。"
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           # 使用内部 client 执行搜索并返回结果
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 提供一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一行 import 即可接入。

结构化输出使用 Pydantic。在 Task 上传入 `output_pydantic=MyModel`。CrewAI 会将 LLM 响应与模型校验并进行强制转换或重试。将此与精确的 `expected_output` 字符串配对。自由文本输出适合草稿；结构化输出则是下游 Flows 可消费的类型。

### 内存钩子

CrewAI 开箱提供四种内存类型。它们可以组合：一个 Crew 可以同时启用全部四种。

> **Validated against** CrewAI 0.86 (2026-05)。近期发布将一切路由到统一的 `Memory` 系统以封装这四种存储。下述概念模型仍然适用，但公共类表面在新版本中可能会塌缩为单个 `Memory` 入口；请查阅 [CrewAI memory docs](https://docs.crewai.com/concepts/memory) 以获取当前 API。

- **Short-term。** 单次运行内的对话缓冲。运行结束时清空。  
- **Long-term。** 跨运行持久化。存储在向量数据库（默认 Chroma，可替换）。按与当前任务的相似度检索。  
- **Entity。** 每实体事实存储。“客户 X 使用企业版”。以实体为键而非相似度。跨运行存活。  
- **Contextual。** 组装时检索。Agent 需要时即时拉取相关记忆，而不是预先加载。

在 Crew 上通过 `memory=True` 或按类型配置启用。后端嵌入提供者可配置（默认 OpenAI，可替换为本地）。内存在与轻量框架对比中显示其价值；纯 LangGraph 要你自己接线这些存储。

### 何时适合使用 CrewAI

- 具有三到六个命名角色和协作工作流的场景。起草、审阅、规划、头脑风暴。  
- 路由决策本身依赖 LLM 判断（Hierarchical）。  
- 团队更愿意阅读 `role + goal + backstory` 而不是图定义时。

### 何时不适合使用 CrewAI

- 有严格顺序的确定性 DAG。请使用 LangGraph（Lesson 13）。图形化结构是正确的抽象；CrewAI 基于角色的表述会带来摩擦。  
- 亚秒级延迟预算。Hierarchical 增加往返。即便 Sequential 也会串行包含背景故事和先前输出的提示词。  
- 单智能体循环。跳过框架；一个 agent 循环（Lesson 1）加上一个工具注册表更简短。

Lesson 17（智能体框架权衡）有矩阵化对比。简短结论：CrewAI 位于“协作、基于角色”角。

### 依赖形态

独立于 LangChain。支持 Python 3.10 到 3.13。使用 `uv`。Star 数：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（2026-05 时的快照）。文档记录了对 AWS Bedrock 的集成；供应商基准在 QA 工作负载上报告相较 LangGraph 有显著速度提升，但其方法学（数据集、硬件、评测指标）未公开，因此将框架厂商的数字视为方向性参考。

### 该模式常见的失败点

- **来自 backstories 的提示词膨胀（prompt-bloat）。** 每个 agent 的 2000 字背景故事和五个 agent 会在首次工具调用前就耗尽上下文预算。将 backstory 控制在 200 字以内。跨 agent 重用短语；不要五次重复风格说明。  
- **管理者 LLM 的 token 成本（manager-LLM tax）。** Hierarchical 在每次专家调用前额外增加一次 manager 调用。五任务的 crew 实际上会有六次 LLM 调用（而不是五次），且 manager 调用包含完整任务列表与先前输出。除非路由依赖于输出，否则切回 Sequential。  
- **脆弱的交接（brittle handoffs）。** 任务 N 的 `expected_output` 是“一个大纲”。任务 N+1 把它当作 `context` 并尝试解析三节，但 LLM 生成了四节，下游 Agent 随意补写。用 Task N 的 `output_pydantic` 修正，让任务 N+1 读取类型化对象而不是自由文本。  
- **Crew 被当作生产环境直接部署。** 自由形式的 Crew 直接上线没有 Flow 包裹。输出变异性高；无法回放；值班无法对比出问题运行和正常运行的差异。用 Flow 包裹。

## 实战构建

`code/main.py` 实现了两种形态的标准库版本以及一个三智能体 crew。

形态：

- 与 CrewAI 表面相匹配的 `Agent`、`Task` dataclass。  
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，并将输出作为 `context` 传递。  
- `HierarchicalCrew.kickoff(topic)` 增加一个 manager Agent，每轮选择下一个专家，直到返回 "done" 为止。  
- 带 `@start` 和 `@listen(topic)` 装饰器的 `Flow`，一个小型事件循环和 trace。  
- 模拟 `@tool` 的 `tool(name)` 装饰器。  
- 带有 `short_term`、`long_term`、`entity` 存储的 `Memory`；模拟相似度检索使用 numpy。  
- 模拟的 LLM 响应是基于 role 与输入前缀的硬编码字符串。无网络。确定性。

具体演示：researcher、writer、editor 三人组就 “agent engineering 2026” 生成一份简报。Researcher 拉取（模拟的）来源，Writer 起草，Editor 精简。同一 crew 也通过一个 Flow 运行以展示确定性形态。

运行：

```bash
python3 code/main.py
```

Trace 包括：顺序 crew 将输出以 `context` 形式串联；Hierarchical crew 的 manager 轮流选择（researcher、writer、editor，然后 "done"）；Flow 按显式 topic（`researched`、`drafted`、`edited`）运行相同三步；工具调用通过 `@tool` 路由；长期内存在的 memory 在两次 kickoff 间存活。

Crew 的 trace 更为流动；manager 理论上可以重新排序。Flow 的 trace 是固定的。这个选择就是本课的要点。

## 使用方式

- 生产环境使用 CrewAI Flow。即便 Flow 只是一步并在内部调用 `Crew.kickoff()`，Flow 也提供审计边界。  
- CrewAI Crew（Sequential）适用于顺序明确的协作工作，尤其是草稿与评审循环。  
- CrewAI Crew（Hierarchical）当路由依赖输出且你有四个或更多专家时使用。  
- LangGraph（Lesson 13）适用于显式状态机、耐久恢复、严格顺序。  
- AutoGen v0.4（Lesson 14）适用于参与者模型并行与故障隔离。  
- OpenAI Agents SDK（Lesson 16）适合以 OpenAI 为主的产品，强调交接与护栏。  
- Claude Agent SDK（Lesson 17）适合以 Claude 为主的产品，带子智能体和会话存储。

## 部署建议

`outputs/skill-crew-or-flow.md` 会为一个任务在 Crew 与 Flow 间做抉择并搭起最小实现脚手架。严禁将没有 backstory 的 Crew、没有显式 topics 的 Flow、以及在专家少于三人的情况下使用 Hierarchical。

## 陷阱

- **Backstory 只是口味层面。** 它会影响输出。为每个 agent 测试三个变体；差异是真实存在的。选定一个并冻结。  
- **跳过 `expected_output`。** 没有每个任务的契约，下游任务会拾取 LLM 产生的任意内容。Crew 运行；审计失败。  
- **内存总是开启。** 每次运行都会写入长期记忆。向量库膨胀。检索变得嘈杂。只在事实确实需要持久化时写入。  
- **Manager 提示词漂移。** Hierarchical 的 manager 提示是隐含的。如果路由出现怪异，开启 verbose 并打印提示检查。  
- **工具副作用在 Crews 中。** Crew 可能比预期调用更多次工具。POST、DELETE、支付类操作应放在 Flow 步骤中，而非 Crew 内的工具。

## 练习

1. 将 Sequential crew 转换为 Flow。统计可变性下降的接触点。记录可读性下降的位置。  
2. 为 crew 添加实体内存：关于某客户的事实在 kickoffs 间持久化。验证检索能拉回正确实体。  
3. 实现一个 Hierarchical 流程：manager 在 writer 输出至少三段之前拒绝路由到 editor。跟踪重试过程。  
4. 为一个（模拟的）网络搜索实现 `BaseTool` 子类。比较 trace 与 `@tool` 装饰器版本的差异。  
5. 在 editor 任务上添加 `output_pydantic=Brief`，其中 `Brief` 包含 `title`、`summary`、`sections`。让 writer 任务一次输出格式错误的 JSON；在 trace 中验证 CrewAI 的重试行为。  
6. 阅读 CrewAI 的文档入门。将玩具实现移植到真实的 `crewai` API。标准库版本跳过了哪些保证？  
7. 将 AgentOps 或 Langfuse（Lesson 24）接入真实运行。标准库版本缺失了哪些 trace？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "Persona" | Role + goal + backstory + tools |
| Task | "Unit of work" | Description + expected output + assignee + optional structured output |
| Crew | "Agent team" | Container for Agents + Tasks + Process |
| Process | "Execution strategy" | Sequential / Hierarchical / Consensus (planned) |
| Flow | "Deterministic workflow" | Event-driven, code-owned, testable |
| Backstory | "Persona prompt" | Tone and judgment shaper for the Agent |
| `@tool` | "Function tool" | Decorator that turns a function into a tool the Agent can call |
| `BaseTool` | "Class tool" | Class-based tool with args schema, retries, async support |
| Entity memory | "Per-entity facts" | Memory scoped to a customer / account / issue |
| Long-term memory | "Cross-run memory" | Vector-backed memory that survives between kickoffs |
| Contextual memory | "Just-in-time retrieval" | Memory pulled at the moment the Agent needs it |
| Manager LLM | "Router agent" | Extra LLM in Hierarchical process that picks the next task |
| `expected_output` | "Task contract" | String that tells the Agent (and audit) what shape to return |

（表格中术语使用原始术语以便与代码/API 对齐；右栏给出更精确的含义。）

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：概念与建议的生产路径  
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)：事件驱动形态，`@start`，`@listen`  
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置工具包  
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)：short-term、long-term、entity、contextual  
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：什么时候多智能体有帮助，什么时候没有  
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案