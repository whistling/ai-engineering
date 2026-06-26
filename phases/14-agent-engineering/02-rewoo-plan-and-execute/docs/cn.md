# ReWOO 和 Plan-and-Execute：解耦规划

> ReAct 将思考和动作交织在一条流里。ReWOO 将它们分离：先做一个完整的计划，然后执行。令牌数减少约 5 倍，在 HotpotQA 上绝对准确率提升约 4 个百分点，而且你可以把规划器蒸馏到一个 7B 的模型中。Plan-and-Execute 对其进行了泛化；Plan-and-Act 将其扩展到了网页导航场景。

**Type:** 构建  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 14 · 01 (智能体循环)  
**Time:** ~60 分钟

## 学习目标

- 解释为什么 ReWOO 的 Planner / Worker / Solver 划分相比 ReAct 的交错循环能节省令牌并提升鲁棒性。  
- 使用纯 stdlib 实现一个计划 DAG、一个按依赖顺序执行的执行器，以及一个将 Worker 输出组合起来的 Solver。  
- 使用 2026 年 Anthropic 提出的“五种工作流模式”框架，判断何时应选择先规划再执行（plan-then-execute）与何时采用交错的 ReAct。  
- 识别何时需要 Plan-and-Act 的合成计划数据（用于长时间跨度的网页或移动任务）。

## 问题描述

ReAct 的思考—动作—观测交错循环简单且灵活，但每次调用工具时都必须携带完整的先前上下文——包括每一个先前的思考。令牌使用量随着深度呈二次增长。更糟的是：当工具在循环中途失败时，模型必须从错误观测中重新推导出整个计划。

ReWOO（Xu 等，arXiv:2305.18323，2023 年 5 月）注意到这一点并下注：事先规划好全部步骤，并行获取证据，最后合成答案。一次 LLM 调用用于规划，N 次工具调用用于获取证据（可并行），一次 LLM 调用用于求解。代价是灵活性降低（计划是静态的），但换来更高的令牌效率和更明确的失败模式。

## 概念

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 生成一个 DAG。每个节点标明要调用的工具、参数以及依赖的早期节点（引用形式如 `#E1`、`#E2`）。Workers 按拓扑顺序执行节点。Solver 将所有证据缝合成最终答案。

### 为什么令牌数少约 5 倍

ReAct 的提示长度随步骤数线性增长。在第 10 步时，提示会包含思考 1、动作 1、观测 1、思考 2、动作 2、观测 2，依此类推。每个中间步骤还会冗余地包含原始提示。

ReWOO 支付一次较大的 planner 提示，N 个较小的 worker 提示（每个只包含工具调用，没有链式上下文），以及一次 solver 提示。论文在 HotpotQA 上测得令牌数约减少 5 倍，同时绝对准确率提升约 4 个百分点。

### 为什么更鲁棒

如果 ReAct 中的 worker 3 失败，循环需要在中途基于错误进行推理。在 ReWOO 中，worker 3 返回错误字符串；solver 在包含原始计划的上下文中看到该错误，并可以做出优雅的降级处理。失败定位按节点进行，而非按步骤进行。

### 规划器蒸馏

论文的第二个结果是：因为 Planner 在规划时不看到观测，你可以把 175B 教师模型的规划输出用来微调一个 7B 的模型。小模型可以负责规划；推理阶段不再需要大模型。这已成为常态——许多 2026 年的生产智能体使用小型规划器配合大型执行器或相反的分工。

### Plan-and-Execute（LangChain，2023）

LangChain 团队在 2023 年 8 月把 ReWOO 泛化成一个模式名：Plan-and-Execute。事先的规划器输出一个步骤列表，执行器按步骤运行，每步之后可以选择性地调用 replanner 来在观测后修正计划。这比 ReWOO 更接近 ReAct（因为 replanner 会把观测带回规划过程），但保留了令牌节省的优势。

### Plan-and-Act（Erdogan 等，arXiv:2503.09572，ICML 2025）

Plan-and-Act 将该模式扩展到长时域的网页与移动智能体。关键贡献是合成计划数据：使用带标注的轨迹生成器生成训练数据，其中显式包含计划。该方法用于微调规划器，使其在 WebArena 类任务上在超过 30–50 步时仍能保持工作，而单条 ReAct 轨迹会失去连贯性。

### 何时选择哪种模式

| Pattern | 何时使用 |
|---------|----------|
| ReAct | 短任务、环境未知、需要反应式异常处理 |
| ReWOO | 工具已知、结构化任务、对令牌敏感、证据可并行获取 |
| Plan-and-Execute | 类似 ReWOO，但在部分执行后允许重规划 |
| Plan-and-Act | 长时域（>30 步），网页/移动/电脑操作类任务 |
| Tree of Thoughts | 当搜索值得付出代价（见 Lesson 04） |

Anthropic（2024 年 12 月）建议：从最简单的方案开始。如果任务只是一次工具调用加上一个总结，不要去构建 ReWOO。如果任务是一个 40 步的研究任务，也不要只用 ReAct。

## 实现

`code/main.py` 实现了一个玩具版的 ReWOO：

- `Planner` — 一个脚本化策略，从提示生成计划 DAG。  
- `Worker` — 通过注册表分发每个节点的工具调用。  
- `Solver` — 脚本化的组合器，读取证据并生成最终答案。  
- 依赖解析 — 引用如 `#E1` 会被替换为先前 worker 的输出。

演示回答问题：“法国首都的人口，按百万四舍五入是多少？” 使用一个两步计划：（1）查询首都；（2）查询人口，然后求解。

运行它：

```
python3 code/main.py
```

运行痕迹会先显示完整计划，然后显示 worker 结果，最后显示 solver 的合成结果。比较令牌计数（我们打印了一个大致的字符计数）与 ReAct 风格的交错运行 —— 在这类结构化任务上 ReWOO 占优。

## 使用方式

LangGraph 将 Plan-and-Execute 作为一个配方发布（`create_react_agent` 用于 ReAct，Plan-Execute 使用自定义图）。CrewAI 的 Flows 直接对该模式建模：你事先定义任务，Flow DAG 会执行它们。Plan-and-Act 的合成计划数据方法仍主要处于研究阶段；但运行时模式（显式计划 DAG）已通过 LangGraph 和 CrewAI Flows 在生产中落地。

## 投产

`outputs/skill-rewoo-planner.md` 从用户请求和工具目录生成 ReWOO 计划 DAG。在交付执行器之前会校验计划（无环、每个引用可解析、每个工具存在）。

## 练习

1. 为独立的计划节点并行化 worker 执行。在一个包含 2 个并行组的 6 节点 DAG 上，这会带来什么收益？  
2. 添加一个在任一 worker 返回错误时触发的 replanner 节点。对 ReWOO 做出最小改动以使其变为 Plan-and-Execute，需要做哪些改动？  
3. 将 `Planner` 替换为一个小模型（7B 级别），并把 `Solver` 保持在 frontier 模型上。比较端到端质量——在哪些场景下这种分工会失败？  
4. 阅读 ReWOO 论文第 4 节关于规划器蒸馏的部分。概念上复现 175B -> 7B 的结果：你需要哪些训练数据，如何对计划质量进行评分？  
5. 将玩具实现移植到 Plan-and-Act 的轨迹形态：计划为序列而不是 DAG。有哪些权衡发生变化？

## 关键术语

| 术语 | 常说的表述 | 真实含义 |
|------|------------|----------|
| ReWOO | “Reasoning without observations” | 先规划，然后并行获取证据，最后求解 —— 规划提示中不包含观测 |
| Plan-and-Execute | “LangChain 的 plan-execute 模式” | 在执行后可选地重规划的 ReWOO 变体 |
| Plan-and-Act | “扩展的 plan-execute” | 对长时域任务的显式规划器/执行器分工，并使用合成计划训练数据 |
| Evidence reference | “#E1, #E2, ...” | 计划节点占位符，在派发时被先前 worker 的输出替换 |
| Planner distillation | “小型规划器，大型执行器” | 在大型教师模型的规划轨迹上微调小模型 |
| Token efficiency | “更少的往返调用” | 论文中在 HotpotQA 上令牌数减少约 5 倍 |
| DAG executor | “拓扑调度器” | 以依赖顺序运行计划节点；在每一层可并行执行 |

## 延伸阅读

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — 规范论文  
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — 使用合成计划扩展的规划-执行方法  
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — 框架配方与教程  
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 关于选择最简单有效模式的建议