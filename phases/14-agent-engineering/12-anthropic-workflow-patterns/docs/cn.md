# Anthropic 的工作流模式：简单优于复杂

> Schluntz 和 Zhang（Anthropic，2024 年 12 月）将工作流（预定义路径）与智能体（动态使用工具）区分开来。五种工作流模式覆盖了大多数用例。从直接的 API 调用开始。只有在步骤无法预测时才加入智能体。

**Type:** 学习 + 构建
**Languages:** Python（stdlib）
**Prerequisites:** Phase 14 · 01（智能体循环）
**Time:** ~60 分钟

## 学习目标

- 说出 Anthropic 的五种工作流模式：提示词链（prompt chaining）、路由（routing）、并行化（parallelization）、编排者-工作者（orchestrator-workers）、评估器-优化器（evaluator-optimizer）。
- 解释智能体与工作流的区别以及各自的工程成本。
- 识别何时选择工作流而非智能体（反之亦然）。
- 使用 stdlib 对一个脚本化的 LLM 实现所有五种模式。

## 问题

团队在许多只需单次函数调用的问题上倾向于采用多智能体框架。代价是真实存在的：框架增加层级，模糊提示词，隐藏控制流，并引入过早的复杂性。Schluntz 和 Zhang 在 2024 年 12 月的文章是最常被引用的行业反驳：先从简单开始，只有当复杂性带来的收益超过代价时才引入它。

## 概念

### 工作流 vs 智能体

- Workflow（工作流）。通过预定义的代码路径来编排 LLM 和工具。工程师掌控图。
- Agent（智能体）。LLM 动态地指挥自己的工具并自行决定步骤。模型掌控图。

两者各有适用场景。工作流更便宜、更快速、更易调试。智能体能解决开放式问题，但使故障模式更难推理。

### 增强型 LLM

所有五种模式的基础：一个 LLM 连接三类能力——检索（search）、工具（actions）、记忆（persistence）。任何 API 调用都可以使用这些能力。

### 五种模式

1. 提示词链（Prompt chaining）。调用 1 的输出作为调用 2 的输入。适用于任务具有清晰线性分解的情况。步骤间可选地加入程序化门控。
2. 路由（Routing）。分类器 LLM 决定调用哪个下游 LLM 或工具。适用于需要不同处理流程的类别性输入（一级支持 vs 退款 vs Bug vs 销售）。
3. 并行化（Parallelization）。并发运行 N 次 LLM 调用，聚合结果。有两种形态：分段（sectioning，不同片段）和投票（voting，相同提示 N 次运行，多数/合成）。
4. 编排者-工作者（Orchestrator-workers）。一个编排者 LLM 动态决定要运行哪些工作者（也可能是 LLM），并综合它们的输出。类似于智能体循环，但编排者不会无限循环。
5. 评估器-优化器（Evaluator-optimizer）。一个 LLM 提出答案，另一个 LLM 对其评估。迭代直到评估器通过。即 Self-Refine（第 05 课）的广义化。

### 工作流胜过智能体的情况

- 可预测的任务。如果你能列举步骤，就应该用工作流。
- 有成本限制的任务。工作流的步骤数是有界的；智能体可能发散。
- 有合规要求的任务。审计者希望读取图，而不是从轨迹中推断它。

### 智能体胜过工作流的情况

- 开放式研究。当下一步依赖于上一步的返回内容时。
- 可变长度任务。可能需要分钟到数小时，且步骤数未知。
- 新颖领域。当你还不知道合适的工作流时——先探索，后代码化。

### 上下文工程伴随学科

“Effective context engineering for AI agents”（Anthropic，2025）将邻近学科形式化：200k 的窗口是预算，而非容器。什么该包含、何时压缩、何时让上下文增长。在本课程 Phase 14 关于上下文压缩的课（在本课程重新编号之前为 Phase 14 早期的第 06 课）中有详细覆盖。

## 构建

`code/main.py` 针对一个 `ScriptedLLM` 实现了所有五种工作流模式：

- `prompt_chain(input, steps)` — 顺序执行。
- `route(input, classifier, handlers)` — 分类 + 分派。
- `parallel_vote(prompt, n, aggregator)` — N 次运行，聚合结果。
- `orchestrator_workers(task, workers)` — 编排者选择工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — 迭代直到通过。

运行它：

```
python3 code/main.py
```

每种模式都会打印其跟踪。每种模式的总代码行数约为 10–15 行；框架的成本通常以数千行来衡量。

## 使用建议

- 对大多数任务使用直接的 API 调用。
- 仅在模式确实需要持久状态（如 LangGraph）、参与者模型并发（如 AutoGen v0.4）或角色模板化（如 CrewAI）时才使用框架。
- 当你想要 Claude Code 那种代码骨架但不想重建它时，可以考虑 Claude Agent SDK。

## 上线

`outputs/skill-workflow-picker.md` 为给定任务描述选择合适模式，包含决策理由和当工作流不足时重构为智能体的路径。

## 练习

1. 为路由实现置信度阈值。低于阈值 -> 升级给人工。对于一级支持用例，阈值应落在何处？
2. 给 `parallel_vote` 添加超时。当某次调用挂起时会发生什么？如何在缺失投票的情况下聚合？
3. 将 `evaluator_optimizer` 改成 bandit：在迭代中保留前两名输出，以免后期的优秀结果被较差的结果覆盖。
4. 将提示词链与路由结合：路由器在三条链中选一条。测量与单一大提示备选方案的 token 成本。
5. 选择你生产环境中的一个功能。画出工作流图。统计步骤数。智能体在这里真的是更好的选择吗？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Workflow | "Predefined flow" | 工程师掌控的 LLM 与工具调用图 |
| Agent | "Autonomous AI" | 模型掌控的图；动态指挥工具 |
| Augmented LLM | "LLM with tools" | LLM + 检索 + 工具 + 记忆；原子单元 |
| Prompt chaining | "Sequential calls" | 调用 N 的输出作为调用 N+1 的输入 |
| Routing | "Classifier dispatch" | 选择哪条链/模型来处理输入 |
| Parallelization | "Fan out" | N 个并发调用；按分段或投票聚合 |
| Orchestrator-workers | "Dispatcher agent" | 编排者 LLM 动态选择专用 LLMs 并汇总 |
| Evaluator-optimizer | "Proposer + judge" | 提议者 + 评判者；迭代直到评估器通过（Self-Refine 广义化） |

## 延伸阅读

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 五种工作流模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 伴随学科
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 何时让有状态图的成本物有所值
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 编排者-工作者模式的产品化实现