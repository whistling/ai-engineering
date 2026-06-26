# Orchestration Patterns: Supervisor, Swarm, Hierarchical

> Four orchestration patterns recur across 2026 frameworks: supervisor-worker, swarm / peer-to-peer, hierarchical, debate. Anthropic's guidance: "It's about building the right system for your needs." Start simple; add topology only when a single agent plus five workflow patterns is insufficient.

**Type:** 学习 + 构建  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 14 · 12 (工作流模式), Phase 14 · 25 (多智能体辩论)  
**Time:** ~60 分钟

## 学习目标

- 说出四种常见的编排模式以及每种适用的场景。
- 描述 2026 年 LangChain 的建议：基于工具调用的监督 vs 使用 supervisor 库的区别。
- 解释 Anthropic 的“构建合适系统”规则以及它如何限制拓扑选择。
- 在 stdlib 上针对同一脚本化 LLM 实现这四种模式。

## 问题背景

团队在未真正需要多智能体之前就去追求“多智能体”。四种模式在各框架中反复出现；一旦你能命名它们，就可以选择合适的模式 —— 或者彻底跳过拓扑。

## 概念

### Supervisor-worker

- 一个中央路由 LLM 将任务分派给专业化的 agents。
- 决策选项：回到自身循环、交给某个专家、或终止。
- 专家之间不直接通信；所有路由都通过 supervisor 进行。

框架示例：LangGraph 的 `create_supervisor`、Anthropic 的 orchestrator-workers、CrewAI 的 Hierarchical Process。

**2026 LangChain 建议：**通过直接的工具调用实施监督，而不是 `create_supervisor`。这样可以获得更细粒度的上下文工程控制 —— 你可以精确决定每个专家能看到什么。

### Swarm / peer-to-peer

- Agents 通过共享的工具面直接相互交接。
- 没有中央路由器。
- 比 supervisor 延迟更低（跳数更少）。
- 更难以推理（没有单一控制点）。

框架示例：LangGraph 的 swarm 拓扑、OpenAI Agents SDK 的交接（当所有 agent 都能相互交接时）。

### Hierarchical

- 监督者管理子监督者，再管理工人。
- 在 LangGraph 中实现为嵌套子图；在 CrewAI 中为嵌套 crew。
- 可以扩展到大量 agent，但代价是运维复杂度增加。

什么时候需要：当单个 supervisor 的上下文预算无法容纳对所有专家的描述时。

### Debate

- 并行的提议者 + 迭代的交叉批评（Lesson 25）。
- 严格来说不完全是编排 —— 更像是验证 —— 但在框架中常作为一种拓扑选项出现。

### CrewAI 的 Crew vs Flow

CrewAI 将两种部署模式形式化：

- **Flow** 用于确定性的事件驱动自动化（对生产环境的推荐起点）。
- **Crew** 用于基于角色的自治协作。

这与上面四种模式是正交的，但与拓扑有映射：Flow 通常是 supervisor 或 hierarchical；Crew 通常是带有 LLM 路由器的 supervisor。

### Anthropic 的指导

“在 LLM 领域的成功不是构建最复杂的系统，而是为你的需求构建合适的系统。”

决策顺序：

1. 单智能体 + 工作流模式（Lesson 12）—— 从这里开始。
2. Supervisor-worker —— 当你有 2-4 个专家时。
3. Swarm —— 当延迟比推理清晰度更重要时。
4. Hierarchical —— 仅在 supervisor 上下文预算不足时。
5. Debate —— 当准确性比成本更重要时。

### 此模式出错的常见情况

- 拓扑优先思维。还没确定多智能体解决什么问题就说“我们需要多智能体”。
- Swarm 中的来回交接。A -> B -> A -> B。使用跳数计数器。
- 伪层级结构。三层只是因为“企业需要”；实际上只有两个团队。应合并简化。

## 构建它

`code/main.py` 在 stdlib 上针对脚本化 LLM 实现了所有四种模式：

- `Supervisor` — 中央路由器。
- `Swarm` — 点对点直接交接。
- `Hierarchical` — 监督者的监督者。
- `Debate` — 并行提议者 + 批评。

每种模式处理相同的三意图任务（退款 / Bug / 销售）。调用跟踪形状不同。

运行它：

```
python3 code/main.py
```

输出：每种模式的调用跟踪 + 操作计数。Supervisor 最简洁；swarm 最短；hierarchical 最深；debate 成本最高。

## 使用场景

- **LangGraph** 适用于 supervisor 和 hierarchical（嵌套子图）。
- **OpenAI Agents SDK** 适用于 将交接作为工具 的场景（supervisor 形态）。
- **CrewAI Flow** 适用于生产确定性场景。
- **自定义实现** 适用于 debate 或当你需要精确控制时。

## 上线指南

`outputs/skill-orchestration-picker.md` 选择拓扑并实现它。

## 练习

1. 将一个 supervisor-worker 转换为 swarm，通过移除路由器。有哪些问题？有哪些改进？
2. 给 swarm 添加跳数计数器：超过 3 次交接就拒绝。能否捕获 A->B->A 的来回情况？
3. 为一个包含 12 个专家领域构建两级层级系统。在哪个地方如果不嵌套会出现上下文预算失败？
4. 在接近生产规模的工作负载上对这四种模式进行性能分析。哪种在延迟、成本、准确性、可调试性上胜出？
5. 阅读 Anthropic 的“Building Effective Agents”文章。将每个生产流映射到四种模式中的一种。有没有不能清晰映射的流程？

## 关键术语

| 术语 | 常说法 | 实际含义 |
|------|--------|----------|
| Supervisor-worker | “路由器 + 专家” | 中央 LLM 将任务分派给专家；专家之间不直接通信 |
| Swarm | “点对点” | 通过共享工具进行直接交接；没有中央路由器 |
| Hierarchical | “多级监督” | 为大规模 agent 人群使用嵌套子图 |
| Debate | “提议 + 批评” | 并行提议者、交叉批评（Lesson 25） |
| Tool-call-based supervision | “不依赖库的监督” | 将监督实现为直接的工具调用以控制上下文 |
| Crew | “自治团队” | CrewAI 的基于角色的协作模式 |
| Flow | “确定性工作流” | CrewAI 的事件驱动生产模式 |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 五种模式 + agent vs workflow
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor、swarm、hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debate 模式