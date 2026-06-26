# Supervisor / Orchestrator-Worker Pattern

> 一个主导代理负责规划与分配；专门化的工人（workers）在并行上下文中执行并汇报结果。这就是 Anthropic 的 Research 系统背后的模式（主导为 Claude Opus 4，子代理为 Sonnet 4），在内部研究评估上相比单代理 Opus 4 测得 +90.2% 的提升。Anthropic 的工程文章指出，BrowseComp 上 80% 的方差仅由令牌使用量解释——多代理的胜出在很大程度上是因为每个子代理获得了全新的上下文窗口。本文从原语构建出 supervisor 模式，并覆盖来自 2026 年生产部署的工程经验教训。

**Type:** 学习 + 构建  
**Languages:** Python（标准库，`threading`）  
**Prerequisites:** 阶段 16 · 04（原始模型）  
**Time:** ~75 分钟

## 问题

研究任务是单代理系统典型会失败的任务。你会问“在 2023 到 2026 年间，多代理系统发生了什么变化？”单代理顺序地阅读五篇论文，用一半的上下文填满它们的文本，然后还要对所有论文一起推理。到读到第五篇时，它已经忘记了第一篇。它无法并行化。

supervisor 模式解决了这个问题：一个主导代理规划搜索，把每个子问题委派给一个 worker，并负责综合。每个 worker 为一个狭窄的问题获得自己的 200k 令牌的上下文窗口。主导从不直接查看原始论文——只看到 worker 的摘要。

Anthropic 的生产 Research 系统在内部研究评估上报告相比单一 Opus 4 提升 +90.2%。同一篇文章指出，BrowseComp 的 80% 方差可由“仅令牌使用量”解释。对子代理提供新鲜上下文是主要机制。

## 概念

### 模式

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

主导从不阅读原始材料。各 worker 在主导合成之前互不见面。每一条箭头都是带有狭窄工件（artifact）的交接。

### 为什么它能胜出

三个机制：

1. **每个子代理的新鲜上下文。** 探索 “FIPA-ACL 遗产” 的 worker 不会携带主导为规划而消耗的 40k 令牌。它为单个问题获得一个 200k 令牌的上下文窗口。
2. **通过提示实现的专业化。** 主导的提示是“分解并综合”，而不是“做研究”。每个 worker 的提示都很窄：例如“找出 X 中发生了什么变化”。聚焦的提示产生聚焦的输出。
3. **并行性。** Workers 并发运行。实际耗时大致为 `max(worker_times) + plan + synthesis`，而不是 `sum(worker_times)`。

### 工程经验教训（Anthropic 2025）

Anthropic 的文章列出若干生产环境经验，至 2026 年仍然适用：

- **按查询复杂度调整规模。** 简单查询：一个代理，3–10 次工具调用。复杂查询：10+ 个代理。估算工作量应由主导来做，而不是调用者。
- **先广泛后深入。** 先把问题分解为广泛的子问题，再根据需要为每个子问题生成更多 worker 以加深讨论。
- **Rainbow 部署（渐进式放量）。** 代理是长时运行且有状态的。传统的蓝绿部署不适用。Anthropic 使用 rainbow：逐步放量新版本，同时旧版本缓慢回收。
- **令牌使用占主导。** 多代理的令牌消耗约为单代理的 ~15×。只有在任务价值足够时才运行多代理。

### LangGraph 的转变

LangGraph 最初发布了一个带有高级 `create_supervisor` 辅助函数的 `langgraph-supervisor` 库。到 2025 年，LangChain 将推荐改为通过工具调用（tool-calling）直接实现 supervisor 模式，因为工具调用能更好地控制“监督者看到什么”（上下文工程）。该库仍然可用；文档现更推荐工具调用形式。

### 失效模式

- **主导幻觉式规划。** 如果主导生成的子问题并未真正把问题分解开，workers 就会在错误的目标上进行精确研究。
- **Workers 过度探索。** 在没有明确范围边界的情况下，workers 会偏离其分配的子问题并污染综合步骤。
- **综合冲突。** 两个 worker 返回矛盾的事实。主导要么重新提问（增加一轮），要么显式指出分歧。沉默地选择一方是最糟的失败：用户永远不知道发生了分歧。

### 何时不应使用 supervisor

- **有序依赖的任务。** 如果第 2 步确实需要第 1 步的输出，并行性不会带来任何收益。应使用流水线（例如 CrewAI Sequential、LangGraph 线性图）。
- **简单查询。** 单代理更快、更便宜。在生成 workers 之前，先用主导做“按复杂度扩展”的检查。
- **严格确定性要求。** Supervisor 依赖 LLM 选择的委派。当审计/重放比适应性更重要时，静态图更合适。

```figure
supervisor-hierarchy
```

## 构建它

`code/main.py` 实现了一个由三个并行 worker 组成的 supervisor，使用 `threading`。主导将查询分解为子问题，workers 在每个子问题上并发运行，主导负责综合。没有真实的 LLM —— workers 被脚本化以模拟抓取并总结的过程。

关键结构：

- `Lead.plan(query)` 将查询分成 3 个子问题。
- `Worker.run(sub_q)` 返回一个伪摘要（在生产中可以是任何使用工具的 agent）。
- `Lead.run(query)` 启动线程中的 workers、join 并进行综合。

运行：

```
python3 code/main.py
```

输出显示计划、并行 worker 跟踪（含开始/结束时间戳）以及最终综合。你可以看到实际耗时的收益：三个各 0.3 秒的 worker 大约在 ~0.35 秒内完成，而不是 0.9 秒。

## 使用它

`outputs/skill-supervisor-designer.md` 接受一个用户查询并生成一个基于 supervisor 模式的设计：主导系统提示、worker 角色、子问题分解规则和综合模板。在构建新的研究型 agent 系统之前使用此文件。

## 部署注意事项

在部署 supervisor 模式之前的清单：

- **模型配对。** 主导部署在推理层级的模型上（Opus 类，`o3` 类）。Workers 使用更快、更便宜的模型（Sonnet，`o4-mini`）。
- **Worker 超时。** 任何超过中位运行时 2× 的 worker 都会被终止；主导要么以更窄的范围重启该 worker，要么在没有它的情况下继续。
- **每个 worker 的令牌上限。** 硬性限制（例如期望综合输入的 10×）可以防止 runaway worker 让预算失控。
- **可观测性。** 跟踪主导的计划、每个 worker 的工具调用以及综合结果。这是事后调试的基础。
- **Rainbow 放量。** 有状态的长时运行代理需要逐步版本迁移，而不是热切换。

## 练习

1. 运行 `code/main.py`，然后把主导改成生成 5 个 worker 而不是 3 个。观察实际耗时的变化。在这个演示中，多少个 worker 时生成开销超过并行带来的节省？
2. 实现 worker 超时：终止任何运行超过 0.5 秒的 worker，并让主导综合剩余结果。你需要哪些可观测性来确认某个 worker 被中断？
3. 在主导的综合中加入冲突检测步骤：如果两个 worker 返回矛盾答案，主导应标注出分歧而不是直接选一方。如何在不调用 LLM 的情况下检测矛盾？
4. 阅读 Anthropic 的 Research-system 工程文章。列出三个这份玩具 demo 需要采用的实践，以便在生产中运行。
5. 比较 LangGraph 的 `create_supervisor`（遗留）与新的工具调用推荐。哪种方式能更好地控制监督者看到的内容？为什么 Anthropic 明确只把子答案传递给综合，而不把 worker 的原始上下文传入？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Supervisor | "Lead agent" | 一个负责规划、委派和综合的编排型代理。自身不执行具体工作。 |
| Worker | "Subagent" | 由 supervisor 调用、范围狭窄且拥有自己上下文窗口的聚焦代理。 |
| Orchestrator-worker | "Supervisor pattern" | 同义不同名。2026 年文献同时使用这两个称呼。 |
| Fresh context | "Clean window" | worker 的上下文从其系统提示和分配的问题开始，而不是主导的历史。 |
| Rainbow deployment | "Gradual rollout" | 有状态的长时运行代理需要版本化的缓慢替换，而不是蓝绿式切换。 |
| Token dominance | "Context is the variable" | Anthropic 的结论：研究评估中 80% 的方差来自总令牌使用量，而非模型选择。 |
| Scale effort | "Match agent count to complexity" | 主导评估查询难度并据此生成 1 到 10+ 个 worker。 |
| Synthesis conflict | "Workers disagree" | 两个 worker 返回矛盾事实；主导必须显化分歧，而不是沉默地选一方。 |

## 深入阅读

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — supervisor 模式的生产参考文档  
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 现推荐的基于工具调用的 supervisor 实现形式  
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — 仍在 2026 年生产中使用的遗留辅助（legacy helper）  
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 基于交接的 supervisor 变体