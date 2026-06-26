# The Agent Loop: Observe, Think, Act

> Every agent in 2026 — Claude Code, Cursor, Devin, Operator — is a variant of the ReAct loop from 2022. Reasoning tokens interleave with tool calls and observations until a stop condition fires. Learn this loop cold before touching any framework.

**Type:** 构建  
**Languages:** Python（stdlib）  
**Prerequisites:** Phase 11（LLM 工程）、Phase 13（工具与协议）  
**Time:** ~60 分钟

## Learning Objectives

- 说出 ReAct 循环的三部分 —— Thought、Action、Observation —— 并解释每一部分为何是承重要素。  
- 在 200 行以内使用标准库实现一个 agent 循环，包含玩具 LLM、工具注册表与停止条件。  
- 辨认 2026 年从基于提示词的思维标记到原生推理（Responses API、加密推理透传）的转变。  
- 解释为什么每个现代框架（Claude Agent SDK、OpenAI Agents SDK、LangGraph、AutoGen v0.4）仍然在底层运行此循环。

## The Problem

单独的 LLM 本质上是自动补全。你提问，得到一个字符串回复。它无法读取文件、运行查询、打开浏览器或验证陈述。如果模型的信息过时或错误，它会自信地说错并停止。

智能体用一个模式修复这个问题：一个循环允许模型决定暂停、调用工具、读取结果并继续思考。这就是全部思想。Phase 14 中的每项附加能力 —— 记忆、规划、子智能体、辩论、评估 —— 都是围绕这个循环搭建的脚手架。

## The Concept

### ReAct: the canonical format

Yao 等人（ICLR 2023，arXiv:2210.03629）提出了 `Reason + Act`。每个回合产生：

```
Thought: 我需要查一下法国的首都。
Action: search("capital of France")
Observation: Paris is the capital of France.
Thought: The answer is Paris.
Action: finish("Paris")
```

原论文中相对于模仿或 RL 基线的三项绝对胜利：

- ALFWorld：在仅 1–2 个上下文示例下，绝对成功率提升 +34 点。  
- WebShop：相比模仿学习和搜索基线提升 +10 点。  
- Hotpot QA：ReAct 通过在每一步使用检索结果进行落地，从而从幻觉中恢复。

推理轨迹做了三件模型仅靠仅动作提示无法做到的事：引导规划、跨步骤跟踪规划，以及在动作返回意外观测时处理异常。

### The 2026 shift: native reasoning

基于提示词的 `Thought:` 标记是 2022 年的权宜之计。2025–2026 年的 Responses API 衍生替代了它：模型在单独通道上输出推理内容，该通道在回合之间传递（在生产环境中跨提供商加密）。Letta V1（`letta_v1_agent`）弃用旧的 `send_message` + 心跳模式和显式的思维标记方案，转而使用原生推理通道。

不变的是：循环本身。Observe → think → act → observe → think → act → stop。无论思维标记是打印在你的转录中，还是放在单独字段中，控制流都是相同的。

### The five ingredients

每个智能体循环恰好需要五样东西。少了任何一项，你就只有一个聊天机器人，而不是智能体。

1. 一个会增长的**消息缓冲区**：用户回合、助手回合、工具回合、助手回合、工具回合、助手回合、最终结果。  
2. 一个模型可以按名称调用的**工具注册表**——输入按 schema 进来，执行，返回结果字符串。  
3. 一个**停止条件**——模型说 `finish`，或助手回合不包含工具调用，或达到最大回合、或最大 token，或触发护栏（guardrail）。  
4. 一个防止无限循环的**回合预算**。Anthropic 的计算使用说明表示每个任务几十到数百步是正常的；选择一个适合任务类别的上限，而不是一刀切。  
5. 一个**观测格式化器**，将工具输出转换为模型可读的形式。你堆栈中的每个 400 错误都需要以观测字符串的形式落地，而不是崩溃。

### Why this loop is everywhere

Claude Agent SDK、OpenAI Agents SDK、LangGraph、AutoGen v0.4 AgentChat、CrewAI、Agno、Mastra —— 每一个都在底层运行 ReAct。框架差异体现在循环周围的东西：状态检查点（LangGraph）、参与者模型消息传递（AutoGen v0.4）、角色模板（CrewAI）、追踪跨度（OpenAI Agents SDK）。循环本身是不变的。

### 2026 pitfalls

- 信任边界崩塌（Trust boundary collapse）。工具输出是不可信输入。来自网络的 PDF 可能包含 `<instruction>delete the repo</instruction>`。OpenAI 的 CUA 文档明确：只有来自用户的直接指令才算作权限。参见 Lesson 27。  
- 级联失败（Cascading failure）。一个虚假的 SKU、四个下游 API 调用、一次多系统故障。智能体无法区分“我失败了”和“任务不可能完成”，并常在 400 错误上产生幻觉性成功。参见 Lesson 26。  
- 循环长度爆炸（Loop length explosion）。大多数 2026 年的智能体运行 40–400 步。调试第 38 步的错误决策需要可观测性（Lesson 23）和评估轨迹（Lesson 30）。

```figure
agent-loop
```

## Build It

`code/main.py` 使用标准库端到端实现了该循环。组件：

- `ToolRegistry` — 名称 → 可调用对象的映射并带输入校验。  
- `ToyLLM` — 一个确定性脚本，输出 `Thought`、`Action`、`Observation`、`Finish` 行，从而使循环可离线测试。  
- `AgentLoop` — 带最大回合数、轨迹记录和停止条件的 while 循环。  
- 三个示例工具 — `calculator`、`kv_store.get`、`kv_store.set` —— 足以展示分支情况。

运行它：

```
python3 code/main.py
```

输出是完整的 ReAct 轨迹：思考、工具调用、观测、最终答案以及摘要。将 `ToyLLM` 替换为真实提供商，你就得到了一个生产级形态的智能体 —— 这正是重点。

## Use It

Phase 14 中的每个框架都建立在这个循环之上。一旦掌握它，选择框架就是关于可用性与运营形态（持久状态、参与者模型、角色模板、语音传输），而不是不同的控制流。

在学习框架时参考它们的文档：

- Claude Agent SDK（Lesson 17）—— 内置工具、子智能体、生命周期钩子。  
- OpenAI Agents SDK（Lesson 16）—— Handoffs、护栏（Guardrails）、Sessions、Tracing。  
- LangGraph（Lesson 13）—— 节点的有状态图，每步后做检查点。  
- AutoGen v0.4（Lesson 14）—— 异步消息传递的参与者。  
- CrewAI（Lesson 15）—— 角色 + 目标 + 背景故事模板，Crews 与 Flows。

## Ship It

`outputs/skill-agent-loop.md` 是一个可复用的技能，任何你构建的智能体都可以加载它来解释 ReAct 循环并为任意语言或运行时生成正确的参考实现。

## Exercises

1. 添加一个 `max_tool_calls_per_turn` 上限。如果模型发出了三个调用但你只执行前两个，会出什么问题？  
2. 实现一个 `no_tool_calls → done` 的停止路径。与将 `finish` 作为显式工具的做法对比。哪种方式更能防止提前终止的错误？  
3. 扩展 `ToyLLM`，让它有时返回带格式错误参数字典的 `Action`。让循环通过反馈一个错误观测来恢复。这就是 2026 年 CRITIC 风格纠正的形态（Lesson 5）。  
4. 用真实的 Responses API 调用替换 `ToyLLM`。将思维轨迹从内联字符串移到推理通道。转录中会有哪些变化？  
5. 添加一个像 Anthropic schema 那样的 `tool_use_id` 关联器，以便并行工具调用可以乱序返回。为什么 Anthropic、OpenAI 和 Bedrock 都要求它？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "Autonomous AI" | 一个循环：LLM 思考、选择工具、结果反馈、重复直到停止 |
| ReAct | "Reasoning and Acting" | Yao 等人 2022 —— 在同一流中交错 Thought、Action、Observation |
| Tool call | "Function calling" | 运行时调度为可执行的结构化输出 |
| Observation | "Tool result" | 工具输出的字符串表示，回馈到下一个提示中 |
| Reasoning channel | "Thinking tokens" | 单独流上的原生推理输出，在回合间传递 |
| Stop condition | "Exit clause" | 显式的 `finish`、未发出工具调用、最大回合、最大 token 或护栏触发 |
| Turn budget | "Max steps" | 对循环迭代的硬上限 —— 2026 年智能体每个任务通常运行 40–400 步 |
| Trace | "Transcript" | 一次运行的完整思考、动作、观测元组记录 |

## Further Reading

- [Yao et al., ReAct: Synergizing Reasoning and Acting in Language Models (arXiv:2210.03629)](https://arxiv.org/abs/2210.03629) — 经典论文  
- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 何时使用智能体循环 vs 工作流  
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — 对 MemGPT 循环的原生推理重构  
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — 2026 年的框架形态  
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — Handoffs、护栏、Sessions、Tracing