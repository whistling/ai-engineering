# Handoffs and Routines — Stateless Orchestration

> OpenAI 的 Swarm（2024 年 10 月）将多代理编排提炼为两个原语：**routines**（将指令 + 工具作为系统提示）和 **handoffs**（返回另一个 Agent 的工具）。没有状态机，没有分支 DSL —— LLM 通过调用正确的 handoff 工具来路由。OpenAI Agents SDK（2025 年 3 月）是其生产级继任者。Swarm 本身仍然是最简洁的概念参考 —— 它的全部源码只需几百行。该模式之所以流行，是因为 API 面向大致为 “agent = prompt + tools; handoff = function returning agent”。限制：无状态，所以记忆由调用方负责。

**Type:** 学习 + 构建  
**Languages:** Python（stdlib）  
**Prerequisites:** Phase 16 · 04（Primitive Model）  
**Time:** ~60 分钟

## 问题

每个多代理框架都希望你学习它的 DSL：LangGraph 的节点和边、CrewAI 的 crews 和 tasks、AutoGen 的 GroupChat 和 managers。DSL 是真正的抽象，但它们会让整体感觉比实际需要的更沉重。

Swarm 往相反的方向推进：利用模型已经具备的工具调用能力。Handoffs 变成了工具调用。编排者是当前持有对话的 agent。状态机在代理的系统提示中隐式表示。

## 概念

### 两个原语

**Routine。** 定义代理角色和可用工具的系统提示。把它看作一个有作用域的指令集：“你是一个分诊代理；如果用户询问退款，就交接给退款代理。”

**Handoff。** 代理可以调用的一个工具，返回一个新的 Agent 对象。Swarm 运行时检测到 Agent 返回值并在下一轮切换活动代理。

这就是全部抽象。

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

分诊代理的系统提示让它基于用户消息选择正确的 handoff。LLM 的工具调用完成路由工作。

### 为什么它会流行

- 小而精的 API：只需学习两个概念。
- 利用模型已有能力：工具调用在各大提供方上已经是生产级别。
- 无需状态机负担：你不必描述整个图；代理的提示描述它们会交给谁。

### 无状态的权衡

Swarm 在运行间明确是无状态的。框架在一次运行内会保留消息历史，但不会持久化任何内容。记忆、连续性、长期任务 —— 都是调用方的问题。

在生产实现（OpenAI Agents SDK，2025 年 3 月）中，这一点是主要改进之一：SDK 在保留 handoff 原语的同时增加了内置会话管理、护栏和追踪。

### 适合 Swarm/handoffs 的场景

- 分诊模式。前线代理将用户路由到专科代理。
- 基于技能的交接。 “如果任务需要写代码，就调用 coder；如果需要调研，就调用 researcher。”
- 简短、有界的对话。客户支持、FAQ 转工单、简单工作流。

### Swarm 的短板

- 长会话且需共享记忆。Handoffs 会把对话状态重置为新代理的提示加上历史。跨代理的持久状态需要调用方管理。
- 并行执行。Handoff 是一次一个 —— 活动代理会切换。并行性需要调用方并行运行多个 Swarm 实例来实现。
- 审计与重放。无状态运行难以精确重放；LLM 的 handoff 选择不是确定性的。

### OpenAI Agents SDK（2025 年 3 月）

生产继任者增加了：

- 会话状态。跨运行的持久线程。
- 护栏。输入/输出验证钩子。
- 追踪。每次工具调用和 handoff 都被记录。
- Handoff 过滤器。控制交接时传递哪些上下文。

handoff 原语保留；生产级易用性在其周围被补充。

### Swarm vs GroupChat

两者都使用 LLM 驱动的路由，但它们在“谁决定下一个”上不同：

- GroupChat：由外部的选择器（函数或 LLM）从参与者中选择下一位发言者。
- Swarm：当前代理通过调用 handoff 工具选择它的继任者。

Swarm 是“agent 决定下一个”；GroupChat 是“manager 决定下一个。”Swarm 的决定存在于活动代理的工具调用中；GroupChat 的决定存在于 `GroupChatManager` 中。

## 构建它

`code/main.py` 从头实现了 Swarm：一个 Agent dataclass、一个 handoff 机制（工具返回 Agent），以及一个在检测到代理切换时处理切换的运行循环。

演示：一个分诊代理将用户路由到退款、销售或支持专家。每个专家都有自己的工具。运行循环会打印每次 handoff。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-handoff-designer.md` 为给定任务设计 handoff 拓扑：有哪些 agent、它们能调用哪些 handoffs、以及哪些上下文会被传递。

## 上线注意

检查清单：

- **Handoff 日志。** 每次 handoff 都写入一条追踪事件，包含 from-agent、to-agent、上下文快照。
- **上下文传递规则。** 决定在 handoff 时传递什么：完整历史（昂贵）、最近 N 条消息、或摘要。
- **Handoff 护栏。** 向具有不同工具权限的专家的 handoff 必须经过鉴权 —— 否则提示注入可能强制不希望的交接。
- **循环检测。** 两个代理反复互相交接是常见故障；用一个简单的最近 K 环形检查来检测。
- **后备代理。** 如果 handoff 目标不存在，则回退到一个安全默认代理。

## 练习

1. 运行 `code/main.py`，让分诊代理交接到 refund 代理。确认第二轮的活动代理是 refund。
2. 添加一个循环检测规则：如果相同的两个代理连续交接 3 次，强制退出。设计后备策略。
3. 阅读 OpenAI Agents SDK 关于 handoff 过滤器的文档。实现一个“交接时摘要”版本：外发代理在交接前将上下文压缩为要点摘要，然后再由进站代理接手。
4. 比较 Swarm handoff 与 GroupChatManager 的选择器。哪种模式更容易受到提示注入攻击，为什么？
5. 阅读 Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。指出 Swarm 做出的一个显式设计决策，以及 OpenAI Agents SDK 是否改变或保留了该决策。

## 关键术语

| 术语 | 常说的 | 实际含义 |
|------|--------|----------|
| Routine | “代理的提示” | 系统提示 + 工具列表。定义角色和可用的 handoffs。 |
| Handoff | “转移到另一个代理” | 活动代理可调用的一个工具，返回一个新的 Agent。运行时切换活动代理。 |
| Stateless | “运行间无记忆” | Swarm 不会持久化任何内容；记忆由调用方负责。 |
| Active agent | “当前谁在说话” | 当前持有对话的代理。handoff 会改变这点。 |
| Context transfer | “交接时传递什么” | 进站代理可见的历史策略：完整、最近 N 条或摘要。 |
| Handoff loop | “代理来回传递” | 两个代理不断互相交接的失败模式。 |
| OpenAI Agents SDK | “生产级 Swarm” | 2025 年 3 月的继任者；在 handoff 原语上加入会话、护栏、追踪。 |
| Handoff filter | “对交接的门控” | SDK 的功能，用于在交接边界检查和修改上下文。 |

## 延伸阅读

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 参考性阐述  
- [OpenAI Swarm repo](https://github.com/openai/swarm) — 原始实现，作为概念参考保留  
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 带有会话和追踪的生产继任者  
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — 说明 Claude Code 子代理如何通过 `Task` 使用类似 handoff 的模式