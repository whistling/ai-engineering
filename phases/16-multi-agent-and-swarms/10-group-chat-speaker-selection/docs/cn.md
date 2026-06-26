# 群聊与发言者选择

> AutoGen GroupChat 和 AG2 GroupChat 在 N 个代理之间共享同一个对话池；一个选择器函数（LLM、轮询，或自定义）决定下一个发言者。 这是涌现式多智能体会话的典型——代理并不知道自己在静态图中的角色，它们只是对共享池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中被保留；AutoGen v0.4 则重写为事件驱动的 actor 模型。微软在 2026 年 2 月将 AutoGen 置于维护模式，并与 Semantic Kernel 合并为 Microsoft Agent Framework（RC 2026 年 2 月）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中都保留——学会一次，到处可用。

**Type:** 学习 + 构建  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 16 · 04（原语模型）  
**Time:** ~60 分钟

## 问题

当工作流已知时，静态图（例如 LangGraph）很棒。真实会话不是静态的：有时编码者问审阅者，有时问研究员，有时问撰稿人。将每一次可能的交接硬编码会导致边数量爆炸。你希望实现“代理对共享池做出反应”，由某个函数决定谁接下来发言。

这正是 AutoGen GroupChat 所做的。

## 概念

### 结构示意

```
              ┌─── 共享池 ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │（所有人读取全部）
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  “下一个发言者 = C”
```

每个代理都能看到每条消息。每个回合都会调用一个选择器函数来决定下一个发言者。

### 三种选择器风格

- **轮询（Round-robin）。** 固定循环。确定性。随 N 线性扩展，但忽略上下文——即便话题是法律审查，也会轮到编码者发言。

- **基于 LLM 的选择。** 调用 LLM，读取最近的消息池并返回最合适的下一个发言者。上下文感知但较慢：每个回合都会增加一次 LLM 调用。AutoGen 的默认方式。

- **自定义。** 一个可按你想要逻辑实现的 Python 函数。典型用法：基于 LLM 的选择加上回退规则（例如“在编码者之后总是给验证者一次机会”）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当一个代理完成一次发言后，管理器调用选择器并返回下一个代理。循环继续，直到满足终止条件。

### 终止

三种常见模式：

- **最大回合数（Max rounds）。** 对总轮次设置硬上限。
- **“TERMINATE” 令牌。** 代理可以发出哨兵消息；当管理器看到该消息时停止对话。
- **目标达成检查（Goal-reached check）。** 每回合运行轻量的验证器，在完成时停止会话。

### AutoGen → AG2 的分裂与 Microsoft Agent Framework 的合并

在 2025 年初，微软开始对 AutoGen（v0.4）进行重大重写，采用事件驱动的 actor 模型。社区将 AutoGen v0.2 的 GroupChat 语义分叉为 AG2，保留了早期采用者已集成的 API。

在 2026 年 2 月，微软宣布 AutoGen 进入维护模式，并将事件驱动的 actor 模型合并到 **Microsoft Agent Framework**（RC 2026 年 2 月，现在已与 Semantic Kernel 合并）。GroupChat 概念在两条路线中都继续存在，但实现细节不同。对 v0.2 兼容代码，AG2 是优选的上游。

### 何时适合使用 GroupChat

- **涌现式会话。** 你不想为每一个可能的下一个发言者预先连线。
- **角色混合任务。** 编码者问研究员，研究员问档案员，档案员又问回编码者。流程不是有向无环图。
- **探索性问题解决。** 想象“头脑风暴会议”，而不是“装配线”。

### 何时不适合

- **严格确定性需求。** LLM 选择器可能不一致。相同的提示，不同运行可能选出不同发言者。
- **拍马屁级联（Sycophancy cascades）。** 代理倾向于服从最自信的发言者。需要通过反提示明确对抗。
- **上下文膨胀（Context bloat）。** 每个代理都会读取所有消息；10 轮后上下文会非常大。使用投影（Lesson 15）来限制视图。
- **“火热”发言者（Hot speakers）。** 一个代理占据主导，因为选择器偏好它的专长。把发言平衡作为选择器特性引入。

### 群聊 vs 监督者（supervisor）

相同的原语，不同的默认设置：

- 监督者（Supervisor）：一个代理负责规划，其他代理执行。选择器的行为是“询问规划者下一步做什么”。
- 群聊（Group chat）：所有代理为对等关系；选择器基于共享池的函数做出决定。

两者都使用 Lesson 04 中的四个原语。群聊默认采用 LLM 选择式编排并使用全池共享状态。

## 构建实现

`code/main.py` 在标准库中从零实现了一个 GroupChat。包含三个代理（coder、reviewer、manager），提供轮询和基于 LLM 的选择器变体，并在检测到 `TERMINATE` 令牌时终止。

演示会打印对话记录以及两种变体下选择器的决策轨迹。

运行：

```
python3 code/main.py
```

## 使用方法

`outputs/skill-groupchat-selector.md` 为给定任务配置一个 GroupChat 选择器 —— 轮询、基于 LLM 的选择，或自定义；以及选择器应使用哪些输入（最近消息、代理专长、发言计数等）。

## 投产清单

- **最大回合上限。** 必须始终设置。典型任务为 10–20 轮。
- **发言者平衡度量。** 跟踪每个代理的发言次数；当不平衡超过阈值时发出警报。
- **终止令牌。** `TERMINATE` 或使用专门的验证代理。
- **投影或作用域化记忆。** 约 10 条消息后，考虑仅给每个代理作用域化的视图以防上下文膨胀。
- **选择器日志。** 对于基于 LLM 的选择器，记录选择器的输入和其选择。否则调试几乎不可能。

## 练习

1. 运行 `code/main.py`。比较轮询与基于 LLM 的选择下的对话。哪种情况下哪个代理占主导？
2. 在选择器中加入“每代理最大发言次数”的规则。它如何影响记录？
3. 实现目标达成终止：当 reviewer 返回 “approved” 时停止。它在达到回合上限前多频繁触发？
4. 阅读 AutoGen 关于 GroupChat 的稳定文档（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。识别 `GroupChatManager` 使用的默认选择器。
5. 浏览 AG2 仓库（https://github.com/ag2ai/ag2）并比较其 v0.2 的 GroupChat 与 v0.4 的事件驱动版本。v0.4 在什么具体属性上（吞吐量、容错性、可组合性）有所增强？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| GroupChat | "Agents in one chat room" | 共享消息池 + 选择器函数。AutoGen / AG2 的原语。 |
| Speaker selection | "Who talks next" | 选择下一个代理的函数。轮询、基于 LLM 的选择，或自定义。 |
| GroupChatManager | "The meeting host" | AutoGen 组件，拥有选择器并在回合间循环。 |
| ConversableAgent | "The base agent" | AutoGen 基类；能够发送和接收消息的代理。 |
| Termination token | "The 'stop' word" | 哨兵字符串（通常为 `TERMINATE`），用于结束会话。 |
| Hot speaker | "One agent dominates" | 故障模式：选择器持续挑同一个代理。 |
| Context bloat | "Pool grows unbounded" | 每个代理读取所有历史消息；上下文随回合数增长。 |
| Projection | "Scoped view" | 针对角色的池视图，以防止上下文膨胀。 |

## 延伸阅读

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现  
- [AG2 repo](https://github.com/ag2ai/ag2) — 社区延续的 AutoGen v0.2  
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/) — 合并后的继任者，RC 2026 年 2 月  
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — 事件驱动 actor 模型重写的详细说明