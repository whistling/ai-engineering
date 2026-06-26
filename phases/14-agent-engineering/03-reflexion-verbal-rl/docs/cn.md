# Reflexion: 语言强化学习（Verbal Reinforcement Learning）

> 基于梯度的强化学习需要成千上万次试验和一个 GPU 集群来修复一个失败模式。Reflexion（Shinn 等，NeurIPS 2023）用自然语言做到这一点：每次失败试验后，智能体写下一条反思，将其存储在情节记忆中，并在下一次尝试时以该记忆为条件。这个模式就是 Letta 的睡眠时计算、Claude Code 的 CLAUDE.md 学习条目和 pro-workflow 的 learn-rule 的背后思路。

**Type:** 构建  
**Languages:** Python（标准库）  
**Prerequisites:** 阶段 14 · 01（Agent Loop），阶段 14 · 02（ReWOO）  
**Time:** ~60 分钟

## 学习目标

- 说出 Reflexion 的三大组成部分（Actor、Evaluator、Self-Reflector）以及情节记忆的作用。  
- 用标准库实现一个 Reflexion 循环，包含二值评估器、反思缓冲区和重新尝试的机制。  
- 在标量、启发式和自我评估的反馈来源之间为给定任务作出选择。  
- 解释为什么语言化的强化（verbal reinforcement）能捕捉到基于梯度的强化学习需要成千上万次试验才能修复的错误。

## 问题描述

智能体在任务中失败。标准的强化学习会再运行成千上万次试验，计算梯度并更新权重。既昂贵又缓慢，而且大多数生产级智能体并没有为每次失败分配训练预算。

Reflexion（Shinn 等，arXiv:2303.11366）提出了一个不同的问题：如果智能体只是思考为什么会失败，并在下一次尝试时把这种思考放到提示词里会怎样？不更新权重。不计算梯度。只是在试验之间通过自然语言进行存储。

结果：在 ALFWorld 上它击败了 ReAct 和其他未微调的基线；在 HotpotQA 上它比 ReAct 有所提升；在代码生成（HumanEval/MBPP）上在当时达到了最先进水平。且没有任何一次梯度更新。

## 概念

### 三个组成部分

```
Actor         : 生成轨迹（ReAct 风格循环）
Evaluator     : 对轨迹进行评分 — 二值、启发式或自我评估
Self-Reflector: 用自然语言写出关于失败的反思
```

外加一个数据结构：

```
Episodic memory: 以往反思的列表，在下一次试验的提示中被置于最前
```

一次试验运行 Actor。Evaluator 对其打分。如果分数低，Self-Reflector 生成一条反思（例如：“我选择了错误的工具，因为我把问题误读为关于 X，而它实际上是在问 Y”）。该反思被加入到情节记忆。下一次试验从头开始，但可以看到这些反思。

### 三种评估器类型

1. **Scalar** — 外部的二值信号。ALFWorld 成功或失败。HumanEval 测试通过或不通过。最简单、信号最强。  
2. **Heuristic** — 预定义的失败特征。“如果智能体连续两次产生相同动作，标记为卡住。” “如果轨迹超过 50 步，标记为低效。”  
3. **Self-evaluated** — LLM 对自己的轨迹进行评分。在没有地面实况可用时需要采用。信号较弱；适合与基于工具的验证（Lesson 05 — CRITIC）配合使用。

到 2026 年的默认组合是：有标量信号时优先使用标量；没有时使用自评；启发式作为安全护栏。

### 为什么这能泛化

Reflexion 与其说是新的算法，不如说是一个有名的模式。几乎每个生产级“自我修复”智能体都运行某种变体：

- Letta 的睡眠时计算（Lesson 08）：一个独立的智能体对过去的对话进行反思并写入记忆块。  
- Claude Code 的 `CLAUDE.md` / “保存记忆”模式：将反思作为学习条目捕获，并置于后续会话的前端。  
- pro-workflow 的 `/learn-rule` 命令：将修正捕获为显式规则。  
- LangGraph 的反思节点：对输出评分，并在需要时路由到精炼流程。

这些都源于同一洞见：自然语言是足够丰富的介质，可以在运行间承载“我从失败中学到了什么”。

### 适用场景与限制

Reflexion 在以下情况有效：

- 有明确的失败信号（测试失败、工具报错、错误答案）。  
- 任务类别可复现（同类问题可以再次提出）。  
- 反思有改进轨迹的空间（有足够的动作预算）。

Reflexion 在以下情况无效：

- 智能体第一次尝试就成功。  
- 失败是外部因素（网络中断、工具故障）——对“网络断了”的反思并不能帮助之后的运行。  
- 反思演变为迷信——对一次偶发的失败构建的叙述被存储下来并误导后续运行。

2026 年的陷阱：记忆腐败（memory rot）。反思会累积；有些会过时或错误；随着情节缓冲区增长，重跑变慢。缓解措施：定期压缩（Lesson 06）、对反思设置 TTL，或使用独立的睡眠时清理智能体（Letta）。

```figure
react-trace
```

## 构建它

`code/main.py` 在一个玩具谜题上实现了 Reflexion：输出一个 3 元素列表，其和等于目标值。Actor 发出候选列表；Evaluator 检查和；Self-Reflector 写出一行关于失败的诊断。反思被放入情节记忆，用于下一次试验。

组件：

- `Actor` — 一个脚本化策略，看到反思后会改进。  
- `Evaluator.binary()` — 基于目标和的通过/失败判断。  
- `SelfReflector` — 生成一行失败诊断。  
- `EpisodicMemory` — 带 TTL 语义的有界列表。

运行它：

```
python3 code/main.py
```

跟踪输出会显示三次试验。试验 1 失败，存入一条反思；试验 2 能看到反思并改进但仍失败；试验 3 成功。与基线运行（无反思）对比——它会停留在试验 1 的答案上不能改进。

## 使用它

LangGraph 将反思作为一个节点模式发布。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 将情节缓冲区外部化为一个 markdown 文件。Letta 的睡眠时计算在离线时运行 Self-Reflector，从而使主智能体保持低延迟。OpenAI Agents SDK 并没有直接提供 Reflexion；你可以通过自定义 Guardrail（按得分拒绝轨迹）以及跨运行持久化的 memory `Session` 来构建它。

## 部署它

`outputs/skill-reflexion-buffer.md` 创建并维护一个带有反思捕捉、TTL 和去重逻辑的情节缓冲区。给定一个任务类和一次失败，它会生成一条真正能帮助下一次试验的反思（而不是泛泛的“要更小心”）。

## 练习

1. 将二值评估器切换为返回距离度量（距离目标有多远）的标量评估。收敛速度会更快吗？  
2. 给反思添加 10 次试验的 TTL。过期之后旧的反思是有害还是有益？  
3. 实现启发式评估器：如果动作重复出现则标记为卡住。它与 Self-Reflector 会如何交互？  
4. 用一个无视反思的对抗性 Actor 运行 Reflexion。要迫使 Actor 注意到反思，最小的提示词工程是什么？  
5. 阅读 Reflexion 论文在 AlfWorld 的第 4 节。概念性地重现那 130% 成功率提升：与原始 ReAct 的关键差异是什么？

## 术语表

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| Reflexion | “自我校正” | Shinn 等 2023 — 包含 Actor、Evaluator、Self-Reflector 以及情节记忆 |
| 语言化强化（Verbal reinforcement） | “无需梯度的学习” | 将自然语言反思置于下一次试验的提示前端 |
| 情节记忆（Episodic memory） | “每任务的反思” | 针对某一任务类别的有界反思缓冲区 |
| 标量评估器（Scalar evaluator） | “二值成功信号” | 基于地面实况的通过/失败或数值分数 |
| 启发式评估器（Heuristic evaluator） | “基于模式的检测器” | 预定义的失败特征（例如卡住循环、步数过多） |
| 自我评估（Self-evaluator） | “LLM 自评其轨迹” | 在没有地面真值时的低信号备选——建议与基于工具的验证配合使用 |
| 记忆腐败（Memory rot） | “陈旧的反思” | 情节缓冲区充满过时条目；可通过压缩/TTL 修复 |
| 睡眠时反思（Sleep-time reflection） | “异步自我反思” | 在主路径之外运行 Self-Reflector，使主智能体保持响应速度 |

## 延伸阅读

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — 权威论文  
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — 生产环境中的异步反思实践  
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 将情节缓冲区作为上下文管理的一部分  
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 反思节点模式概览