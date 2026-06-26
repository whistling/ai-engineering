# The Autonomous Coding Agent Landscape (2026)

> SWE-bench Verified 在不到三年的时间里从 4% 上升到 80.9%。同样的 Claude Sonnet 4.5 在 SWE-agent v1 上在 SWE-agent v1 上得分 43.2%，在 Cline autonomous 上得分 59.8% —— 围绕模型的脚手架现在与模型本身同等重要。OpenHands（前身 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环直接在沙箱中执行 Python 操作，而不是 JSON 工具调用。头条数字掩盖了一个方法论问题：500 道 SWE-bench Verified 任务中有 161 道只需 1–2 行修改，而 SWE-bench Pro（10+ 行任务）对同样的前沿模型的得分位于 23–59%。

**Type:** 学习  
**Languages:** Python（标准库，CodeAct vs JSON tool-call 比较）  
**Prerequisites:** Phase 14 · 07（工具使用），Phase 15 · 01（长时程代理）  
**Time:** ~45 分钟

## 问题

“哪个编码代理最好”是错误的问题。正确的问题是：在与我的工作相匹配的任务分布上，使用我将在生产中运行的脚手架，端到端的可靠性是多少？

在 2022 到 2026 年间，领域里学到的一点是脚手架——检索层、规划器、沙箱、编辑-验证循环、反馈格式——具有承重作用。Claude Sonnet 4.5 在 SWE-agent v1 上在 SWE-bench Verified 得 43.2%；同一模型在 Cline 的自治脚手架中得 59.8%。绝对差异 16.6 个百分点，相同权重。基础模型只是一个组件；循环才是产品。

配套的问题是基准饱和掩盖了回退。SWE-bench Verified 已接近饱和，简单任务尾部（500 道任务中有 161 道需要 ≤2 行）拉高了高分。现实世界的质量更适合用像 SWE-bench Pro（10+ 行修改）这样的分布来衡量，在那里相同的领先者仍然位于 23–59%。

## 概念

### SWE-bench，一段话概述

SWE-bench（Jimenez 等人）取自真实的 GitHub issue 并带有真实的补丁（ground-truth patches），要求代理生成一个能让测试套件通过的补丁。SWE-bench Verified（OpenAI，2024）是一个人工整理的 500 题子集，已移除模糊和损坏的任务。SWE-bench Pro 是更难的继任者——需要 10+ 行修改的任务，当前前沿代理位于 23–59%。

### 2022 → 2026 曲线实际展示了什么

- 2022：研究模型在原始 SWE-bench 上约 ~4%。
- 2024：GPT-4 + Devin 风格脚手架达到 ~14%；SWE-agent 约 ~12%。
- 2025：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 中推动进入 40–55% 区间。
- 2026：Claude Sonnet 4.5 与前沿竞争对手在 SWE-bench Verified 上达到 70–80%+。Epoch AI 的排行榜实时追踪这些结果。

斜率来自三类复合因素：更好的基础模型、更好的脚手架（CodeAct、反思、验证循环）、以及更好的基准（Verified 移除了噪声）。

### CodeAct vs JSON tool calls

OpenHands（All-Hands-AI，arXiv:2407.16741，前身 OpenDevin）做了一个具体的架构赌注：不是让模型发出由宿主解析并执行的 JSON 工具调用，而是让模型输出 Python 代码，由 Jupyter 风格的内核在沙箱中运行。代理可以在一次动作中遍历文件、串联工具、并在内部捕获自己的异常。

权衡如下：

- JSON tool calls：每次动作为一次回合；易于审计；组合性受限；默认安全因为每个调用都通过显式验证器。
- CodeAct：一个动作可以是整个程序；具备组合性；需要一个强化的沙箱（OpenHands 使用 Docker 隔离）；失败模式包括沙箱运行时允许的任何问题。

两种架构都已投入生产。CodeAct 在开源平台（OpenHands、smolagents）中占主导；JSON tool calls 在托管服务（Anthropic Managed Agents、OpenAI Assistants）中仍占主导，那里提供方控制执行器。

### 2026 年景观中的脚手架

| Scaffold | License | Execution model | Notable property |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | CodeAct in Docker | 最活跃的开源平台；事件流可重放 |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | 第一个端到端的 SWE-bench 脚手架 |
| Aider | Apache-2 | edit-via-diff in local repo | 极简脚手架，回归稳定性强 |
| Cline | Apache-2 | VS Code agent with tool policy | 在 Sonnet 4.5 上得分最高的开源脚手架 |
| Devin (Cognition) | Proprietary | Managed VM + planner | 首个“AI 软件工程师”产品类别 |
| Claude Code | Proprietary | Permission modes + routines | Lesson 10 详述了代理循环 |

### 为什么脚手架占主导

一次编码运行是一个长时程轨迹（Lesson 1）。可靠性会在步骤间复合。脚手架能带来分数的三个方面：

1. 检索：找到要读取的正确文件是沉默的瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引、以及 Aider 的仓库地图都在解决这个问题。
2. 验证循环：运行测试、读取堆栈跟踪并重试，在 SWE-bench 上能带来 10+ 个百分点的差距。
3. 失败遏制：在出错时回滚的沙箱能防止损害累积。同一模型有无验证循环看起来像两个不同的产品。

### 基准饱和与真实分布

OpenHands 的作者和 Epoch AI 都指出 SWE-bench Verified 存在容易任务尾部：500 道题中有 161 道只需要 1–2 行修改。高分部分由这部分尾部驱动。SWE-bench Pro 限制为 10+ 行修改，即使对于前沿系统也返回 23–59% 的分数。你在生产中的任务分布几乎肯定更接近 Pro 而不是 Verified。

为选择代理的含义：用你自己的 bug 积压构造一个类 Pro 的子集来运行。最重要的分数是代表你发布内容的任务上的分数。

## 使用说明

`code/main.py` 比较了两个玩具代理脚手架在固定迷你任务分布上的表现：

1. 一个 **JSON tool-call** 脚手架，每回合执行一次动作。
2. 一个 **CodeAct** 脚手架，每次动作可以输出一小段 Python 代码。

两者都使用一个存根“模型”（确定性规则），以便比较将脚手架与模型质量隔离开来。输出显示 CodeAct 脚手架在更少回合内解决更多任务，但代价是每次操作的破坏范围更大。

## 交付

`outputs/skill-scaffold-audit.md` 可帮助你在采用前审计所提议的编码代理脚手架：检索质量、是否存在验证器、沙箱隔离，以及基准与分布的匹配度。

## 练习

1. 运行 `code/main.py`。每个脚手架在相同任务集上需要多少回合？每次操作的破坏范围各是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文论证在复杂任务上 CodeAct 击败 JSON tool calls。指出论文承认的一个失败模式，并用一句话说明在生产中何时该模式会占主导。

3. 从你的 bug 积压中选一个需要跨两个文件进行 10+ 行修改的任务。估计在以下两种情况下前沿模型的端到端成功概率：（a）JSON tool calls，（b）CodeAct。说明差距的理由。

4. SWE-bench Verified 中有 161 道单文件、1–2 行的任务。构建一个排除它们的得分。排行榜如何重排？

5. 阅读 “Introducing SWE-bench Verified”（OpenAI）。解释用于移除模糊任务的具体方法论，并指出一个该策展可能遗漏的类别。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| SWE-bench | "Coding benchmark" | 真实的 GitHub issue，带有 ground-truth 补丁和测试套件 |
| SWE-bench Verified | "Cleaned subset" | 500 道人工整理的任务，存在容易尾巴 |
| SWE-bench Pro | "Harder subset" | 10+ 行修改；前沿得分位于 23–59% |
| CodeAct | "Code-as-action" | 代理输出 Python；Jupyter 风格内核在沙箱中执行 |
| JSON tool call | "Function calling" | 每次动作是一个结构化的 JSON 有效载荷，执行前会被验证 |
| Scaffold | "Agent framework" | 围绕基础模型的检索 + 规划器 + 执行器 + 验证循环 |
| ACI (Agent-Computer Interface) | "SWE-agent's format" | 为 LLM 设计的命令集，而非面向人的 shell |
| Verifier loop | "Test-and-retry" | 运行测试、读取输出、修订补丁；最大的非模型可靠性提升 |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始基准与方法论。  
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 关于如何构建该人工整理子集的说明。  
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架构与事件流设计。  
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时追踪的得分。  
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时程编码代理可靠性的框架。