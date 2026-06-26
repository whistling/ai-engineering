# 角色专精 — 规划者、批评者、执行者、验证者

> 到 2026 年最常见的多智能体分工：一个智能体负责规划，一个执行，一个批评或验证。MetaGPT（arXiv:2308.00352）将其形式化为以角色提示词编码的 SOP：产品经理、架构师、项目经理、工程师、QA 工程师——遵循 `Code = SOP(Team)`。ChatDev（arXiv:2307.07924）通过“聊天链”把设计者、程序员、审阅者、测试者串起来，并引入“可通信去错觉”（agents 在缺失细节时显式请求）。验证者承担承重作用：Cemri 等人在 MAST（arXiv:2503.13657）中表明，每一次多智能体失败都能追溯到缺失或损坏的验证。PwC 报告称在 CrewAI 中通过结构化验证循环将准确率从 10% 提升到 70%（7× 增益）。

**Type:** 学习 + 构建  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)  
**Time:** ~60 分钟

## 问题

通用的多智能体系统会产出通用的结果。三名程序员在群聊中各写出三种风格的同样平庸代码。你可以增加更多智能体、增加更多轮次，但仍无法越过质量门槛。

解决办法不是更多智能体——而是“不同”的智能体。分配不同的角色。给批评者的工具不要给规划者。给验证者一个客观的测试套件。现在系统内部有带证据的分歧和修正，而不仅仅是并行猜测。

## 概念

### 四个经典角色

**Planner.** 阅读目标，产出步骤列表或规格说明。工具：知识检索、文档。输出：结构化计划。

**Executor.** 按照计划逐步读取并产生产出物。工具：实际的工作工具（代码编译器、shell、API 客户端）。输出：产出物（如代码）。

**Critic.** 根据规划者的意图审阅执行者的输出。工具：只读访问产出物、静态分析。输出：接受/拒绝，并给出理由。

**Verifier.** 读取产出物并运行确定性检查。工具：测试运行器、类型检查器、模式校验器。输出：通过/不通过，并给出证据。

Critic 是主观、有观点的，通常基于 LLM；Verifier 是客观、确定性的，通常基于代码。两者并不相同。

### MetaGPT 的 SOP 模式

MetaGPT（arXiv:2308.00352）将软件工程 SOP 编码为角色提示词：

- **Product Manager** 撰写 PRD（产品需求文档）。
- **Architect** 产出系统设计。
- **Project Manager** 拆分任务。
- **Engineer** 实现功能。
- **QA Engineer** 运行测试。

每个角色都有严格的输入/输出模式。角色提示词明确说明该角色是什么以及必须产出什么。`Code = SOP(Team)` 的表达意味着：确定性的 SOP 将一组 LLM 转换为可预测的流水线。

### ChatDev 的可通信去错觉

ChatDev 引入了一个关键动作：当执行者需要计划中未给出的具体细节时，它会在继续之前显式向设计者询问。这防止了经典的 LLM 错误：凭空编造看起来合理的细节。

实现方式：角色提示词包含“当你需要未被提供的具体信息时，显式向相关角色按名询问，然后再输出结果”。

### 为什么验证者最重要

Cemri 等人在 MAST 中追踪了 1642 次多智能体执行失败。21.3% 的失败源于验证缺失——系统交付了未经任何检查的答案。其余 79% 通常可以追溯到“存在一个检查，但它悄然失败或从未被运行”。验证是承重的角色。

PwC 报告（CrewAI 部署，2025 年）显示，加入结构化验证循环将准确率从 10% 提升到 70%。单靠一个角色就带来了 7× 的提升。

### 批评者 vs 验证者

- 批评者是用 LLM 审阅产出物的角色。主观、有观点。可能被看起来合理的文字迷惑。
- 验证者是对产出物运行的确定性程序。客观。给出通过/不通过并附带证据。

二者都要使用。批评者能发现验证者无法明确表述的品味或设计问题；验证者能发现批评者因只是从表面判断而察觉不到的运行时缺陷。

### 反模式

系统中所有角色都是 LLM，且每个角色的输出都是“看起来不错”。这是典型的 MAST 失败模式。至少加入一个由代码决定通过/不通过的验证者，绝不要全是 LLM。

### 框架映射

- **CrewAI** — `Agent(role, goal, backstory)` 是教科书式的角色专精表面。
- **LangGraph** — 节点可以有专门化的提示词；边强制流水线化交接。
- **AutoGen** — 基于角色的 ConversableAgents，在 GroupChat 里用单词命名。
- **OpenAI Agents SDK** — 在角色专精的 Agents 之间手动接力工具。

## 实现

`code/main.py` 实现了一个四角色流水线来构建一个简单的 Python 函数：

- **Planner** 产出规格。
- **Executor** 生成代码字符串。
- **Critic**（LLM 模拟）标记明显的问题。
- **Verifier** 在沙箱中使用 `exec` 对生成的代码运行测试用例。

演示运行两次：一次执行者生成正确代码（批评者 + 验证者均通过），一次执行者生成不符合规格的代码（批评者因为代码看起来合理而遗漏了错误，验证者通过测试失败捕获了该错误）。

运行命令：

```
python3 code/main.py
```

## 使用方法

`outputs/skill-role-designer.md` 接受一个任务并生成角色名单（3-5 个角色）、每个角色的输入/输出模式，以及验证者检查项。在将智能体接入框架之前先使用此文件。

## 部署清单

- **至少一个确定性的验证者。** 绝不全部由 LLM 决定。
- **每个角色有明确的 I/O 模式。** 规划者返回的是规格而不是随意的文字；执行者按该模式读取。
- **可通信去错觉。** 执行者在信息缺失时必须向规划者询问；不得自行臆造。
- **批评者/验证者顺序。** 先运行批评者（廉价，能抓设计问题），再运行验证者（慢但能抓运行时缺陷）。
- **循环预算。** 在升级给人工之前，最多允许 2 次批评者-执行者的修订轮次。

## 练习

1. 运行 `code/main.py`，观察验证者如何捕获批评者遗漏的错误。添加一个静态分析检查（统计 `return` 出现次数）作为额外的验证器。它能捕获运行时测试未发现的哪些问题？
2. 添加第五个角色：“需求分析员”，将用户诉求翻译成对规划者友好的规格。哪些可通信去错觉的请求应该上报给他们？
3. 阅读 MetaGPT 第 3 节（“Agents”）。列出 MetaGPT 五个角色的输入/输出模式。
4. 阅读 ChatDev 的聊天链图（arXiv:2307.07924 图 3）。指出可通信去错觉在何处打破了原本可能无限循环的环路。
5. PwC 报告的 7× 准确率提升来自验证循环。假设三类任务，在这些任务中加入验证者不会有帮助——为何确定性检查在这些场景中不可行或代价太高？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Role specialization | "Different agents, different jobs" | 为 planner/executor/critic/verifier 等角色定制的不同系统提示词。 |
| SOP pattern | "Encoded standard operating procedure" | MetaGPT 的表述：每个角色的严格 I/O 模式将团队转成流水线。 |
| Communicative dehallucination | "Ask before inventing" | ChatDev 模式：执行者在细节缺失时向规划者询问，而不是自作主张。 |
| Critic | "LLM reviewer" | 主观、有观点的审阅者。能抓住品味类问题，但可能被看起来合理的文字蒙蔽。 |
| Verifier | "Deterministic check" | 基于代码的通过/不通过判断。测试运行器、类型检查器、模式校验器。不会被蒙蔽。 |
| Verification gap | "No one checked" | MAST 中 21.3% 的失败原因。产出在未被检测的情况下交付，若检测了会被发现错误。 |
| Revision loop | "Critic sends it back" | 批评者拒绝会触发执行者基于反馈的重跑。需要限制预算。 |
| All-LLM anti-pattern | "Looks good to me" | 所有角色都是 LLM，且没有确定性检查。典型的 MAST 失败模式。 |

## 延伸阅读

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — 将 SOP 作为角色提示词的参考论文  
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — 聊天链 + 可通信去错觉  
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分类法；验证缺口占 21.3% 的失败  
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction) — 生产级的角色规范表面