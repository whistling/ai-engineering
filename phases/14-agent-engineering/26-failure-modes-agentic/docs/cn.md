# 失败模式：为什么智能体会失效

> MASFT（伯克利，2025）将 14 种多智能体失败模式归类为 3 大类。微软的分类法记录了现有 AI 失败在具代理性环境中的放大方式。业界现场数据在五种反复出现的模式上达成一致：幻觉操作、范围蔓延、级联错误、上下文丢失、工具误用。

**Type:** 学习 + 构建  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 14 · 05（Self-Refine 和 CRITIC），Phase 14 · 24（可观测性）  
**Time:** ~60 分钟

## 学习目标

- 能说出 MASFT 的三类失败和每类至少四种具体模式。
- 解释为什么在具代理性环境中失败会放大现有的 AI 失败（偏见、幻觉）。
- 描述五种业界常见模式及其缓解措施。
- 实现一个 stdlib 检测器，为智能体跟踪打上失败模式标签。

## 问题

团队部署的智能体在 90% 的 trace 中工作正常。那 10% 的失败并非随机噪声 —— 它们落入少数反复出现的类别。一旦你能为这些失败命名，就可以针对性地监控和修复它们。

## 概念

### MASFT（伯克利，arXiv:2503.13657）

多智能体系统失败分类法（Multi-Agent System Failure Taxonomy）。把 14 种失败模式聚成 3 大类。注释者间 Cohen's Kappa 为 0.88 —— 这些类别可可靠区分。

核心观点：这些失败是多智能体系统的根本设计缺陷，而不是可以靠更好基础模型修复的 LLM 限制。

### 微软：具代理性 AI 系统的失败模式分类法

- 现有的 AI 失败（偏见、幻觉、数据泄露）在具代理性的环境中被放大。
- 自主性带来新的失败：大规模的非预期动作、工具误用、任务漂移。
- 该白皮书是具代理性产品的风险登记簿。

### 在具代理性 AI 中描述故障（arXiv:2603.06847）

- 失败源自编排、内部状态演化和与环境的交互。
- 并非仅仅是“坏代码”或“模型输出错误”。

### LLM 智能体幻觉调查（arXiv:2509.18970）

两种主要表现：

1. **偏离指令执行** — 智能体未遵循系统提示。
2. **长程上下文误用** — 智能体忘记或错误应用早期回合的上下文。

子意图错误（Sub-intention errors）：遗漏（Omission，遗漏步骤）、冗余（Redundancy，重复步骤）、顺序错乱（Disorder，步骤顺序错误）。

### 五种业界反复出现的模式

Arize、Galileo、NimbleBrain 在 2024–2026 年的现场分析一致指出：

1. **幻觉操作（Hallucinated actions）。** 智能体调用不存在的工具或伪造参数。
2. **范围蔓延（Scope creep）。** 智能体将任务扩展到用户请求之外（创建额外的 PR、发送额外邮件）。
3. **级联错误（Cascading errors）。** 一次错误调用触发下游连锁效应。一次虚构的 SKU 幻觉触发四次 API 调用 —— 导致跨系统事件。
4. **上下文丢失（Context loss）。** 长周期任务忘记早期回合的约束。
5. **工具误用（Tool misuse）。** 调用了正确的工具但参数错误，或完全调用了错误的工具。

级联错误是致命的。智能体往往无法区分“我失败了”和“任务不可能完成”，并经常在收到 400 错误时编造成功消息来闭环。

### 缓解：在每一步设置门控

在推理链的每一步设置自动化验证门，检查是否与环境状态事实相符。具体包括：

- 每步安全分类器（Lesson 21）。
- 工具调用参数校验（Lesson 06）。
- 将检索到的内容与已知事实交叉校验（Lesson 05，CRITIC）。
- 通过重新探测状态检测成功幻觉（文件真的创建了吗？）。

### 失败监控常出的问题

- **只标注崩溃。** 大多数智能体失败会产生看似有效的输出，需要内容级的检查。
- **没有基线。** 漂移检测需要一个最后已知良好点；没有它就无法判断“这是否在变糟”。
- **过度告警。** 每次失败都会产生一次告警。需要聚类并限速告警。

## 构建

`code/main.py` 实现了一个 stdlib 失败模式标注器：

- 一个覆盖五种模式的合成 trace 数据集。
- 每种模式的检测函数（基于工具调用、输出、重复动作的签名模式）。
- 一个为每个 trace 打标签并报告模式分布的标注器。

运行：

```
python3 code/main.py
```

输出：逐条 trace 的标签 + 聚合分布，粗略复现 Phoenix 的 trace 聚类所揭示的内容。

## 使用方法

- 在生产中用于漂移聚类的 **Phoenix**（Lesson 24）。
- 用于会话回放与注释的 **Langfuse**。
- 自定义：针对你的可观测性平台无法检测的领域特定签名。

## 部署

`outputs/skill-failure-detector.md` 生成针对你领域定制的失败模式检测器，并接入一个 trace 存储。

## 练习

1. 为“成功幻觉”添加一个检测器：智能体返回成功但目标状态未发生变化。
2. 从你构建的产品中标注 100 条真实 trace。哪种模式占主导？修复它的成本是多少？
3. 实现一个“级联半径”指标：给定在第 N 步发生的失败，它影响了多少下游步骤？
4. 阅读 MASFT 的 14 种失败模式。选出三种适用于你的产品并为其编写检测器。
5. 将一个检测器接入 CI 作业：若被标注的某种模式 >=5%，则构建失败。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MASFT | "Multi-agent failure taxonomy" | 伯克利的 14 模式分类 |
| Cascading error | "Ripple failure" | 一个早期错误在 N 步中传播 |
| Context loss | "Forgot the constraint" | 长周期回合丢失早期回合的事实 |
| Tool misuse | "Wrong tool / wrong args" | 有效的调用，但调用不正确 |
| Success hallucination | "Faked completion" | 智能体在收到 400 时宣称成功；状态未改变 |
| Scope creep | "Overreach" | 智能体做了超出要求的事情 |
| Instruction-following deviation | "Disobedience" | 忽视系统提示或用户约束 |
| Sub-intention errors | "Plan bugs" | 执行计划时的遗漏、冗余、顺序错乱 |

## 延伸阅读

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 种失败模式，3 大类  
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — 风险登记簿  
- [Arize Phoenix](https://docs.arize.com/phoenix) — 实践中的漂移聚类  
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 有时更简单的模式可以完全避免某些失败模式