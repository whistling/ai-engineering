# Self-Refine and CRITIC: 迭代输出改进

> Self-Refine（Madaan 等，2023）在一个 LLM 上实现三种角色——generate、feedback、refine——并在循环中运行。平均增益：在 7 个任务上绝对提高约 +20。CRITIC（Gou 等，2023）通过将验证步骤路由到外部工具来强化反馈步骤。在 2026 年，这一模式出现在每个框架中，被称为 “evaluator-optimizer”（Anthropic）或输出护栏循环（OpenAI Agents SDK）。

**Type:** 构建  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 14 · 01（Agent Loop），Phase 14 · 03（Reflexion）  
**Time:** ~60 分钟

## 学习目标

- 陈述 Self-Refine 的三条提示（generate、feedback、refine），并解释为何历史记录对 refine 提示很重要。  
- 解释 CRITIC 的关键洞见：在没有外部落地的情况下，LLM 在自我验证方面不可靠。  
- 用标准库实现带历史记录且可选外部验证器的 Self-Refine 循环。  
- 将此模式映射到 Anthropic 的 “evaluator-optimizer” 工作流以及 OpenAI Agents SDK 的输出护栏。

## 问题

一个智能体给出一个几乎正确的答案。也许一行代码有语法错误；也许摘要太长；也许计划遗漏了一个边界情况。你想要的是：智能体自我批评其输出，然后修复它。

Self-Refine 表明这可以用单一模型实现，不需要训练数据或强化学习。但有一个问题：LLM 在难事实上的自我验证很差。CRITIC 指出了解决方法——将验证步骤通过外部工具（搜索、代码解释器、计算器、测试运行器）路由出去。

这两篇论文共同定义了 2026 年的迭代改进默认模式：generate -> verify（尽可能使用外部工具）-> refine，当验证器通过时停止。

## 概念

### Self-Refine（Madaan 等，NeurIPS 2023）

一个 LLM，三种角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 会看到完整历史 — 所有先前的输出和批评 — 因此不会重复犯错。论文做了消融实验：去掉历史后质量急剧下降。

要点：在包括 GPT-4 的 7 个任务（数学、代码、缩写、对话）上平均提升约 +20。无需训练、无需外部工具、单模型。

### CRITIC（Gou 等，arXiv:2305.11738，v4 2024 年 2 月）

Self-Refine 的弱点：反馈步骤实际上是模型对自身打分。对于事实性断言这不可靠（幻觉通常对产生它的模型看起来很有说服力）。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：

- 用于事实性断言的搜索引擎。  
- 用于代码正确性的代码解释器。  
- 用于算术的计算器。  
- 特定领域的验证器（单元测试、类型检查器、linter）。

验证器生成基于工具结果的结构化批评。然后 refiner 在此批评的条件下进行重写。

要点：CRITIC 在事实性任务上优于 Self-Refine，因为批评是有落地依据的。在没有外部验证器的任务（创意写作、格式化）上，CRITIC 会退化为 Self-Refine。

### 停止条件

两种常见形式：

1. Verifier 通过。外部测试返回成功。可用时优先（单元测试、类型检查器、护栏断言）。  
2. 没有反馈。模型说“输出没问题”。更便宜但不可靠；与最大迭代上限配合使用。

2026 年默认：结合两者。“如果验证器通过或模型说没问题且迭代次数 >= 2，或迭代次数 >= max_iterations 则停止。”

### Evaluator-Optimizer（Anthropic，2024）

Anthropic 在 2024 年 12 月的文章将此称为五类工作流模式之一。两个角色：

- Evaluator：对输出评分并生成批评。  
- Optimizer：根据批评修订输出。

循环直到 evaluator 通过。这就是 Anthropic 框架下的 Self-Refine/CRITIC。Anthropic 添加的关键工程细节：evaluator 与 optimizer 的提示应当在结构上明显不同，防止模型只是自我盖章。

### OpenAI Agents SDK 的输出护栏

OpenAI Agents SDK 将此模式作为“输出护栏”发布。护栏是运行在智能体最终输出上的验证器。如果护栏触发（抛出 `OutputGuardrailTripwireTriggered`），则该输出被拒绝，智能体可以重试。护栏可以调用工具（CRITIC 风格）或是纯函数（Self-Refine 风格）。

### 2026 年的陷阱

- 橡皮章循环（Rubber-stamp loops）。同一模型用相同提示做生成与批评会趋于“看起来不错”。使用结构不同的提示，或用更小的廉价模型做批评。  
- 过度精炼。每次 refine 都增加延迟和 token 成本。预算 1–3 次；超过后升级为人工复审。  
- 在琐碎任务上使用 CRITIC。若无外部验证器，CRITIC 会退化为 Self-Refine；不要为空洞的验证器支付额外延迟。

## 实现

`code/main.py` 实现了在一个玩具任务上的 Self-Refine 和 CRITIC：给定一个主题生成一个简短的要点列表。验证器检查格式（3 个要点，每项少于 60 个字符）。CRITIC 添加了一个外部“事实验证器”，会惩罚已知的幻觉。

组件：

- `generate` — 脚本化的生成器。  
- `feedback` — 类 LLM 的自我批评。  
- `verify_external` — CRITIC 风格的有落地依据的验证器。  
- `refine` — 根据历史重写输出。  
- 停止条件 — 验证器通过或最大 4 次迭代。

运行方式：

```
python3 code/main.py
```

比较 Self-Refine 与 CRITIC 的运行。CRITIC 捕获了 Self-Refine 未能发现的事实性错误，因为外部验证器具有落地依据，而自我批评没有。

## 使用场景

Anthropic 的 evaluator-optimizer 在 Claude 友好的语言中呈现了该模式。OpenAI Agents SDK 的输出护栏呈现了 CRITIC 的形态（护栏可以调用工具）。LangGraph 提供了一个类似 Self-Refine 的反思节点。Google 的 Gemini 2.5 Computer Use 在每一步增加了一个安全评估器，这是 CRITIC 的变体：每个动作在提交前都被验证。

## 部署

`outputs/skill-refine-loop.md` 根据任务形态、验证器可用性和迭代预算配置了一个 evaluator-optimizer 循环。该文件生成 generator、evaluator/verifier 和 optimizer 的提示，以及停止策略。

## 练习

1. 将玩具程序的 max_iterations 设置为 1。CRITIC 仍然有帮助吗？  
2. 将外部验证器替换为一个嘈杂的验证器（30% 的假阳性率）。循环会怎样表现？这是 2026 年大多数护栏堆栈的现实。  
3. 实现 “不同模型的 generator-critic” 变体：大模型生成，小模型批评。它会跑赢同一模型吗？  
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。列出三类验证工具并为每类给出一个示例。  
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的 verifier 角色。该 SDK 做对了什么，有什么不足？

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|----------:|---------|
| Self-Refine | “LLM 自我修正” | 在一个模型中实现的 Generate -> feedback -> refine 循环，带历史 |
| CRITIC | “工具落地的验证” | 用外部验证器（搜索、代码、计算、测试）替换 feedback |
| Evaluator-Optimizer | “Anthropic 的工作流模式” | 两个角色——evaluator 评分、optimizer 修订——循环至收敛 |
| Output guardrail | “事后检查” | OpenAI Agents SDK 中在智能体生成输出后运行的验证器 |
| Verify step | “批评阶段” | 承重的决策：是基于落地的验证还是模型自评 |
| Refine history | “模型之前尝试过的内容” | 在 refine 提示中预置的先前输出与批评；去掉后质量崩溃 |
| Rubber-stamp loop | “自我同意失败” | 同一提示的批评返回“看起来不错”；用结构不同的提示修复 |
| Stop condition | “收敛测试” | 验证器通过 或 无反馈 且 迭代上限；不要仅依赖单一条件 |

## 延伸阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 经典论文  
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — 工具落地的验证  
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — evaluator-optimizer 工作流模式  
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 将输出护栏视为 CRITIC 形态的验证器