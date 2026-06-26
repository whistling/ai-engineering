# 可扩展监督与弱到强泛化

> Burns 等人（OpenAI Superalignment，《Weak-to-Strong Generalization》，2023）提出了一个针对超对齐问题的代理：使用较弱模型产生的标签来微调强模型。如果强模型能够从不完美的弱监督中正确泛化，当前面向人类规模的对齐方法可能可扩展到超人类系统。可扩展监督与 W2SG（弱到强泛化）是互补的。可扩展监督（辩论、递归奖励建模、任务分解）提高监督者的有效能力，使其能跟上被监督模型的能力增长。W2SG 确保强模型能从监督者所提供的任何不完美监督信号中正确泛化。Lang 等人的 Debate Helps W2SG（arXiv:2501.13124，2025 年 1 月）将两者结合起来。

**Type:** 学习  
**Languages:** Python (标准库, W2SG gap 模拟器)  
**Prerequisites:** Phase 18 · 01 (指令遵循), Phase 18 · 10 (AI 控制), Phase 09 (强化学习 基础)  
**Time:** ~60 分钟

## 学习目标

- 定义可扩展监督和弱到强泛化，并解释它们如何互补。  
- 描述 Burns 等人 2023 的实验设置：使用 GPT-2 的标签来微调 GPT-4。  
- 解释 Performance Gap Recovered (PGR) 指标及其衡量内容。  
- 列出三种主要的可扩展监督机制（辩论、递归奖励建模、任务分解）及每种机制的一项优势。

## 问题

到目前为止 Phase 18 中的每种对齐技术都假设监督者可以评估模型的行为。当模型是超人类时，监督者成为薄弱环节。超对齐问题是：较弱的监督者能否可靠地产生一个更强且对齐的模型？

Burns 等人把这个问题简化为一个可操作的实证设置：用弱监督来监督强模型，度量有多少强模型能力能在弱监督下保存。这不是超对齐的解决方案——它是一个代理，使我们能够衡量进展。

## 概念

### W2SG：Burns 等人的设置

- 弱模型：GPT-2 类。  
- 强模型：GPT-4 类。  
- 目标：任务上的强 GPT-4 上限（ceiling）。

流程：
1. 获取弱模型在任务上的零样本预测（zero-shot predictions）。  
2. 用弱标签数据对强模型进行微调（fine-tune）。  
3. 测量微调后强模型的准确率。

基线：弱模型自身的准确率，以及强模型在真实标签（gold labels）上的监督上限（ceiling）。

差距指标：Performance Gap Recovered (PGR) = (fine-tuned - weak) / (ceiling - weak)。PGR = 1.0 表示弱监督完全弥补了差距；PGR = 0 表示弱监督没有帮助。

### Burns 等人的实证发现

在自然语言处理（NLP）、国际象棋谜题和奖励建模任务上，PGR 一直为正：强模型在一定程度上超越了弱监督者的错误。不同任务的 PGR 大致在 ~20% 到 ~80% 之间变化。强模型从不完美标签中“理解”了意图任务，并用其预训练先验进行填补。

Burns 等人指出的局限：
- 此处的弱到强差距是能力差距，而不是对齐差距。弱模型是已对齐的；真正的超人类模型可能由人类对齐，但能力差距更大。  
- 强模型的泛化可能更多依赖于任务先验，而不是恢复真实意图。  
- 如何引出强模型的潜在知识是真正的问题；PGR 测量的是一个特定的可操作化指标。

### 可扩展监督：三种机制

- **辩论（Debate, Irving et al. 2018）。** 两个 U 的实例就相反立场进行争辩；弱裁判（judge）做出选择。假设：找到一个有说服力的真实答案比找到一个有说服力的错误答案更容易。2024–2025 年的最新工作（Khan 等人、Michael 等人）表明，辩论有时有帮助、有时无帮助，其效果依赖于任务结构。  
- **递归奖励建模（Recursive Reward Modeling, Leike et al. 2018）。** U 帮助人类训练 U+1 的奖励模型。监督者的有效能力会随着模型能力而增长。  
- **任务分解（Task Decomposition, Christiano、Shlegeris、Amodei 2018）。** 将困难任务分解为人类可以检验的子任务，并递归进行。假设任务可分解。

每种机制都对任务结构或中间组件的对齐有某些假设。

### 为什么可扩展监督与 W2SG 是互补的

可扩展监督提高了监督者的有效信号质量。W2SG 则从监督者可以提供的任何不完美信号中弥补差距。

Lang 等人 — Debate Helps Weak-to-Strong Generalization（arXiv:2501.13124）将两者结合：用辩论协议生成更好的弱标签，然后用这些标签训练强模型。在 NLP 任务上报告了 PGR 的提升。

### 组织上的戏剧

OpenAI 的 Superalignment 团队在 2024 年 5 月于 Jan Leike 离职并去往 Anthropic 后解散。该议程（可扩展监督、W2SG、自动化对齐研究）在 Anthropic 和学术实验室继续开展——MATS（Lesson 28）、Redwood（Lesson 10）、Apollo（Lesson 8）、METR（Lesson 28）。组织结构发生了变化；研究问题并未改变。

### 在 Phase 18 中的位置

第 6–10 课描述了在假设 U 不值得信任时的威胁与防御范式。第 11 课是进攻范式：使监督者足够强以验证 U 的对齐。第 12–16 课随后转向对抗性评估的实用工具。

## 使用说明

`code/main.py` 在合成任务上模拟 W2SG 微调。弱标注器（weak labeler）具有 70% 的准确率并带有结构化错误；强模型在真实标签上的上限为 95%。你对强模型进行弱标签微调，测量 PGR，并与强模型在真实标签上的表现以及弱模型自身进行比较。

## 交付成果

本课生成 `outputs/skill-w2sg-pgr.md`。给定一个监督设置的描述，它会识别弱监督者、强模型、监督质量，并计算（或请求计算）PGR。它会标注该结论是“弱者可以监督强者”还是“弱者 + 监督机制可以监督强者”。

## 练习

1. 运行 `code/main.py`。报告 weak_accuracy = 0.60、0.70、0.80 时的 PGR。解释 PGR 曲线的形状。  

2. 修改弱标注器使其产生结构化错误（例如，总是在某一类输入上出错）。PGR 会增加、减少还是保持不变？解释原因。  

3. 阅读 Burns 等人 2023 第 4.3 节（NLP 任务）。复现“置信度辅助损失（confidence auxiliary loss）”的直觉：当强模型比弱标签更有置信度时，谁会“胜出”？  

4. 设计一个将辩论与任务分解结合用于软件工程任务的可扩展监督协议。分别列出每个组件的一个失败模式，并解释这两者的组合如何解决或未能解决这些失败模式。  

5. 阐明什么样的证据会反驳“弱到强泛化是通向超对齐的可行路径”这一主张。具体说明你需要看到的经验特征签名（empirical signature）。

## 术语要点

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Scalable oversight | "making the overseer stronger" | 增强监督者评估更高能力模型的能力的机制 |
| W2SG | "weak supervises strong" | 用弱标签微调强模型并测量恢复的能力 |
| PGR | "performance gap recovered" | (fine-tuned - weak) / (ceiling - weak); 1.0 = 完全弥补, 0 = 无帮助 |
| Debate | "two U instances argue" | 一种可扩展监督机制，由两个 U 辩护者对抗，弱裁判做出选择 |
| RRM | "recursive reward modeling" | U 帮助训练 U+1 的奖励模型；监督者能力随 U 跟进 |
| Task decomposition | "sub-tasks the human checks" | 将难任务拆成人类能检验的子任务，递归进行 |
| Superalignment | "aligning superhuman AI" | 关注对齐人类无法直接评估的模型的研究议程 |

## 扩展阅读

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — W2SG 论文  
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — 辩论机制  
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — 递归奖励建模  
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — 2024 年有关更有说服力的辩手如何影响真实性的实证研究  
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — 2025 年将辩论与 W2SG 结合的工作