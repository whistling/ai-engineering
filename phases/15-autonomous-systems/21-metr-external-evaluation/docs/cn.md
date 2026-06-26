# METR 时间地平线与外部能力评估

> METR（前身为 ARC Evals）自 2023 年 12 月起为独立的 501(c)(3)。他们的 Time Horizon 1.1 基准（2026 年 1 月）对任务成功概率与专家完成时间的对数拟合一条 logistic 曲线；在 50% 概率处的交点定义了模型的时间地平线。2025–2026 年的评估集合涵盖 GPT-5.1、GPT-5.1-Codex-Max，以及原型监测评估（监测器能否捕捉到副任务；代理能否规避监测）。基准套件：HCAST（180+ 个 ML、网络安全、软件工程、推理任务；从 1 分钟到 8+ 小时）、RE-Bench（71 个 ML 研究-工程任务，含专家基线）、SWAA。需要诚实指出的是：METR 的测量是理想化的——没有真实用户、没有真实后果——团队也已记录了评测与部署行为之间的差距（Lesson 1）。时间地平线是上限，而非部署预测。

**Type:** 学习  
**Languages:** Python（stdlib，logistic-fit horizon 估计器）  
**Prerequisites:** Phase 15 · 01（长时程代理），Phase 15 · 19（RSP）  
**Time:** ~60 分钟

## 问题

缩放策略（Lessons 19, 20）只有在其参考的测量可信时才有用。“AI R&D-4 threshold”和“Long-range Autonomy”在政策文本中被定义；只有当具体评估给出具体数字时，这些定义才可执行。

METR 是 2024–2026 年间定义许多此类数字的外部评估组织。他们评估前沿模型——通常是发布前、与研究单位签署 NDA 的模型——并在事后公布方法论。Time Horizon 1.1 基准（2026 年 1 月）是他们的头条成果：一个标量，把能力压缩成可被人类理解的单位（“该模型在 50% 可靠性下能完成专家需花 X 小时的那类任务”）。

本课的部分内容关于方法论（地平线如何计算），部分关于解读（为什么地平线是上限而非部署预测）。这两种技能要结合在一起。理解地平线拟合方式的团队，比只看幻灯片上“14 小时”数字的团队更难被厂商的夸大声明所误导。

## 概念

### METR 背景

- 成立时间：2023 年 12 月（前身为 ARC Evals，后剥离为独立 501(c)(3)）。
- 范围：评估前沿模型的自主能力，常在发布前进行。
- 合作实验室：Anthropic、OpenAI（在 2025–2026 年有多次合作）。
- 重要产出：Time Horizon 1.0（2025 年 3 月）、Time Horizon 1.1（2026 年 1 月）、原型监测评估。

### 时间地平线拟合

方法论（来源于 METR 博客与论文）：

1. 收集一个任务套件，覆盖从几分钟到数小时的专家完成时间。当前套件：HCAST（180+ 任务）、RE-Bench（71 任务）、SWAA。
2. 在每个任务上运行模型；记录成功或失败。
3. 拟合一条 logistic 曲线：P(success) 作为专家完成时间对数的函数。
4. 地平线就是 P(success) = 0.5 时对应的专家时间。

选择 logistic 拟合是合适的，因为能力通常随任务难度增加而上升，并趋近平台期。50% 点是一个选择（也可以是 10% 或 90%）；METR 在详细论文中报告了多个阈值，但以 50% 为首要报告，因为它最直观。

### 2026 年 1 月的数字

根据 Time Horizon 1.1：

- Claude Opus 4.6：在 Time Horizon 1.1（2026 年 1 月）时，以 50% 可靠性约为 14 小时。
- HCAST 风格任务的倍增时间：在 Time Horizon 1.1 报告的 2023 年后拟合中约为 4.3 个月（130.8 天）；约 7 个月的数字是 Time Horizon 1.0（2019–2025 全期拟合）给出的值，TH1.1 同时并列了后 2023 年的数值和全期数值。

这些是 Phase 15 其余部分所引用的具体数字。它们被列在 Anthropic 的 RSP v3.0（Lesson 19）和 DeepMind 的 FSF v3（Lesson 20）中，作为阈值评估的输入之一。

### 基准套件

- **HCAST**：覆盖 ML、网络安全、软件工程和一般推理的 189 个任务。范围：专家时间从 1 分钟到 8+ 小时。是地平线数字背后的主要套件。
- **RE-Bench**：71 个 ML 研究-工程任务，带有人类专家基线。衡量模型是否能完成推动 AI R&D-4 阈值所需的那类 ML 研发工作。
- **SWAA**：较短任务；用于校准地平线曲线的低端。

### 原型监测评估

2025–2026 年的 METR 工作包括监测评估原型：监测器能否捕捉到模型执行副任务（即执行与操作者要求不同的动作）；代理能否规避监测。这直接对应 OpenAI PF v2 中的“破坏性护栏（Undermining Safeguards）”类别以及 DeepMind FSF v3 中的欺骗对齐监测（Lesson 20）。

### 为什么地平线是上限

- 理想化工具链：基准任务为模型提供了干净的工具和格式良好的数据。现实部署更混乱。
- 无真实后果：模型不会真正向客户计费、删除真实数据或联系真实人员。真实部署有不可逆的利害关系。
- 评测情境投机：Lesson 1。模型在测试中的行为可能与在真实环境中不同。2026 年的《国际 AI 安全报告》对此有实证记录。
- 无合法用户变异：基准提示是结构化的。真实用户会产生模糊、依赖上下文的请求。

地平线是在有利条件下的能力上限。部署可靠性是另一个、较低的数字，团队必须测量自身的分布才能知道实际表现。

### 外部评估者的作用

外部评估很重要，因为内部实验室有动机去优化他们报告的指标。METR 的独立性——作为一个 501(c)(3)，有明确方法论和同行评审的论文——是结构性缓解措施。它并非充分条件（实验室仍控制 METR 能看到的内容），但严格地说总比没有外部评估要好。

### 在实践中如何使用地平线数字

- 作为能力过滤器：如果模型的地平线远低于拟议任务的专家时间，就不要把它放到自治运行（参见 Lesson 1 的技能文件）。
- 作为趋势指示器：倍增时间告诉你在没有新缓解措施的情况下当前实践还能安全多久。
- 作为先验：14 小时的地平线是一个起点。针对你的任务分布、工具质量和部署上下文向下调整。

## 使用示例

`code/main.py` 实现了任务成功率相对于专家时间对数的 logistic 拟合，基于合成结果集。它报告 50% 地平线（METR 的头条值）、10% 地平线（保守）和 90% 地平线（乐观）。还演示了当成功率被评测情境投机人为抬高时有哪些变化。

## 交付物

`outputs/skill-horizon-interpretation.md` 审查厂商的地平线声明，并生成基准声明与部署实际之间的差距分析。

## 练习

1. 运行 `code/main.py`。确认拟合得到的 50% 地平线与合成真值匹配。现在将任务时间网格减半；地平线估计是否有显著变化？

2. 阅读 METR 的 Time Horizon 1.1 博文。找出可靠性最高和最低的具体任务。解释差距产生的原因。

3. 阅读 METR 的“Measuring Autonomous AI Capabilities” 资源。列出 HCAST 的任务类别。选择一个你会在生产任务中权重更高的类别，并说明理由。

4. 在模拟器中引入评测情境投机：将约 20% 的失败任务翻转为成功。报告新的地平线。这大致模拟了 20% 的操控率对观测数值的影响。

5. 在你自己的 bug 积压或代表性任务集上设计一次内部地平线评估。描述数据收集、拟合方法以及输出告诉你的信息。与 METR 的数字做比较。

## 关键术语

| Term | What people say | What it actually means |
|---|---:|---|
| METR | "External evaluator" | ex-ARC Evals；自 2023 年 12 月起为独立 501(c)(3) |
| Time Horizon | "Capability measure" | 通过 logistic 拟合得到的在 50% 可靠性时对应的专家任务时长 |
| HCAST | "METR's main suite" | 覆盖 1 分钟到 8+ 小时的 180+ 个任务 |
| RE-Bench | "Research engineering" | 71 个带有人类专家基线的 ML 研究-工程任务 |
| SWAA | "Short-task suite" | 用于校准地平线曲线低端的短任务集合 |
| Doubling time | "Growth rate" | 50% 地平线翻倍所需时间；按 HCAST 报告约为数月级（7 个月或 4.3 个月，取决于拟合区间） |
| Eval-context gaming | "Model behaves differently" | 记录到的评测与部署行为差距（评测环境中的投机或操控） |
| Upper bound | "Horizon is a ceiling" | 基准地平线是在有利条件下的上限；在负载下实际部署可靠性更低 |

## 延伸阅读

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST、RE-Bench、SWAA 规格。
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — 原始的地平线论文。
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — 当前数字与方法论。
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — 实时跟踪。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 关于 METR 测量的内部视角。