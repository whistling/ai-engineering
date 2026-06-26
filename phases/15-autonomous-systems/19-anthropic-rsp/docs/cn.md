# Anthropic Responsible Scaling Policy v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代了 2023 年的政策。两层缓解措施：Anthropic 单方面会做的事情 vs 被表述为行业范围建议的事项（包括 RAND SL-4 安全标准）。将 Frontier Safety Roadmaps 和 Risk Reports 提升为常设文件，而不是一次性成果。删除了 2023 年的暂停承诺。引入了 AI R&D-4 阈值：一旦超过，Anthropic 必须发布一份肯定性论证，识别不一致风险和缓解措施。Claude Opus 4.6 未达到该阈值。Anthropic 在 v3.0 公告中指出，“自信地排除这一点变得困难”。SaferAI 将 2023 年的 RSP 评为 2.2；他们将 v3.0 降级为 1.9，把 Anthropic 与 OpenAI 和 DeepMind 一起列入“弱”RSP 类别。定性阈值取代了 2023 年的定量承诺；移除暂停条款是最明显的回退。

**Type:** 学习  
**Languages:** Python（stdlib，RSP 阈值决策引擎）  
**Prerequisites:** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)  
**Time:** ~45 分钟

## 问题

前沿实验室发布的规模化政策既是技术文档，也是治理文件，还是向监管机构发出的信号。RSP v3.0 是当前 Anthropic 的文档。仔细阅读它很重要，不是因为遵守它有约束力（没有），而是因为其表述方式会影响实验室如何构想灾难性风险，以及它们如何向公众传达权衡。

v3.0 与 v2.0 的差异是有用的分析单元。新增了什么：Frontier Safety Roadmaps、Risk Reports、AI R&D-4 阈值。移除了什么：2023 年的暂停承诺。重构了什么：将缓解措施分为 Anthropic 单方面执行和行业级建议的两栏。外部评估 —— SaferAI —— 将分数从 2.2（v2）降到 1.9（v3.0）。这正说明一个规模化政策如何在看起来更光鲜的同时变得不那么严格。

## 概念

### 两层缓解计划

- **Anthropic 单方面行动**：Anthropic 无论其他实验室如何都会执行的事项。比如在某个阈值以上停止训练、特定的安全措施、具体的部署门槛。
- **行业范围建议**：Anthropic 认为行业应集体采取的行动。包括 RAND SL-4 安全标准。这些不是 Anthropic 的承诺；它们是政策倡议。

两层结构在 v2 中不存在。这意味着读者需要查看每项承诺位于哪一栏。位于“行业范围建议”栏的安全措施不是 Anthropic 的保证；它只是 Anthropic 的期望。

### AI R&D-4 阈值

这是 RSP v3.0 指名为下一个重要阈值的能力水平。具体来说：一个能够以有竞争力的成本自动化大量 AI 研究的模型。一旦 Anthropic 认为某个模型超过该阈值，他们必须在继续扩展之前发布一份肯定性论证，说明已识别的不一致风险和可行的缓解措施。

Claude Opus 4.6 在 v3.0 公告中未超过该阈值。文档中补充道：“自信地排除这一点变得困难。” 这句话很关键；它承认该阈值已经足够接近，成为一个实实在在的关注点，而不是一种推测性的界限。

第 6 课（自动化对齐研究）和第 7 课（递归自我改进）直接与该阈值相关。自动化对齐研究者达到研究质量门槛表明 AI R&D-4 阈值正在逼近。

### Frontier Safety Roadmaps 与 Risk Reports

v3.0 将两种文档类型提升为常设文件：

- **Frontier Safety Roadmap**：前瞻性文档，描述计划中的安全工作、能力预期和缓解研究。
- **Risk Report**：回顾性文档，针对发布后特定模型，描述观察到的能力和剩余风险。

两者均公开，并按规定节奏更新。其用处在于：读者可以跟踪 Anthropic 在 Roadmap 中承诺要做的事情与他们在 Risk Report 中实际报告的情况之间的差异。

### 移除暂停条款

2023 年的 RSP 包含明确的暂停承诺：如果模型超过特定能力阈值，训练将在缓解措施到位之前暂停。v3.0 用更软化的表述替代了明确的暂停（发布肯定性论证，如果缓解足够则继续）。SaferAI 和其他分析人士直言不讳地指出这是新文档中最显著的回退。

支持该变动的政策论点是：2023 年的定量阈值在 2026 年的能力基准重新标定后变得难以实现。反方论点是：规模化政策中的暂停条款是承诺机制；移除它会削弱政策的可信度。

### SaferAI 的降级

SaferAI 是一家独立组织，对 RSP 类文档进行评分。他们的公开评分：2023 年 Anthropic RSP 得分 2.2（满分标尺中 4.0 为当前最好的 RSP，1.0 为名义水平）。v3.0 得分 1.9。这使 Anthropic 从“中等”降为“弱”，与 OpenAI 和 DeepMind 一同进入弱势类别。

SaferAI 降级的因素包括：
- 定性阈值取代了定量阈值。
- 暂停承诺被移除。
- AI R&D-4 阈值下的缓解被表述为“肯定性论证”而不是具体措施。
- 审查机制依赖于 Anthropic 的 Safety Advisory Group，独立监督有限。

### 本课不涵盖的内容

这不是一堂合规课。RSP v3.0 不是法规；没有任何强制力要求 Anthropic 遵守它。本课的价值在于以应有的具体性和怀疑精神阅读该文档。规模化政策是前沿实验室就灾难性风险姿态发出的主要公共信号。善于阅读它们是任何依赖前沿能力工作的人的实用技能。

## 使用方式

`code/main.py` 实现了一个小型决策引擎，反映了 RSP 阈值评估的形态：给定一个候选模型和一组能力测量，返回是否超过 AI R&D-4 阈值、所需的肯定性论证章节，以及是否可以继续部署。它故意很简单；目的是使文档逻辑显性化。

## 交付物

`outputs/skill-scaling-policy-review.md` 根据 v3.0 参考模板对一个规模化政策（Anthropic、OpenAI、DeepMind 或内部）进行审查：两层结构、阈值、暂停承诺、独立审查。

## 练习

1. 运行 `code/main.py`。输入三个能力水平不同的合成模型。确认阈值评估器的行为与预期一致，并生成正确的肯定性论证模板。

2. 通读 RSP v3.0（32 页）。找出所有位于“行业范围建议”层的承诺。在 v2 中哪些会列为“Anthropic 单方面”？

3. 阅读 SaferAI 的 RSP 评分方法。通过将他们的量表应用于文档，重现 1.9 的分数。哪一行评分最驱动了降级？

4. 2023 年的暂停承诺被移除。提出一项替代承诺，在承认 2026 年基准重标定问题的同时，保留政策的可信度。

5. 将 RSP v3.0 与 OpenAI Preparedness Framework v2（第 20 课）进行比较。选一个 v3.0 更强的方面。再选一个 Preparedness Framework 更强的方面。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| RSP | "Anthropic's scaling policy" | 负责任的规模化政策（Responsible Scaling Policy）；v3.0 于 2026-02-24 生效 |
| AI R&D-4 | "Research-automation threshold" | 能以有竞争力的成本自动化大量 AI 研究的能力阈值 |
| Affirmative case | "Safety justification" | 已发布的论证，说明已识别风险并且缓解措施足够 |
| Frontier Safety Roadmap | "Forward plan" | 关于计划中的安全工作和预期能力的常设文档 |
| Risk Report | "Retrospective on a model" | 关于发布后观察到的能力和剩余风险的常设文档 |
| Two-tier mitigation | "Unilateral vs industry" | Anthropic 的单方面承诺 vs 行业建议，两者分列 |
| Pause commitment | "2023 clause" | 明确的训练暂停承诺；在 v3.0 中被移除 |
| SaferAI rating | "Independent RSP grade" | 第三方量表；v3.0 得分 1.9（v2 为 2.2） |

## 进一步阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整的 32 页政策文档。  
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — 关于与 v2 差异的摘要。  
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — RSP v3.0 中引用的常设文档。  
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 当前前沿模型的回顾报告。  
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 与可测自治性联系起来的研究。