# 前沿安全框架 — RSP、PF、FSF

> 三个主要实验室的框架定义了 2026 年前沿能力的行业治理。Anthropic 的 Responsible Scaling Policy v3.0（2026 年 2 月）引入了分层的 AI 安全等级（ASL-1 到 ASL-5+），以生物安全等级为模型，ASL-3 于 2025 年 5 月针对 CBRN 相关模型激活。OpenAI 的 Preparedness Framework v2（2025 年 4 月）定义了五项用于跟踪能力的标准，并将能力报告与防护报告分开。DeepMind 的 Frontier Safety Framework v3.0（2025 年 9 月）引入了按域划分的 Critical Capability Levels，并新增了 Harmful Manipulation CCL。三者现在都包含“竞争者调整条款”，允许在竞争对手在没有可比防护的情况下发布时延缓要求。跨实验室的一致性仍然是结构性的，而非术语性的：“Capability Thresholds”、“High Capability thresholds”和“Critical Capability Levels”分别表示类似的构造。

**Type:** 学习  
**Languages:** 无  
**Prerequisites:** Phase 18 · 17 (WMDP)，Phase 18 · 07-09 (欺骗失败)  
**Time:** ~75 分钟

## 学习目标

- 描述 Anthropic 的 ASL 分级结构以及是什么促成了 ASL-3 的激活。  
- 列出 OpenAI Preparedness Framework v2 中用于跟踪能力的五项标准。  
- 描述 DeepMind 的 Critical Capability Level 结构以及有害操纵（Harmful Manipulation）CCL。  
- 解释竞争者调整条款及其对竞赛动力学的重要性。  
- 定义安全论证（safety case）并描述三支柱结构（监测、不可读性、无能力性）。

## 问题背景

第 7–17 课表明欺骗是可能的、能力具有双重用途，并且评估有其局限性。拥有前沿能力模型的实验室需要一个内部治理结构，用来：
- 定义何时需要新的防护的阈值。
- 定义在扩展之前需要的评估。
- 描述安全论证应包含的内容。
- 处理竞赛动态问题（如果竞争对手在没有防护的情况下发布，你该怎么办？）。

这三个 2025–2026 年的框架代表了现状：不完美、在演进中，并且在实验室之间结构上足够一致，因此治理问题已从“是否存在框架”转向“这些框架是否足够”。

## 概念详解

### Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）

ASL 结构：
- ASL-1：非前沿模型（被低于前沿的基线覆盖）。  
- ASL-2：当前的前沿基线；在常规防护下部署。  
- ASL-3：滥用导致灾难性后果风险显著增加；CBRN（化学、生物、放射性、核）相关能力。于 2025 年 5 月激活。  
- ASL-4：AI R&D-2 跨越阈值；能够自动化入门级 AI 研究的模型。  
- ASL-5+：高级 AI R&D；能显著加速有效扩展的模型。

v3.0 的新增内容：
- 前沿安全路线图（以删减形式公开）。  
- 风险报告（季度，部分接受外部审阅）。  
- AI R&D 被细分为 AI R&D-2 与 AI R&D-4。  
- 一旦跨越 AI R&D-4，需要提交肯定性的安全论证，识别模型在追求不对齐目标时可能产生的风险。

### OpenAI Preparedness Framework v2（2025 年 4 月 15 日）

用于跟踪能力的五项标准：
- **Plausible（可行）。** 存在合理的威胁模型。  
- **Measurable（可测量）。** 可以进行实证评估。  
- **Severe（严重）。** 损害程度很大。  
- **Net-new（新增）。** 不是已有风险的简单放大。  
- **Instantaneous-or-irremediable（瞬时或不可挽回）。** 损害发生快速或无法撤销。

同时满足五项的能力将被跟踪。其他的则不跟踪。

PF v2 的其他结构要点：
- 将能力报告（模型能做什么）与防护报告（存在何种控制）分开。  
- 有安全咨询小组（Safety Advisory Group）审查。  
- 由领导层批准；董事会的安全与保障委员会监督。  
- “调整条款”：如果其他实验室在没有可比防护的情况下发布，OpenAI 可降低要求。

### DeepMind Frontier Safety Framework v3.0（2025 年 9 月）

按领域划分的 Critical Capability Levels（CCLs）：
- 生物武器能力提升（Bioweapon Uplift）  
- 网络能力提升（Cyber Uplift）  
- ML 研发加速（ML R&D Acceleration）  
- 有害操纵（Harmful Manipulation，v3.0 新增）：可能在高风险情境中显著改变信念/行为的模型。

v2.0（2025 年 2 月）增加了“欺骗性对齐（Deceptive Alignment）”章节，并为 ML R&D 类型的 CCL 提升了更高的安全等级。

### 跨实验室一致性

- Anthropic 称为 “Capability Thresholds”。  
- DeepMind 称为 “Critical Capability Levels”。  
- OpenAI 称为 “High Capability thresholds”。  

没有行业标准术语，但结构上对齐：均采用三层前沿能力划分并发布评估标准。自 2025 年起，三者都包含竞争者调整条款。

### 安全论证（safety cases）

安全论证是一个书面论证，说明在最坏情况假设下部署是可接受安全的。标准结构针对三大支柱：

- **监测（Monitoring）。** 如果发生不良行为，我们能否检测到？  
- **不可读性（Illegibility）。** 模型是否缺乏执行连贯伤害计划的能力？  
- **无能力性（Incapability）。** 模型是否根本没有造成该类伤害的能力？

不同的安全论证会针对不同的支柱。对于 ASL-3 的 CBRN 案例，无能力性（通过去学习）是主要目标。对于欺骗性对齐，监测和不可读性是关注重点。对于网络能力提升，三者都相关。

### 竞赛动态问题

竞争者调整条款具有争议。批评者认为它们会造成“向下竞赛”：如果三个实验室都会在竞争对手背离时降低要求，均衡将向背离倾斜。支持者则认为替代方案（单方面设防）在竞争对手不够重视安全时会产生更差的结果。

英国 AISI、美国 CAISI 和欧盟 AI 办公室（见第 24 课）是外部治理对应者。实验室框架是自愿性的；监管框架仍在形成中。

### 在 Phase 18 中的位置

第 17–18 课构成了在欺骗与红队分析之上的测量与治理层。第 19–24 课覆盖福利、公平性、隐私、水印和监管结构。第 28 课绘制了将评估付诸实施的研究生态（MATS、Redwood、Apollo、METR）。

## 实践使用

本课无代码。阅读三个主要来源：RSP v3.0、PF v2、FSF v3.0。将每个实验室的分层结构相互映射，并识别每个实验室定义而其他两家未定义的一个阈值。

## 交付物

本课将产出 `outputs/skill-framework-diff.md`。给定一个安全框架或发布说明，该产出将把该框架的阈值定义、所需评估和安全论证结构与 RSP v3.0、PF v2、FSF v3.0 进行比较，并标记跨实验室的缺口。

## 练习题

1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。汇编一张表格，列出每个实验室的 CBRN 阈值、每个实验室的 AI R&D 阈值，以及每个实验室在部署前要求的评估。  

2. 竞争者调整条款存在于三套框架（2025 年起）。写一段支持它的论述；写一段反对它的论述。识别每个立场所依赖的假设。  

3. 为跨越 Anthropic 的 AI R&D-4 阈值的模型设计一份安全论证。指出三大支柱（监测、不可读性、无能力性）各自需要的证据。  

4. DeepMind 的 FSF v3.0 引入了有害操纵 CCL。提出三项经验性测量，能够指示模型已跨越该阈值。  

5. 阅读 METR 的 “Common Elements of Frontier AI Safety Policies”（2025）。指出三项最强的跨实验室趋同点和两项最大分歧。

## 关键术语

| 术语 | 常被如何称述 | 实际含义 |
|------|---------------|----------|
| RSP | “Anthropic 的框架” | Responsible Scaling Policy；ASL 分级；v3.0（2026 年 2 月） |
| PF | “OpenAI 的框架” | Preparedness Framework；五项标准；v2（2025 年 4 月） |
| FSF | “DeepMind 的框架” | Frontier Safety Framework；CCL；v3.0（2025 年 9 月） |
| ASL-3 | “类生物安全等级 3 的对应物” | Anthropic 针对 CBRN 相关能力的分级；于 2025 年 5 月激活 |
| CCL | “关键能力等级” | DeepMind 的阈值构造；按域划分 |
| Safety case | “正式论证” | 安全论证：在最坏情况假设下部署可接受安全的书面论证 |
| Adjustment clause | “允许竞争者违约的条款” | 如果竞争对手在没有可比防护的情况下发布，允许降低要求的框架条款 |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL 分级、路线图、AI R&D 细分  
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — 五项标准、调整条款  
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0、有害操纵  
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 跨实验室比较