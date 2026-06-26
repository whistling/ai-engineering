# Alignment Research Ecosystem — MATS, Redwood, Apollo, METR

> 五个组织定义了 2026 年的非实验室（非厂内）对齐研究层。MATS (ML Alignment & Theory Scholars)：自 2021 年末以来 527+ 名研究人员，180+ 篇论文，10K+ 次引用，h 指数 47；2024 年夏季学员队列作为 501(c)(3) 成立，约 90 名学者和 40 名导师；2025 年前约 80% 的校友从事安全/安保相关工作，200+ 人在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo 就职。Redwood Research：由 Buck Shlegeris 创立的应用对齐实验室；提出了 AI Control（Lesson 10）；与 UK AISI 合作开展控制安全案例研究。Apollo Research：为前沿实验室提供部署前的策划（scheming）评估；撰写了 In-Context Scheming（Lesson 8）和 Towards Safety Cases for AI Scheming。METR (Model Evaluation and Threat Research)：基于任务的能力评估、自治任务的时间地平线研究；《Common Elements of Frontier AI Safety Policies》比较了各实验室的框架。Eleos AI Research：模型福利（model-welfare）部署前评估（Lesson 19）；进行了 Claude Opus 4 的福利评估。

**Type:** 学习  
**Languages:** 无  
**Prerequisites:** Phase 18 · 01-27（之前的 Phase 18 课程）  
**Time:** ~45 分钟

## 学习目标

- 识别非实验室对齐研究生态系统中的五个组织及其核心产出。  
- 描述 MATS 的规模（学者数量、论文、h 指数）及其作为人才管道的角色。  
- 描述 Redwood 的 AI Control 议程及其与 UK AISI 的合作关系。  
- 描述 METR 的基于任务的评估方法论。

## 问题背景

前沿实验室（Lesson 18）在内部进行安全评估并发布选定结果。实验室外的生态系统是评估被验证的地方，也是首次发现新型失效模式与培养人才的所在。理解该生态系统有助于判断哪些研究发现被哪些群体信任。

## 概念说明

### MATS (ML Alignment & Theory Scholars)

始于 2021 年末。研究导师制项目；学者与资深研究者一起在特定对齐问题上工作 10–12 周。

规模（2026 年）：
- 自成立以来 527+ 名研究人员。  
- 发表 180+ 篇论文。  
- 10K+ 次引用。  
- h 指数为 47。  
- 2024 年夏季：90 名学者 + 40 名导师；注册为 501(c)(3)。

职业去向：截至 2025 年前的校友中约 80% 从事安全/安保工作。200+ 人在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo 任职。

### Redwood Research

应用对齐实验室。由 Buck Shlegeris 创立。提出了 AI Control 议程（Lesson 10）。与 UK AISI 合作开展控制安全案例研究。为 DeepMind 和 Anthropic 提供评估设计咨询。

代表性论文：Greenblatt、Shlegeris 等人，《AI Control》（arXiv:2312.06942，ICML 2024）；Alignment Faking（Greenblatt、Denison、Wright 等，arXiv:2412.14093，与 Anthropic 联合）。

风格：侧重具体的威胁模型、最坏情形的对手、以及可以被压力测试的具体协议。

### Apollo Research

为前沿实验室提供部署前的策划（scheming）评估。撰写了 In-Context Scheming（Lesson 8，arXiv:2412.04984）。是 2025 年 OpenAI 反策划训练合作的伙伴。发表了《Towards Safety Cases for AI Scheming》（2024）。

风格：在具代理性的设置中进行评估，关注可能出现的欺骗行为；采用三支柱分解法（不对齐、目标导向性、情境意识）。

### METR (Model Evaluation and Threat Research)

基于任务的能力评估。自治任务完成的时间地平线研究。《Common Elements of Frontier AI Safety Policies》（metr.org/common-elements，2025）比较了各实验室的框架。

与 Apollo 合作共同作为 AI Scheming 安全案例草案的合著者。

风格：长时间地平线任务评估、经验性能力度量、框架综合。

### Eleos AI Research

模型福利（model-welfare）部署前评估。进行了文档化在系统卡第 5.3 节的 Claude Opus 4 福利评估。为 Lesson 19 中与福利相关的主张提供外部方法论审查。

### 流程

MATS 培养研究人员。毕业生进入 Anthropic、DeepMind、OpenAI（实验室安全团队）或去往 Redwood、Apollo、METR、Eleos（外部评估组织）。外部评估者与实验室及 UK AISI / CAISI 合作。出版物将研究成果反馈回 MATS，为下一届学员提供输入。

### 为什么这一层很重要

单一来源的评估不可靠：实验室对自家模型进行评估存在结构性利益冲突。外部评估者可以提出并验证实验室可能未充分报告的失效模式。2024 年的 Sleeper Agents 论文（Lesson 7）由 Anthropic + Redwood 合作；Alignment Faking 由 Anthropic + Redwood；In-Context Scheming 由 Apollo；Anti-Scheming 由 Apollo + OpenAI。多组织结构起到了质量控制的作用。

### 在 Phase 18 中的位置

Lessons 7–11 引用了 Redwood 和 Apollo 的工作；Lesson 18 引用了 METR 的框架比较；Lesson 19 引用了 Eleos。Lesson 28 是整个 Phase 所依赖的生态系统的明确组织地图。

## 使用建议

无代码。阅读 METR 的《Common Elements of Frontier AI Safety Policies》作为外部综合如何为实验室内部政策工作增值的示例。

## 交付

本课产出 `outputs/skill-ecosystem-map.md`。给定一项对齐主张或评估，该产出会识别相关组织、发表渠道和方法学风格，并与已知的对应组织交叉核验。

## 练习

1. 从 Lessons 7–15 中选一篇论文，识别参与的组织。将作者与 MATS 校友及当前生态系统的隶属关系进行交叉核对。  
2. 阅读 METR 的《Common Elements of Frontier AI Safety Policies》。识别他们强调的三个跨实验室趋同点以及两处最大分歧。  
3. MATS 的职业去向约为 80% 从事安全/安保。论证这种选择压力是适应性的（培养该领域）还是存在偏差（过滤掉异端观点）。  
4. Redwood 与 Apollo 都从事控制/策划工作但风格不同。选取一种失效模式并描述两者各自会如何调查它。  
5. Eleos AI 是唯一的纯粹模型福利组织。设计一个假想的第二个组织，聚焦不同但相关的福利问题（如认知自由、机器人化身等），并阐述其方法论。

## 关键词

| 术语 | 人们如何称呼 | 实际含义 |
|------|---------------|----------|
| MATS | "the mentorship program" | ML Alignment & Theory Scholars；自 2021 年以来 527+ 名研究人员 |
| Redwood Research | "the control lab" | 应用对齐；AI Control 作者；UK AISI 合作伙伴 |
| Apollo Research | "the scheming evals" | 为前沿实验室提供部署前的策划（scheming）评估 |
| METR | "the task-horizon evals" | 基于任务的能力评估；框架综合 |
| Eleos AI | "the welfare lab" | 模型福利部署前评估 |
| Talent pipeline | "MATS -> labs" | MATS 毕业生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| External evaluation | "non-lab check" | 非模型生产者执行的评估；增加可信度 |

## 延伸阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — 导师项目  
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control 相关论文与资料  
- [Apollo Research](https://www.apolloresearch.ai/) — 策划（scheming）评估研究  
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比较  
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福利方法论