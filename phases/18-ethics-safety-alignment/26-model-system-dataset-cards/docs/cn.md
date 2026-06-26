# 模型卡、系统卡与数据集卡

> 三种文档格式构成了 AI 透明度的结构。Model Cards（Mitchell 等人，2019）——模型的营养成分表：训练数据、量化的分解分析、伦理考量、注意事项；在 Hugging Face 的模型卡中只有 0.3% 记录了伦理考量（Oreamuno 等人，2023）。Datasheets for Datasets（Gebru 等人，2018，CACM）——动机、组成、收集过程、标注、分发、维护；类似电子元件规格书的类比。Data Cards（Pushkarna 等人，Google 2022）——模块化的分层细节（望远镜级、潜望镜级、显微级），作为面向不同读者的边界对象。2024–2025 年的发展：通过 LLM 自动生成（CardGen，Liu 等人 2024）；模型卡详尽度与 Hugging Face 下载量相关，最高可提高 29%（Liang 等人 2024）；可验证的证明（Laminator，Duddu 等人 2024）；碳/水等可持续性报告的补充（Jouneaux 等人，2025 年 7 月）；欧盟/ISO 的监管卡片开始出现。System Cards（Sidhpurwala 2024；Meta 的系统级透明工作；“Blueprints of Trust” arXiv:2509.20394）——覆盖端到端 AI 系统的文档，包含安全能力、提示词注入防护、数据外泄检测、与人类价值对齐等内容。

**Type:** 构建  
**Languages:** Python（stdlib、model-card + datasheet + system-card 生成器）  
**Prerequisites:** 第18阶段 · 18（安全框架）、第18阶段 · 24（监管）  
**Time:** ~60 分钟

## 学习目标

- 描述原始的 Mitchell 等人 2019 模型卡和 Gebru 等人 2018 数据表。
- 描述 Data Cards 的望远镜级 / 潜望镜级 / 显微级分层结构。
- 描述 System Cards 及其端到端覆盖范围。
- 列举三项 2024–2025 年的发展（自动生成、可验证的证明、可持续性报告）。

## 问题

监管框架（第24课）和实验室安全政策（第18课）都要求文档记录。文档格式已从模型专属（模型卡）演进到数据集专属（数据表）再到系统专属（系统卡）。每种格式针对不同的透明度范围。2024–2025 年的自动化与可验证证明工作解决了长期存在的采用问题。

## 概念

### Model Cards（Mitchell 等人 2019）

章节包括：
- 模型详情。
- 预期用途。
- 影响因素（评估时相关的人群或环境因素）。
- 指标。
- 评估数据。
- 训练数据。
- 定量分析（按影响因素分解）。
- 伦理考量。
- 限制与建议。

采用问题：Oreamuno 等人 2023 对 Hugging Face 模型卡的审计发现，只有 0.3% 记录了伦理考量。

### Datasheets for Datasets（Gebru 等人 2018）

类似电子规格书的类比。章节包括：
- 动机（为什么创建该数据集）。
- 组成（包含哪些内容）。
- 收集过程（如何组装）。
- 标注（如适用）。
- 用途（预期用途、禁止用途、风险）。
- 分发。
- 维护。

发表于 CACM 2021。数据表是上游文档；模型卡依赖于数据表的准确性。

### Data Cards（Pushkarna 等人，Google 2022）

模块化的分层细节。三个缩放层级：
- **望远镜级（Telescopic）。** 面向非专家的高层摘要。
- **潜望镜级（Periscopic）。** 面向 ML 实践者的中层概览。
- **显微级（Microscopic）。** 面向审计员的细粒度特征级文档。

边界对象框架：不同读者可从同一文档中提取不同信息。

### System Cards

范围：端到端 AI 系统，包括模型 + 安全栈 + 部署上下文。典型章节包括：
- 安全能力。
- 提示词注入防护。
- 数据外泄检测。
- 与声明的人类价值对齐。
- 事件响应。

参考 Sidhpurwala 2024 与 Meta 的系统级透明工作。“Blueprints of Trust”（arXiv:2509.20394）将 System Card 形式化，作为对 Model Cards 的部署层补充。

### 2024–2025 年的发展

- **CardGen（Liu 等人 2024）。** 通过 LLM 自动生成模型卡；在标准化的 Mitchell 2019 字段上报告的客观性高于许多人类撰写的卡片。
- **下载量相关性（Liang 等人 2024）。** 详尽的模型卡与 Hugging Face 上最高可达 29% 的下载增长相关——采用压力现在由市场驱动，而不仅仅是合规驱动。
- **Laminator（Duddu 等人 2024）。** 通过硬件 TEE / 加密签名实现可验证的证明——使模型卡能携带证明（proof-of-claim），而不仅仅是声明。
- **可持续性（Jouneaux 等人，2025 年 7 月）。** 在模型卡中加入碳、水和计算能耗足迹的字段；相关的 ISO 标准正在出现。
- **监管卡。** 欧盟 AI 法规（第24课）、GPAI《行为准则》中关于透明度的章节要求将模型卡作为合规产物。

### 本内容在第18阶段中的位置

第24–25课属于监管和 CVE 层面。第26课是文档层。第27课是训练数据治理（数据表的上游）。第28课是产生卡片中引用评估的研究生态。

## 使用方法

`code/main.py` 生成用于玩具部署的最小模型卡、数据表和系统卡。每个文件遵循规范的章节结构。你可以检查格式并比较三种范围的差异。

## 交付

本课产生 `outputs/skill-card-audit.md`。给定一个模型卡、数据表或系统卡，该工具会审核章节覆盖情况、数值分解细化程度，以及是否存在可验证证明。

## 练习

1. 运行 `code/main.py`。检查生成的卡片。识别仅含占位符的薄弱章节，并说明哪些证据可以增强它们。
2. 在模型卡中添加一个跨两个人口统计群体的量化分解分析（第20课）。
3. 阅读 Oreamuno 等人 2023 关于 0.3% 采用率的研究。提出一项结构性变更，能提高伦理考量字段的采用率。
4. Laminator（Duddu 等人 2024）使用 TEE 实现可验证证明。设计一个模型卡字段，用于携带某项评估结果的加密证明，并描述验证者的角色。
5. 为你过去的一个项目或一个假设部署撰写一份 System Card（系统卡，非模型卡）。识别对第三方审计员价值最高的章节。

## 术语表

| 术语 | 常见说法 | 实际含义 |
|------|---------|----------|
| Model Card | “the Mitchell card” | Mitchell 等人 2019 的机器学习模型标准化文档 |
| Datasheet | “the Gebru datasheet” | Gebru 等人 2018 的数据集标准化文档 |
| Data Card | “the Pushkarna card” | Google 2022 的模块化分层数据文档 |
| System Card | “the deployment card” | 包含安全栈的端到端 AI 系统文档 |
| Boundary object | “different readers, one doc” | Data Cards 的框架：同一文档服务于不同受众 |
| Verifiable attestation | “the Laminator attestation” | 附加到文档声明上的加密或 TEE 证明 |
| Sustainability field | “carbon / water footprint” | 2025 年出现的用于环境核算的字段（碳/水/能耗） |

## 进一步阅读

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — 规范性的模型卡
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — 数据表论文
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — 分层数据文档
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — System Card 形式化文档