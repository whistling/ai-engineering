# 合规 — SOC 2、HIPAA、GDPR、PCI-DSS、欧盟 AI 法案、ISO 42001

> 多框架覆盖已成为 2026 年企业成交的基础门槛。**欧盟 AI 法案（EU AI Act）**：自 2024 年 8 月 1 日生效。大多数高风险要求自 2026 年 8 月 2 日开始强制执行。对高风险系统义务的罚款上限为 1,500 万欧元或全球年营业额的 3%（Art. 99(4)）；对禁止性 AI 行为的罚款上限为 3,500 万欧元或全球年营业额的 7%（Art. 99(3)）。若面向欧盟用户则适用。**科罗拉多州 AI 法案（Colorado AI Act）**：生效日为 2026 年 6 月 30 日（因 SB25B-004 从 2026 年 2 月延后）——对高风险系统要求进行影响评估，并赋予对 AI 决策的上诉权。弗吉尼亚州对信贷/就业/住房/教育领域有类似要求。**SOC 2 Type II**：已成为 B2B AI 的事实性要求（对金融科技类客户尤其是 Type II，而非 Type I）。**GDPR**：迄今记录的最大针对 AI 的罚款为对 Clearview AI 的 3,050 万欧元（荷兰数据保护机构，2024 年 9 月）；意大利监管机构对 OpenAI 开出的 1,500 万欧元罚款（2024 年 12 月）于 2026 年 3 月上诉被撤销。推理阶段对 PII 的实时脱敏是可辩护的标准；事后处理清理不足以合规。**HIPAA**：医疗场景受约束——在没有签署 BAA 的情况下不能将 PHI 发送到外部 AI 服务。**PCI-DSS**：涉及支付数据的 AI 交互层需要配置与合同约定来覆盖，并非自动涵盖。**ISO 42001**：新兴的 AI 治理标准，正与 ISO 27001 一并成为采购要求。参考配置：OpenAI 保持 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA、以及用于 ChatGPT 支付组件的 PCI-DSS。跨框架映射可减少审计疲劳：访问控制可映射到 ISO 27001 A.5.15-5.18、GDPR 第 32 条、HIPAA §164.312(a)。

**Type:** 学习  
**Languages:** (Python optional — compliance is policy + process, not code)  
**Prerequisites:** Phase 17 · 25（安全）, Phase 17 · 13（可观察性）  
**Time:** ~60 分钟

## 学习目标

- 列举 2026 年与 LLM 产品相关的七个框架，并将每个框架匹配到相应的客户细分。  
- 引述欧盟 AI 法案的执行时间表（2024 年 8 月生效；高风险要求自 2026 年 8 月强制执行）以及两级罚款上限（高风险义务：€15M / 3%；禁止性行为：€35M / 7%）。  
- 解释为何事后 PII 清理不足以满足 GDPR 要求，并把推理层的实时脱敏命名为可辩护的标准。  
- 描述跨框架控制映射（例如访问控制映射到 ISO 27001 A.5.15-5.18 + GDPR 第 32 条 + HIPAA §164.312(a)）。

## 问题

企业客户的采购要求 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001，以及“一份欧盟 AI 法案合规声明”。你的团队只有 SOC 2 Type I。距离 Type II 还有六个月，而且还没有开始 GDPR 第 30 条记录的准备工作。

多框架覆盖并非单纯 LLM 问题——它是企业 SaaS 的问题，加上 LLM 特殊的覆盖项。到 2026 年，采购团队希望看到一个按框架为行、按控制为列的矩阵，而不是一份 PDF。

## 概念

### 七个框架

| Framework | Scope | LLM-specific requirement |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS 基线 | 流程控制需在 6-12 个月内接受审计 |
| HIPAA | 美国医疗 | 需要 BAA；未经签署协议 PHI 不能离开基础设施 |
| GDPR | 面向欧盟用户 | 推理阶段实时 PII 脱敏；数据主体权利；第 30 条记录 |
| PCI-DSS | 支付数据 | 涉及支付的 AI 需要通过配置 + 合同进行覆盖 |
| EU AI Act | 面向欧盟用户 | 风险分层；高风险系统：符合性评估、文档、日志记录 |
| Colorado AI Act | 面向科罗拉多州居民 | 影响评估；上诉权 |
| ISO 42001 | AI 治理 | 新兴标准；与 ISO 27001 配合使用 |

### 欧盟 AI 法案时间表

- 2024 年 8 月 1 日：生效。  
- 2025 年 2 月 2 日：禁止性 AI 行为开始强制执行。  
- 2026 年 8 月 2 日：高风险系统开始强制执行（符合性评估、文档、日志记录）。  
- 2027 年 8 月：在受协调立法约束的产品中对高风险系统强制实施。

风险分层：不可接受（禁止）、高风险（需符合性 + 日志）、有限风险（需透明度）、最低风险（无约束）。大多数 B2B LLM SaaS 属于有限风险；当涉及就业、信贷、教育、执法、移民、关键服务时，可能被判定为高风险。

罚款（第 99 条）：对高风险系统义务的违规可处以最高 1,500 万欧元或全球年营业额的 3%（Art. 99(4)）；对禁止性 AI 行为的违规可处以最高 3,500 万欧元或全球年营业额的 7%（Art. 99(3)）；适用更高者。

### GDPR — 推理层实时脱敏是标准

事后处理（在 LLM 已看到数据后再做脱敏）并非可辩护的合规做法——模型已经“看到”了该数据。2026 年的可辩护标准是推理层的实时脱敏：

- 在调用 LLM 之前进行实体识别。  
- 使用一致的分词策略（例如 Mesh 方法）来保留语义。  
- 仅存储脱敏后的提示词 + 经同意的原始数据（opt-in）。

近期执法案例：对 Clearview AI 的 3,050 万欧元罚款（荷兰 DPA，2024 年 9 月）是迄今为止记录的最大一笔针对 AI 的 GDPR 罚款；意大利监管机构对 OpenAI 的 1,500 万欧元罚款（2024 年 12 月）是迄今最大的一笔针对 LLM 的罚款，但该罚款于 2026 年 3 月上诉被撤销，目前仍在进一步审查中。事后处理声明在审计中已被判定为不成立。

### HIPAA — BAA 不是可选项

未经签署 Business Associate Agreement（BAA）就不能将 PHI 发送给外部 AI 服务。三大超大规模云提供商的 LLM 平台（Bedrock、Azure OpenAI、Vertex）均提供 BAA。OpenAI 的直接 API 提供 BAA。Anthropic 的直接 API 也提供 BAA。在发送 PHI 之前必须确认有 BAA。

### SOC 2 Type II

Type I：控件已设计并记录。  
Type II：控件在 6-12 个月内有效运行并接受审计。

到 2026 年，B2B 采购默认要求 Type II。Type I 可以作为起点；Type II 是准入门槛。

常见的审计驱动项：访问日志（谁看过什么）、变更管理（如何部署）、风险评估（季度）、事件响应（是否测试）。可重用 Phase 17 · 25 中的审计日志输出。

### 跨框架映射

一条访问控制策略可以满足多个框架控制：

| Control | Frameworks |
|---------|-----------|
| Access logging | ISO 27001 A.5.15-5.18、GDPR 第 32 条、HIPAA §164.312(a) |
| Change management | ISO 27001 A.8.32、PCI DSS 要求 6、HIPAA 违规通知范围 |
| Encryption in transit | ISO 27001 A.8.24、GDPR 第 32 条、HIPAA §164.312(e) |
| Secrets management | ISO 27001 A.8.19、PCI DSS 要求 8、SOC 2 CC6.1 |

合规工具（Drata、Vanta、Secureframe）可自动化这种映射。在规模化时非常值得投入成本。

### ISO 42001 — 新兴标准

于 2023 年末发布。作为 AI 治理框架，正与 ISO 27001 一并成为采购要求，覆盖风险管理、数据质量、透明度、人类监督等。

### OpenAI 的参考配置

OpenAI 保持 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及用于 ChatGPT 支付组件的 PCI-DSS。这大致代表 2026 年的企业级基础要求。

### 你应该记住的数字

- 欧盟 AI 法案罚款：高风险义务：最高 €15M / 3%（Art. 99(4)）；禁止性行为：最高 €35M / 7%（Art. 99(3)）。  
- 欧盟 AI 法案高风险强制执行：2026 年 8 月 2 日。  
- 记录中的最大 AI 专项 GDPR 罚款：€30.5M，Clearview AI（荷兰 DPA，2024 年 9 月）。  
- 最大的 LLM 专项 GDPR 罚款：€15M，OpenAI（意大利 Garante，2024 年 12 月；2026 年 3 月在上诉中被撤销）。  
- SOC 2 Type II 时间窗口：控件需在 6-12 个月内运行。  
- 科罗拉多州 AI 法案生效日：2026 年 6 月 30 日（因 SB25B-004 从 2026 年 2 月延后）。

## 使用方法

`code/main.py` 是一个合规映射的电子表格脚本（Python）——给定一个控制项，列出它满足的框架。

## 交付物

本课程生成 `outputs/skill-compliance-matrix.md`。根据客户细分和地域，指定所需的框架和控制项。

## 练习

1. 你的第一个企业客户要求 SOC 2 Type II、HIPAA BAA、欧盟 AI 法案声明。赢得该交易的最低可行合规姿态是什么？  
2. 将三个假设的 LLM 产品按欧盟 AI 法案的风险分层分类。高风险状态下需要哪些变化？  
3. 你不小心在没有 BAA 的情况下将 PHI 发送给了某个提供商。走一遍事件响应流程。  
4. 论证 ISO 42001 对于中端市场 AI 厂商在 2026 年是否“必要”。  
5. 将你的 LLM 审计日志字段（Phase 17 · 25）映射到至少三个框架控制。

## 术语表

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SOC 2 Type II | "audited controls" | 在 6-12 个月内运行的控件，需独立鉴证 |
| HIPAA BAA | "healthcare contract" | Business Associate Agreement；处理 PHI 必需 |
| GDPR | "EU privacy" | 在 2026 年的可辩护标准是推理阶段的实时 PII 脱敏 |
| EU AI Act | "EU AI rules" | 高风险强制执行自 2026 年 8 月；€15M / 3%（高风险义务）；€35M / 7%（禁止性行为） |
| Colorado AI Act | "US AI state law" | 2026 年 6 月 30 日生效（受 SB25B-004 延期）；要求影响评估 |
| ISO 42001 | "AI governance" | 新兴的 AI 风险与透明度治理框架 |
| ISO 27001 | "security ISMS" | 信息安全管理体系基线 |
| Conformity assessment | "EU AI doc package" | 高风险要求：文档、测试、日志记录 |
| Cross-framework mapping | "one control, many frames" | 单一策略可满足多个框架控制 |

## 参考阅读

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — 参考合规模型。  
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)  
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)  
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — 原始文本。  
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — 原始文本。  
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI 管理体系标准。