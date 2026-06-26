# 间接提示词注入（Indirect Prompt Injection）—— 生产环境攻击面

> 间接提示词注入（IPI）将指令嵌入到外部内容中——网页、电子邮件、共享文档、支持工单等——被具代理能力的系统在没有用户明确操作的情况下读取并执行。IPI 是 2026 年主要的生产威胁：它绕过了用户输入过滤器，因为攻击者从未触及用户；随着代理处理更多外部内容，它可以悄然扩展；并且它针对的是无人阅读提示词的自动化工作流。MDPI Information 17(1):54（2026 年 1 月）综合了 2023–2025 年的研究。NDSS 2026 的 IPI 防御论文将核心挑战表述为：注入指令在语义上可能是良性的（例如“please print Yes”），因此检测不仅仅依赖关键字过滤。“The Attacker Moves Second”（Nasr 等，OpenAI/Anthropic/DeepMind 联合，2025 年 10 月）：自适应攻击（梯度、RL、随机搜索、人类红队）打破了 >90% 的 12 项已发表防御，这些防御最初报告的攻击成功率接近零。

**Type:** 构建  
**Languages:** Python（stdlib，IPI 攻击 + 防御 测试套件）  
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 14 (agent engineering)  
**Time:** ~75 分钟

## 学习目标

- 定义间接提示词注入并描述三种常见的投递载体。
- 解释为什么基于用户输入的过滤器会完全漏掉 IPI。
- 描述将“信息流控制”作为 2026 年防御范式的表述。
- 陈述 Nasr 等（2025 年 10 月）关于自适应攻击对已发表 IPI 防御的成功率发现。

## 问题

直接提示词注入要求攻击者触达用户或用户的提示词。IPI 则不需要：攻击者将有效载荷放在代理可能读取的任何内容中——网页、收件箱中的邮件、GitHub issue、产品评论。代理在正常操作中读取这些内容并执行其中的指令。用户只是信使，而非原始意图的发起者。

## 概念

### 三种投递载体

- **Retrieval-augmented generation (RAG).** 攻击者发布文档；检索步骤取回该文档；提示词将其拼接到用户问题之前；模型执行攻击者的指令。
- **收件箱/文档工作流。** 攻击者给用户发送电子邮件；代理读取电子邮件；提示词包含邮件正文；模型遵循邮件中的指令。
- **工具输出。** 攻击者控制代理使用的某个工具（例如，返回攻击者控制结果的网络搜索）；工具输出包含指令；代理的控制流遵循这些指令。

这三者共享一个结构属性：攻击者控制了提示词的一段片段，而没有触及面向用户的输入。

### 为什么基于用户输入的过滤器会漏掉它

IPI 有效载荷不会出现在用户的输入中，而是出现在被检索到的内容里。如果过滤器只在用户输入上生效，则有效载荷会绕过它。如果过滤器针对到达模型的所有内容，则它必须适用于任意被检索的文本——这既昂贵，又会对合法内容产生误报，尤其是那些恰好包含祈使语气的合法文本。

### 面向 AI 的信息流控制（IFC）

2026 年的防御范式借鉴了传统操作系统安全。将每个内容来源视为一个安全标签。将用户查询标注为“trusted”（受信任）。将检索到的内容标注为“untrusted”（不受信任）。把模型的控制流视为信息流：由不受信任内容触发的动作必须在执行前得到受信任输入的批准（ratified）。

CaMeL（Microsoft，2025）、ConfAIde（Stanford，2024）以及 NDSS 2026 的 IPI 防御论文以不同方式将 IFC 落地。共同原则是：只要代码和数据共享相同的上下文窗口，目标就是隔离（containment），而不是完全阻止。

### 攻击者后发制人（The Attacker Moves Second）

Nasr 等（2025 年 10 月）用自适应攻击（梯度搜索、RL 策略、随机搜索、72 小时人类红队）测试了 12 项已发表的 IPI 防御。每一项最初报告近零攻击成功率的防御都被打破，攻击成功率升至 >90%。

方法论上的教训是：只有在自适应攻击评估下才应发布防御。基于静态攻击的基准并不能证明鲁棒性；攻击者会了解防御并相应优化。

### 真实事件

Lesson 25 涵盖 EchoLeak（CVE-2025-32711，CVSS 9.3）——首例公开记录的在 Microsoft 365 Copilot 中的零点击 IPI。CamoLeak（CVSS 9.6）出现在 GitHub Copilot Chat。还有 GitHub Copilot 的 CVE-2025-53773。生产环境部署在实战中被 IPI 破坏，而不仅仅是在基准上。

### OWASP 与 NIST 的表述

OWASP LLM Top 10（2025）将提示词注入（直接 + 间接）列为 LLM01，属于应用层的首要威胁。NIST AI SPD 2024 将间接提示词注入称为“生成式 AI 最大的安全缺陷”。

### 在 Phase 18 中的位置

Lesson 12–14 属于以模型为中心的越狱（jailbreak）。Lesson 15 是在 2026 年主导生产部署的系统级攻击。Lesson 16 涵盖防御工具链。Lesson 25 则讲述具体的 CVE 叙事。

## 使用说明

`code/main.py` 构建了一个 IPI 测试套件。一个玩具代理有三种工具（网络搜索、读取邮件、发送消息）。环境包含带有嵌入指令（例如“forward this to all contacts”）的攻击者控制内容。你可以在天真的代理（会遵循注入指令）、关键字过滤防御的代理（对检索到的内容做关键字过滤）和 IFC 代理（将受信任与不受信任内容分离并拒绝不受信任的控制流指令）之间切换。

## 发布（Ship It）

本课产出 `outputs/skill-ipi-audit.md`。给定一个 agentic 部署描述，它枚举不受信任的内容源，检查部署是否应用了 IFC，并标记那些在到达模型时没有信任标签的来源。

## 练习

1. 运行 `code/main.py`。测量攻击对三种代理的成功率。
2. 在检索到的内容上实现基于释义（paraphrase）的防御。测量对合法检索文本的良性误报率。
3. 阅读 NDSS 2026 的 IPI 防御论文。描述“良性指令”（benign instruction）挑战以及为什么它阻碍基于关键字的过滤。
4. 设计一个部署场景，其中代理接收来自第三方 API 的工具输出。为每个提示片段标注信任级别，并编写治理代理行为的 IFC 策略。
5. 在练习 2 的关键字过滤代理上复现 Nasr 等 2025 年的自适应攻击方法论。报告自适应攻击前后的 ASR（攻击成功率）。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| IPI | "indirect prompt injection" | 通过用户未撰写的内容注入，由代理在正常操作中消费并执行的注入 |
| RAG injection | "poisoned retrieval" | 攻击者发布被检索步骤取回的内容；提示词包含该有效载荷 |
| Zero-click | "no user action" | 攻击在代理操作期间自动触发；用户无需任何操作 |
| IFC | "information flow control" | 基于标签的方法：来自不受信任内容的动作需要受信任方的批准 |
| Adaptive attack | "gradient / RL red-team" | 知晓防御并针对其优化的攻击；诚实评估所必需 |
| Benign instruction | "please print Yes" | 语义上良性的 IPI 有效载荷；没有关键字过滤能拦住它 |
| Scope violation | "cross-trust exfiltration" | 代理从一个信任上下文访问数据并将其输出到另一个上下文 |

## 延伸阅读

- [MDPI Information 17(1):54 — Indirect Prompt Injection Survey (January 2026)](https://www.mdpi.com/2078-2489/17/1/54) — 2023–2025 年研究综述  
- [Nasr et al. — The Attacker Moves Second (joint OpenAI/Anthropic/DeepMind, October 2025)](https://arxiv.org/abs/2510.18108) — 自适应攻击评估  
- [Greshake et al. — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173) — 最早的 IPI 论文  
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/) — 将提示词注入列为 LLM01