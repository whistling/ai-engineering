# Red-Team Tooling — Garak, Llama Guard, PyRIT

> Three production tools frame the 2026 red-team stack. Llama Guard (Meta) — a Llama-3.1-8B classifier fine-tuned on 14 MLCommons hazard categories; the 2025 Llama Guard 4 is a 12B natively multimodal classifier pruned from Llama 4 Scout. Garak (NVIDIA) — open-source LLM vulnerability scanner with static, dynamic, and adaptive probes for hallucination, data leakage, prompt injection, toxicity, and jailbreaks. PyRIT (Microsoft) — multi-turn red-team campaigns with Crescendo, TAP, and custom converter chains for deep exploitation. Llama Guard 3 is documented in Meta's "Llama 3 Herd of Models" (arXiv:2407.21783); Llama Guard 3-1B-INT4 in arXiv:2411.17713; Garak's probe architecture in github.com/NVIDIA/garak. These tools are the 2026 production interface between red-team research (Lessons 12-15) and deployment (Lesson 17+).

**Type:** 构建  
**Languages:** Python（stdlib、工具架构模拟器与 Llama Guard 风格分类器模拟）  
**Prerequisites:** 阶段 18 · 第 12-15 课（越狱与 IPI）  
**Time:** ~75 分钟

## 学习目标

- 描述 Llama Guard 3/4 在安全堆栈中的位置：输入分类器、输出分类器，还是两者兼有。  
- 列出 14 个 MLCommons 危害类别，并举出一个不明显的例子（Code Interpreter Abuse，代码解释器滥用）。  
- 描述 Garak 的探针架构：probes、detectors、harnesses。  
- 描述 PyRIT 的多轮攻防活动结构以及它如何与 Garak 探针组合使用。

## 问题背景

第 12-15 课展示了攻击面。生产部署需要可重复、可扩展的评估。到 2026 年，有三款工具占主导地位：Llama Guard（防御分类器）、Garak（扫描器）、PyRIT（活动编排器）。每个工具针对红队生命周期中的不同层面。

## 概念说明

### Llama Guard（Meta）

Llama Guard 3 是基于 Llama-3.1-8B 微调的输入/输出分类器，覆盖 MLCommons AILuminate 的 14 个类别：
- 暴力犯罪、非暴力犯罪、性相关、CSAM、诽谤  
- 专业建议、隐私、知识产权、非定向武器、仇恨  
- 自杀/自伤、性内容、选举、代码解释器滥用（Code Interpreter Abuse）

支持 8 种语言。用法：可置于 LLM 之前（输入审查）、置于 LLM 之后（输出审查），或两端同时部署。两种用法会产生不同的训练分布 —— Llama Guard 3 以单一模型发布以同时处理两类任务。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，移动 CPU 上约 30 tokens/s）是量化的边缘版本。

Llama Guard 4（2025 年 4 月）为 12B 原生多模态模型，从 Llama 4 Scout 裁剪而来。它用一个能够接收文本 + 图像的分类器，取代了 8B 文本与 11B 视觉的先前版本。

### Garak（NVIDIA）

开源的漏洞扫描器。架构包括：
- **Probes（探针）。** 用于触发幻觉、数据泄露、提示词注入、有毒内容和越狱的攻击生成器。分为静态（固定提示）、动态（生成式提示）、自适应（根据目标输出响应）。  
- **Detectors（检测器）。** 将模型输出与预期失败模式进行评分 —— 是否有毒、是否泄露、是否越狱等。  
- **Harnesses（运行框架）。** 管理 probe-detector 配对、执行活动并生成报告。

TrustyAI 将 Garak 与 Llama-Stack 防护结合（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器），用于端到端的受保护目标评估。分层评分（Tier-based scoring，TBSA）取代二元通过/失败 —— 同一探针下模型在严重性层级 3 上可以通过，而在层级 5 上失败。

Garak 的探针架构可见于 github.com/NVIDIA/garak。

### PyRIT（Microsoft）

Python Risk Identification Toolkit。用于多轮红队攻防活动。核心组件：
- **Converters（转换器）。** 将种子提示转换为其他形式 —— 释义、编码、翻译、角色扮演等。  
- **Orchestrators（编排器）。** 运行活动：Crescendo（逐级升级）、TAP（分支）、RedTeaming（自定义循环）。  
- **Scoring（评分）。** 使用 LLM 作为裁判或分类器作为裁判。

PyRIT 是 Garak 的更重量级同类工具。Garak 通常运行数千个单轮探针；PyRIT 运行设计为突破特定失败模式的深度多轮攻防活动。

### 技术栈部署建议

在模型的输入与输出两端部署 Llama Guard。夜间运行 Garak 做回归测试。发布前运行 PyRIT 做深度攻防演练。这是 2026 年大多数生产部署的默认配置。

### 评估陷阱

- **裁判者身份。** 三款工具都可以使用 LLM 作为裁判；裁判校准会显著影响报告的攻击成功率（ASRs，第 12 课）。在使用工具时应明确裁判者。  
- **探针陈旧。** 随着模型对已知探针的修补，Garak 的探针会失效。自适应探针（如 PAIR 形式）比静态探针衰减更慢。  
- **Llama Guard 在良性内容上的误报率（FPR）。** 早期版本对政治和 LGBTQ+ 内容存在过度标记；Llama Guard 3/4 的校准有所改进，但并未针对每次部署进行专门校准。

### 在阶段 18 中的位置

第 12-15 课为攻击家族；第 16 课关注生产工具链；第 17 课（WMDP）是双用途能力的评估；第 18 课是将这些工具纳入策略结构的前沿安全框架。

## 使用方法

`code/main.py` 构建了一个玩具的 Llama Guard 风格分类器（基于关键字 + 14 类别的语义特征）、一个玩具 Garak harness（探针-检测器循环），以及一个 PyRIT 风格的多轮转换器链。你可以将这三种工具对一个模拟目标运行，并观察不同的覆盖特征。

## 交付物

本课会产生 `outputs/skill-red-team-stack.md`。给定一个部署描述，该文档会指出三款工具中哪些合适、每款工具需要如何配置、以及应该运行的回归测试频率。

## 练习

1. 运行 `code/main.py`。比较 Llama-Guard 风格分类器对单轮攻击与多轮攻击的检测率。  
2. 实现一个新的 Garak 探针：一个 base64 编码的有害请求。测量其被 Llama-Guard 风格分类器检测到的情况。  
3. 在 PyRIT 风格的转换器链中扩展一个“先翻译成法语，再释义（paraphrase）”的转换器。重新测量攻击成功率。  
4. 阅读 Llama Guard 3 的危害类别清单。识别两个类别，其训练数据在现实中会对合法开发者内容产生高误报率。  
5. 比较 Garak 与 PyRIT 的设计原则。论证在何种部署场景下各自是合适的工具。

## 术语表

| 术语 | 大家怎么说 | 实际意思 |
|------|-----------|---------|
| Llama Guard | "the classifier" | 基于微调的 Llama-3.1-8B/4-12B 安全分类器，覆盖 14 个危害类别 |
| Garak | "the scanner" | NVIDIA 的开源漏洞扫描器；包含 probes、detectors、harnesses |
| PyRIT | "the campaign tool" | Microsoft 的多轮红队编排器；包含 converters、orchestrators、scoring |
| Prompt-Guard | "the small classifier" | Meta 的 86M 提示注入分类器，通常与 Llama Guard 配合使用 |
| TBSA | "tier-based scoring" | Garak 的分层评分方案，用以替代二元的通过/失败 |
| Converter chain | "paraphrase + encode + ..." | PyRIT 用于构建多步攻击的组合原语 |
| MLCommons hazard categories | "the 14 taxonomies" | Llama Guard 针对的行业标准危害分类法 |

## 拓展阅读

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 8B 分类器  
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — 量化的移动端分类器  
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — 扫描器仓库与文档  
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — 攻防活动工具包