# Bias and Representational Harm in LLMs

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed (Computational Linguistics 2024, arXiv:2309.00770). Foundational 2024 survey distinguishing representational harms (stereotypes, erasure) from allocational harms (unequal resource distribution) and categorizing evaluation metrics as embedding-based, probability-based, or generated-text-based. 2024-2025 empirical: An et al. (PNAS Nexus, March 2025) measure intersectional gender x race bias across GPT-3.5 Turbo, GPT-4o, Gemini 1.5 Flash, Claude 3.5 Sonnet, Llama 3-70B on automated resume evaluation for 20 entry-level jobs. WinoIdentity (COLM 2025, arXiv:2508.07111) introduces uncertainty-based fairness evaluation for intersectional identities. Yu & Ananiadou 2025 identify gender neurons in MLP layers; Ahsan & Wallace 2025 use SAEs to reveal clinical racial bias; Zhou et al. 2024 (UniBias) manipulates attention heads for debiasing. Meta-critique (arXiv:2508.11067): 10-year literature disproportionately focuses on binary-gender bias.

**Type:** 构建  
**Languages:** Python（标准库，玩具级基于嵌入的偏差探测器）  
**Prerequisites:** Phase 05（词嵌入），Phase 18 · 01（指令遵循）  
**Time:** ~60 分钟

## Learning Objectives

- 定义表征性伤害（representational harm）与分配性伤害（allocational harm），并举出各自一个在 LLM 部署中的例子。  
- 说出 Gallegos 等人 2024 所述的三类评估指标，并描述每类中的一个度量方法。  
- 描述交叉性（intersectionality）以及为什么 WinoIdentity 的基于不确定性的公平性测量能弥补单轴偏差评估的空白。  
- 描述两种针对偏差的机械可解释性方法（性别神经元、SAE 特征、注意力头操控）。

## The Problem

前面的课程涵盖了蓄意伤害（越狱、策划）和安全治理。偏差是一种不带意图便会出现的伤害——来源于训练数据分布、提示词表述、累积的设计决策等。测量与减少偏差在方法论上与对抗鲁棒性不同，是一个独立的挑战。

## The Concept

### Representational vs allocational

- **Representational harm（表征性伤害）。** 刻板印象、抹除、贬低性的描绘。一个把护士描绘为仅为女性的 LLM，就在产生表征性伤害。  
- **Allocational harm（分配性伤害）。** 不平等的物质性结果。一个系统性给黑人求职者的简历打分较低的 LLM，就在产生分配性伤害。

两者并不相同。模型可能在“表征上无偏”（呈现多样化描绘），但在“分配上有偏”（给出不均等的推荐）。评估需要同时覆盖两方面。

### Three evaluation-metric categories (Gallegos et al. 2024)

- **Embedding-based（基于嵌入）。** 在 RLHF 之前的嵌入上做 WEAT 风格测试。衡量身份词与属性词之间的统计关联。限制：测量的是表示而不是行为。  
- **Probability-based（基于概率）。** 比较支持刻板印象的完成与违背刻板印象的完成的对数似然。发生在解码器端，能够捕捉部分行为性偏差。  
- **Generated-text-based（基于生成文本）。** 在生成文本上的下游任务测量：简历评分、推荐信写作、对话等。生态有效性最高，但最难复现。

### Intersectionality（交叉性）

仅在“性别”上做偏差评估会遗漏只在（性别, 种族）对上触发的偏差。An 等人 2025 发现在简历评分中，GPT-4o 对黑人女性的惩罚性评分高于对黑人男性或白人女性的评分。单轴评估无法捕捉这种效应。

WinoIdentity（COLM 2025）提出了基于不确定性的交叉性公平性测量。它衡量模型在不同交叉身份元组上的输出不确定性是否存在差异——而不仅仅关注点预测。这能发现模型在各组上错误率相同但不确定性不同的情形，这种差异会影响下游的分配行为。

### Mechanistic approaches（机械性方法）

2024–2025 的可解释性工作使得偏差成为可机械化干预的对象：

- **Gender neurons（性别神经元，Yu & Ananiadou 2025）。** 在 MLP 层中存在与性别特定行为相关联的特定神经元。切除这些神经元可以在有限能力损失下降低性别差距指标。  
- **Clinical racial bias via SAEs（通过 SAE 发现临床种族偏差，Ahsan & Wallace 2025）。** 稀疏自编码器（SAE）将内部表示分解为可解释的维度；可以识别并抑制与种族相关的特征。  
- **UniBias（Zhou et al. 2024）。** 通过操控注意力头进行零样本去偏。特定注意力头会放大对身份类别的敏感性；将这些头归零或重新加权可以在无需微调的情况下减少偏差。

### The meta-critique（元批评）

对过去十年文献的回顾（arXiv:2508.11067, 2025）发现该领域过度聚焦于二元性别偏差。其他轴线——残疾、宗教、迁移身份、多语言身份——受到的关注远少于性别。元批评指出，狭窄的关注会通过忽视某些群体而伤害边缘化群体：在二元性别上表现良好的去偏方案，可能在没人检查的维度上表现很差。

### Where this fits in Phase 18

课程 20–21 正式覆盖偏差与公平性。课程 22 讲隐私。课程 23 讲水印。这些属于用户伤害层，补充之前的欺骗/安全层内容。

## Use It

`code/main.py` 构建了一个玩具级基于嵌入的偏差探测器：在一个简单的共现嵌入上测量身份词与属性词之间的 WEAT 风格距离。你可以注入一个偏差并观察指标触发；应用一个简单的去偏操作并观察部分恢复。

## Ship It

本课产出 `outputs/skill-bias-eval.md`。给定一个模型卡或公平性声明，本评审对三类评估（嵌入、概率、生成文本）、交叉性覆盖情况，以及任何去偏干预的机制进行审计。

## Exercises

1. 运行 `code/main.py`。报告在去偏步骤前后得到的 WEAT 风格偏差分数。解释为什么该指标不会降为零。  
2. 将探针扩展为交叉性测试：（gender, race）x（career, family）。报告跨轴偏差分数。  
3. 阅读 An et al. 2025（PNAS Nexus）。指出他们报告的两个交叉性效应，这两者是单轴性别评估会遗漏的。  
4. Yu & Ananiadou 2025 识别了性别神经元。设计一个伪证实验（falsification experiment），以区分“这些神经元导致性别偏差”和“这些神经元与性别偏差相关但不致因”的情形。  
5. 元批评认为该领域过度聚焦二元性别。选择一个研究较少的轴线，描述一个针对它的表征性伤害测量协议。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| 表征性伤害（Representational harm） | "stereotypes / erasure" | 对某一群体的有偏见描绘 |
| 分配性伤害（Allocational harm） | "unequal decisions" | 对某一群体造成不公平的物质性结果 |
| WEAT | "the embedding test" | Word Embedding Association Test；基于共现的偏差探针 |
| 交叉性（Intersectionality） | "combined identity effects" | 在多条身份轴交叉处出现的偏差 |
| 性别神经元（Gender neurons） | "MLP bias neurons" | 与性别相关行为激活相关联的特定神经元 |
| SAE feature | "interpretable dimension" | 稀疏自编码器（SAE）识别的可解释特征；对机械性偏差分析有用 |
| UniBias | "attention-head debiasing" | 通过重加权/归零注意力头实现的零样本去偏 |

## Further Reading

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — 规范性综述  
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) — 五模型交叉性研究  
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — 新基准，基于不确定性的交叉性公平性  
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612) — 零样本去偏研究