# STaR, V-STaR, Quiet-STaR — Self-Taught Reasoning

> 最小的自我改进循环就藏在推理里。模型生成思维链，保留那些得到正确答案的链，然后在这些链上进行微调。这就是 STaR。V-STaR 增加了一个验证器以便在推理时能做更好的选择。Quiet-STaR 则把推理压缩到每个 token 之前。三者都有效，但都不是魔法 —— 这个循环会保留任何碰巧得到正确答案的捷径推理。

**Type:** 学习  
**Languages:** Python（标准库，bootstrap-loop 模拟器）  
**Prerequisites:** Phase 13 · 01-03 (推理与思维链), Phase 15 · 01 (长时程框架)  
**Time:** ~60 分钟

## 问题

教模型推理的直观办法是收集人类撰写的推理痕迹（reasoning traces）。这是昂贵、缓慢且受限于人类愿意写多少高质量思维链的数量。

STaR（Self-Taught Reasoner，Zelikman 等，2022）提出：如果模型自己写推理并根据已知答案给自己打分，会怎样？循环如下：

1. 采样一个推理轨迹和答案。
2. 如果最终答案正确，就保留该推理轨迹。
3. 在保留的轨迹上微调模型。
4. 重复以上步骤。

它有效。GSM8K 和 CommonsenseQA 在没有新的人类标注下都得到了提升。但这个循环有一个内在偏差：任何产生正确答案的推理都会被保留，不管推理本身是否合理。V-STaR（Hosseini 等，2024）用一个学习到的验证器来修补这一点；Quiet-STaR（Zelikman 等，2024）将这个想法推广到了按 token 的内部推理。

## 概念

### STaR：对“有效”进行自举

从一个具有弱推理能力的基础模型开始。对于每个训练问题，采样一个推理和答案。如果答案与标签匹配，就保留（问题、推理、答案）三元组。在保留集合上微调模型，然后重复。

一个关键的变体很重要。如果模型永远不能把某个问题答对，循环就无法在该问题上学习。STaR 加入了 **rationalization（合理化重试）**：对于模型失败的问题，注入正确答案作为提示并重新提示模型生成能够导出该答案的推理。被合理化的推理会加入训练集。

原始论文中的结果（Zelikman 等，2022）：在反复的 STaR 回合与合理化后，GPT-J 基线模型在 GSM8K 上从 5.8% 提升到 10.7% —— 大约绝对提升 5 个百分点。在 CommonsenseQA 上，经过 STaR 训练的 GPT-J 6B 达到了 72.5%，可与通过人工标注思维链微调的 GPT-3 175B（约 73%）相当 —— 也就是说用自我生成的推理可以替代大约 30 倍规模的模型与人工标注。

### V-STaR：用 DPO 训练验证器

STaR 会丢弃错误的推理。Hosseini 等（2024）观察到这些也可以作为数据：每一对（推理，是否正确）都能训练一个验证器。他们用 Direct Preference Optimization 在正确与错误解之间训练一个排序器。在推理时采样 N 个推理并选出验证器打分最高的那个。

他们报告的增益：在 GSM8K 和 MATH 上相比之前的自我改进基线提高了 +4 到 +17 个百分点，大部分收益来自在推理时使用验证器进行选择，而不是用验证器来对生成器做额外微调。

### Quiet-STaR：按 token 的内部推理

Zelikman 等（2024）提出：如果模型在每个 token 位置学习生成一个短小的内部推理（而不是仅在问题和答案之间生成），会怎样？Quiet-STaR 训练模型在每个预测 token 之前发出一个隐藏的“思考”，然后通过学习到的权重把带有“思考”的预测与基础预测混合。

结果：Mistral 7B 在零样本下在 GSM8K 上从 5.9% 提升到 10.9%；在 CommonsenseQA 上从 36.3% 提升到 47.2%，且无需任务特定的微调。模型学会了“何时思考”——困难的 token 有更长的内部推理；简单的几乎不需要。

### 为什么三者都有安全隐患

这三种方法都使用最终答案作为梯度信号。通过有缺陷的推理（利用捷径、猜测或使用不具备泛化性的模式）达到正确答案的推理会被正向强化。在训练分布内这些捷径有效，但在分布外就会悄然失效。

V-STaR 的验证器通过学习对推理进行排序来缓解这一问题，但验证器是在相同标签集上训练的。它可能学会偏好格式良好的错误推理而不是诚实的不确定。更安全的设计是将 STaR 风格的数据与 (a) 过程监督的奖励模型（奖励中间步骤，而不仅是答案）和 (b) 能打破简单捷径的留出 OOD 评估 相结合。

### 比较

| 方法 | 训练信号 | 推理成本 | 数据浪费 | 已知失败模式 |
|---|---|---|---|---|
| STaR | 如果正确则保留 (rationale, answer) | 1x | 丢弃所有错误推理 | 捷径推理 |
| STaR + rationalization | 上述 + 注入正确答案的重试 | 1x | 更少 | 被合理化的推理可能不可信 |
| V-STaR | STaR + 用两类样本训练的 DPO 验证器 | Nx（best-of-N） | 最小 | 验证器可能强化自信的错误 |
| Quiet-STaR | 按 token 的推理 + 混合权重 | 1.5-3x | 最小 | 仍然是以答案为条件的梯度 |

### 它在 2026 年技术栈中的位置

STaR 已经过时。然而这种模式在 2025-2026 年屡见不鲜。可验证数学问题上的强化学习（DeepSeek-R1、Kimi-k1.5、o1）可以看作是以 STaR 的答案条件梯度信号放大后的版本。过程奖励模型（Lightman 等，2023；OpenAI 的 “Let's verify step by step”）是过程监督的替代品。AlphaEvolve（第 3 课）是针对代码的 STaR，用程序评估器替代标签。Darwin Godel Machine（第 4 课）是用于智能体自我脚手架的 STaR。

理解 STaR 会让以上这些方法变得直观。它是最小可行的自我改进循环。

```figure
reflection-loop
```

## 使用方法

`code/main.py` 在一个玩具算术任务上运行一个模拟的 STaR 循环。你可以观察到：

- 精度如何随自举回合上升。
- 捷径如何潜入：模拟器包含一个“懒惰”推理类，它 40% 的时候得到正确答案但泛化性差。观察 STaR 是否会保留它们。
- 验证器（V-STaR 风格）如何在推理时有帮助，但无法完全删除训练过程中引入的捷径。

## 部署

`outputs/skill-star-loop-reviewer.md` 帮助你在训练前审计一个拟议的自学推理流水线。

## 练习

1. 运行模拟器。将捷径频率设置为 0，然后设置为 0.4。即便两次运行在训练分布上都超过 90%，最终精度会差多少？

2. 给模拟器添加一个留出的 OOD 测试。从不同的分布中抽取问题，并在训练分布与 OOD 集上评估自举得到的模型。量化两者间的差距。

3. 阅读 Quiet-STaR 论文（arXiv:2403.09629）第 3 节。用三句话分别解释“思考结束（end-of-thought）”标记和混合权重头。

4. 将 STaR 的“答对即保留”筛选与一个对每一步独立奖励的过程监督替代方案进行比较。指出标注成本差异和可能的质量差异。

5. 设计一种评估方法来发现部署模型中的捷径推理。它不必完美 —— 只要能打破 STaR 循环可能强化的最简单捷径即可。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|---|---|---|
| STaR | "Self-Taught Reasoner" | 在模型自生成并得到正确答案的推理上微调；重复进行 |
| Rationalization | "Hinted retry" | 注入正确答案并对基础模型失败的问题重新提示以生成推理 |
| V-STaR | "Verifier STaR" | 用 DPO 在正确与错误推理上训练验证器，在推理时用于选择 |
| Quiet-STaR | "Per-token rationales" | 在每个 token 位置生成隐藏思考；与基础预测混合 |
| Answer-conditioned gradient | "Outcome-based signal" | 训练循环奖励最终答案，而非推理步骤 |
| Process reward model | "Step-level verifier" | 在每一步正确性上训练的奖励模型，与 STaR 的做法形成对比 |
| Shortcut rationale | "Right answer, wrong reasoning" | 通过不具泛化性的模式到达标签的推理；STaR 会保留这些 |

## 延伸阅读

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — 原始论文。  
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — 为推理时选择增加了基于 DPO 的验证器。  
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — 按 token 的内部推理方法。  
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — 过程奖励模型，是一种替代的梯度信号。  
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在可验证任务上的强化学习，把 STaR 扩展到前沿训练。