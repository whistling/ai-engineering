# The Direct Preference Optimization Family

> Rafailov 等人（2023）展示了 RLHF 的最优解在偏好数据意义下有闭式解，因此可以跳过显式奖励模型，直接优化策略。这个洞见催生了一系列方法 —— IPO、KTO、SimPO、ORPO、BPO —— 它们各自修复了 DPO 的不同失效模式。到 2026 年，直接对齐算法（DAAs）在前沿的后训练运行中比 PPO 更常见。但来自 Lesson 2 的过度优化曲线仍然适用：DAAs 并未逃脱 Goodhart，只是把问题移动到了新的表面。

**Type:** 学习  
**Languages:** Python（标准库，六变体偏好损失比较器）  
**Prerequisites:** Phase 18 · 01（InstructGPT）、Phase 18 · 02（奖励劫持）、Phase 10 · 08（DPO 基础）  
**Time:** ~75 分钟

## 学习目标

- 从带有 KL 项的 RLHF 最优解推导出 DPO 的闭式形式。  
- 说明 IPO、KTO、SimPO、ORPO、BPO 各自修复 DPO 的哪一种失败模式。  
- 区分“隐式奖励差距”（implicit reward gap）与“偏好强度”（preference strength），并解释为什么 IPO 的恒等映射重要。  
- 解释 Rafailov 等人（NeurIPS 2024）为什么证明尽管没有显式 RM，DAAs 仍然会过度优化。

## 问题

RLHF 目标（Lesson 1）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此奖励可以由最优策略与参考策略的比率隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

把它代入 Bradley-Terry 偏好似然中，分区函数 `Z(x)` 会被消去，因为它只依赖于 `x`。剩下的就是仅在策略参数上的损失 —— 不需要奖励模型。这就是 DPO。

但有个问题：推导假设最优解可达、偏好数据为同分布（in-distribution）、且参考策略是正确的锚点。现实中这些都不完全成立。每个家族成员修复了不同的被违反的假设。

## 概念

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出的问题：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无上界的。微小的偏好也能产生任意大的差距。  
- 损失会把被选择和被拒绝的对数概率朝相反方向推动。它可以在被拒绝项下降更快的情况下把被选择项的绝对对数概率压低。这就是“Degraded Chosen Response”（被选择响应退化）现象。  
- 分布外（out-of-distribution）的偏好（例如罕见-罕见 对 vs 罕见-罕见 对）会产生任意的隐式奖励。

### IPO (Azar et al., 2024)

Identity Preference Optimization 用对偏好概率的恒等映射替代了 log-sigmoid。损失变为对一个有界目标的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边际由 `1/(2 beta)` 有界。偏好强度与隐式奖励差距成比例。不会爆炸。

### KTO (Ethayarajh et al., 2024)

Kahneman-Tversky Optimization 完全放弃成对结构。给定一个单独的标注输出以及二元的“可取”或“不可取”信号，它映射为一种前景理论（prospect-theory）的效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对损益使用不同权重（损失厌恶）。优点：可以使用未配对的数据，这类数据更为丰富。

### SimPO (Meng et al., 2024)

Simple Preference Optimization 将训练信号与生成对齐。完全移除参考策略，并按长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

使用一个边际 `gamma` 来稳定训练。长度归一化消除了利用 DPO 长度偏差的激励（按构造，较长的 `y_w` 会带来更大的对数概率差）。

### ORPO (Hong et al., 2024)

Odds-Ratio Preference Optimization 在标准 SFT 负对数似然上添加了一个偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有参考策略 —— SFT 项充当正则化器。从基础模型到对齐模型在单阶段内训练。没有单独的 SFT 检查点。

### BPO (ICLR 2026 submission, OpenReview id=b97EwMUWu7)

识别出 Degraded Chosen Responses 问题：DPO 保持了排序 `y_w > y_l`，但被选择响应 `y_w` 的绝对对数概率可能下降。BPO 添加了一行修正，惩罚被选择响应的向下移动。在 Llama-3.1-8B-Instruct 的数学推理任务上，报告比 DPO 高 +10.1% 的准确率。

### 通用结论：DAAs 仍会过度优化

Rafailov 等人在“Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms”（NeurIPS 2024）中，用 DPO、IPO、SLiC 在多数据集、不同 KL 预算下训练策略。以真实奖励对 KL 的曲线呈现出与 Gao 等人相同的峰值-崩塌形状。隐式奖励在训练中对分布外样本进行查询；KL 正则无法稳定这一点。

DAAs 并未逃离 Goodhart。它们只是把受伤的表面从“奖励模型被过度优化”变成“参考策略比率被过度优化”。通用的修复 —— 更好的数据、集成（ensembles）、早停（early stopping） —— 对两者都适用。

### 如何选择（2026 年）

- 如果有大量成对偏好数据：使用 DPO（保守选择 beta），若出现长度偏差则用 SimPO。  
- 如果有未配对的二元反馈：KTO。  
- 如果想要从基础模型到对齐模型的单阶段流水线：ORPO。  
- 如果在 DPO 日志中看到被选择项的对数概率下降：BPO。  
- 如果偏好强度差异很大且 DPO 饱和：IPO。

每个团队会对这五种方法在一组任务上都跑一遍，然后按任务选最优者。不同任务（如数学推理与安全）没有理由有相同的最优解。

```figure
dpo-margin
```

## 使用示例

`code/main.py` 比较六个损失（DPO、IPO、KTO、SimPO、ORPO、BPO）在一个玩具偏好数据集上的表现，其中真实偏好强度因对而异。每个损失在相同的 500 对样本上进行优化，策略为一个小的 softmax 模型。绘制最终胜率、被选择对数概率漂移以及每种方法的隐式奖励分布。

## 发布建议

本课会生成 `outputs/skill-preference-loss-selector.md`。给定数据集统计（成对 vs 未配对、偏好强度是否可变、长度分布）和目标（单阶段还是先 SFT 再偏好），推荐一个偏好损失并报告它所防护的失败模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 与 BPO 的最终被选择对数概率下降情况。BPO 应保持较高的被选择绝对概率 —— 验证这一点。  

2. 修改偏好数据，使所有对的强度相等。六种方法中哪一种最稳健？哪一种会退化？解释 IPO 在此场景下的优势。  

3. 使被拒绝响应的平均长度为被选择响应的 2 倍。保持其它条件不变，数值展示 DPO 的长度利用（length exploitation）以及 SimPO 的修复效果。  

4. Rafailov 等人（NeurIPS 2024）声称 DAAs 会过度优化。重现一个单点实验：绘制被选择与被拒绝的 KL 差（chosen-minus-rejected KL divergence），观察在大 beta 时 DPO 的过度优化。  

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写出 BPO 在 DPO 上添加的那行一行修正。与 `code/main.py` 中的实现进行对照确认。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|----------|
| DPO | “没有奖励模型的 RLHF” | 从闭式 RLHF 最优解推导出的损失；仅涉及策略参数 |
| Implicit reward | “对数比率” | `beta * log(pi(y\|x) / pi_ref(y\|x))` — DPO 隐含的奖励 |
| IPO | “有界的 DPO” | 用恒等替换 log-sigmoid；隐式奖励差距被 `1/(2 beta)` 截断 |
| KTO | “未配对的 DPO” | 基于单标签的前景理论效用，带有损失厌恶 |
| SimPO | “无参考的 DPO” | 对数似然按长度归一化并加边际；不使用参考策略 |
| ORPO | “一阶段的 DPO” | NLL + 概率比（odds-ratio）偏好项；从基础模型单阶段训练 |
| BPO | “保持行为的 DPO” | 在 DPO 上加一项惩罚，防止被选择响应的绝对对数概率下降 |
| Degraded Chosen | “被选择项下降” | 只要被拒绝项下降更快，DPO 会降低被选择项的对数概率 |
| DAA | “直接对齐算法” | 任何跳过显式奖励模型的偏好损失方法 |

## 进一步阅读

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)  
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO  
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)  
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)  
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)  
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)  
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)