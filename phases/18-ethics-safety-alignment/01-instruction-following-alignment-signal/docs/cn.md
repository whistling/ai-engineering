# Instruction-Following as Alignment Signal

> 每一个随后对 RLHF 的批评都是针对这条管线的。在你研究优化压力如何扭曲代理目标之前，你必须先看到这个代理。InstructGPT（Ouyang 等，2022）定义了参考架构：在指令-响应对上进行监督微调（supervised fine-tuning），在成对偏好排名上训练一个奖励模型（reward model），并用带有 KL 惩罚项的 PPO 针对该奖励模型进行优化。一个 1.3B 的 InstructGPT 在人工偏好评估中被优先选择胜过了 175B 的 GPT-3。那一个结果是到 2026 年为止每个前沿实验室仍然部署 RLHF 形状的后训练管线的原因。

**Type:** 学习  
**Languages:** Python（标准库，玩具三阶段管线）  
**Prerequisites:** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)  
**Time:** ~45 分钟

## 学习目标

- 说出 InstructGPT 管线的三个阶段以及每个阶段使用的损失函数。
- 解释为什么 1.3B 的指令调优模型在人工偏好评估中胜过原始的 175B GPT-3。
- 说明第 3 阶段中的 KL 惩罚在保护什么，以及为何移除它会坍缩为寻模行为（mode-seeking）。
- 描述对齐税（alignment tax）以及 Ouyang 等人用来缓解它的 PPO-ptx 方法。

## 问题

预训练语言模型完成文本。它们并不“回答问题”。向 GPT-3 问“写一个反转列表的 Python 函数”时，你常常会得到另一个提示，因为大部分训练分布是网页文本，网页文本倾向于继续产生更多网页文本。模型在执行它的工作 —— 只是这个工作本身是错的。

每个认真对待这个问题的实验室都使用的人类偏好作为代理（proxy）来修正这个问题。把两个完成结果给标注员；标注员选出更好的一个；用标注结果训练一个奖励模型让它学习标注员的偏好。然后用强化学习循环把策略朝着奖励模型评分高的输出方向移动。用一句话概括就是完整的 InstructGPT 论点。论文的其余部分是工程实现。

## 概念

### Stage 1: supervised fine-tuning (SFT)

收集提示-响应对，其中响应是一个善意的人类会写出的答案。Ouyang 等人使用了 13k 个来自标注员和 OpenAI API 的提示。在这些数据上用标准的交叉熵损失对基础模型进行微调。

SFT 带来的效果：模型现在会“回答问题”而不是继续输出问题的后续文本。它没有带来的是：当存在多个合理答案时，标注员更偏好哪个答案的信息。

### Stage 2: reward model (RM)

对于每个提示，从 SFT 模型中采样 K 个完成。标注员对它们进行排序。训练一个奖励模型，使其为任意提示-响应对打分，以便对于被偏好（preferred）的 `y_w` 相对于 `y_l`：

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这是 Bradley-Terry 的成对偏好损失。RM 通常用 SFT 模型初始化，并把语言模型头替换为一个标量头。

奖励模型规模通常较小：对于 175B 的 InstructGPT，6B 就足够了。它们也很脆弱 —— 论文第 5 节主要讨论了在小规模上出现的奖励操控（reward-hacking）行为。

### Stage 3: PPO with a KL penalty

定义目标：

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

用 PPO 最大化该目标。KL 项阻止 `pi` 偏离 SFT 策略太远。没有它，优化器会找到对奖励模型具有高分的对抗性示例 —— 这些字符串之所以得高分是因为奖励模型没有见过它们，而不是因为人类真正更偏好它们。

KL 系数 `beta` 是单个最重要的 RLHF 超参数。太低：会发生奖励操控。太高：相对于 SFT 没有改进。

### 对齐税（the alignment tax）

经过 RLHF 后，模型在人工偏好上被人类更偏好，但在标准基准（SQuAD、HellaSwag、DROP）上退化。Ouyang 等人称之为对齐税，并用 PPO-ptx 修复：在 RL 目标中混入预训练梯度，以防模型忘记如何完成那些它从未因之得到奖励的下游任务。

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx 成为常规做法。Anthropic、DeepMind 和 Meta 都使用某种变体。

### 结果

一个 1.3B 的 InstructGPT（SFT + RM + PPO-ptx）在大约 70% 的时间里被标注员偏好于 175B 的基线 GPT-3。在来自生产流量的隐藏测试提示上，这一差距还会扩大。从这个数字可以读出两点：

1. 对齐是与能力不同的一个轴。175B 模型在能力上更强；1.3B 模型在对齐上更好；标注员更偏好被对齐的那个。
2. 能力的下限由基础模型决定。你无法通过 RLHF 把一个基础模型训练成知道它从未见过的事实。

### 为什么这是 Phase 18 的参考点

后续课堂的每一个批评 —— 奖励操控（Lesson 2）、DPO（Lesson 3）、谄媚（sycophancy）（Lesson 4）、CAI（Lesson 5）、潜伏代理（sleeper agents）（Lesson 7）、对齐伪装（alignment faking）（Lesson 9）—— 都是在反对该管线的某个部分。奖励操控攻击第 2 阶段。DPO 把第 2 阶段和第 3 阶段合并。CAI 替换了人类标注员。谄媚显示标注员本身就是有偏的信号。对齐伪装表明策略可以完全绕开第 3 阶段。若不先把这条管线放在脑中，你无法理解这些批评。

## 使用方法

`code/main.py` 在玩具偏好数据上模拟这三阶段流程。基础“策略”是在动作集合 {A, B, C} 上的一个有偏硬币。Stage 1 的 SFT 模拟标注员在 200 个提示上的动作。Stage 2 从 500 个成对排名中拟合一个 Bradley-Terry 奖励模型。Stage 3 运行一个简化的带 KL 惩罚的 PPO 更新到 SFT 策略。你可以观察奖励如何上升、KL 发散如何增长，以及策略如何漂移 —— 你也可以关闭 KL 项来观察在 50 步更新以内出现的奖励操控。

要观察的内容：

- `beta = 0.1` 与 `beta = 0.0` 下的奖励轨迹。
- 训练步骤中的 KL(pi || pi_SFT)。
- 最终动作分布与标注员偏好的比较。

## 发布产物

本课产出 `outputs/skill-instructgpt-explainer.md`。给定一个 RLHF 管线描述或论文摘要，它能识别被修改的是哪三个阶段中的哪一阶段、每个阶段使用的损失函数，以及是否存在 KL 惩罚或等价的正则项。

## 练习

1. 运行 `code/main.py`。将 `beta = 0.0`，报告经过 200 次 PPO 步骤后的动作分布。用一段话解释模式寻求（mode-seeking）行为。

2. 将奖励模型修改为对动作 B 有 +0.5 的偏置（模拟奖励缺陷）。在 `beta = 0.1` 下运行 PPO。KL 惩罚能否阻止策略剥削该偏置？在什么样的 `beta` 下剥削开始可见？

3. 阅读 Ouyang 等人（arXiv:2203.02155）图 1。通过运行 PPO 1、5、20、100 步并测量相对于 SFT 模型的偏好来重现标注员偏好曲线。

4. 论文第 4.3 节报告了 1.3B 的 InstructGPT 在大约 70% 的时间里胜过 175B 的 GPT-3。为什么在生产环境的隐藏提示上这一比例会比在标注员自己的提示上更高？

5. 在相同偏好数据上将 PPO 损失替换为 DPO（Phase 10 · 08）。比较最终策略漂移（相对于 SFT 的 KL）和最终奖励。哪种方法在匹配奖励的情况下漂移得更远？

## 术语表

| Term | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| SFT | “instruction tuning” | 阶段 1：在提示-响应对上用交叉熵进行微调 |
| Reward model | “the RM” | 对（提示, 响应）做标量回归的模型，使用 Bradley-Terry 在成对标签上训练 |
| Bradley-Terry | “pairwise preference loss” | -log sigmoid(r_w - r_l)；将成对排序问题转化为二分类 |
| KL penalty | “the regularizer” | `beta * KL(pi \|\| pi_SFT)` —— 将 RL 策略保持在 SFT 锚点附近 |
| PPO-ptx | “PPO with pretraining mix” | 在 PPO 目标中加入一部分预训练对数似然，以抵消对齐税 |
| Alignment tax | “the RLHF regression” | RLHF 在未被目标化的标准基准上的性能下降 |
| Labeler preference | “the ground truth” | 人类排序的样本；RM 是对其的统计代理，而不是“人类价值观”的代理 |

## 延伸阅读

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — InstructGPT 论文，是后来每个 RLHF 管线的基础  
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — 针对摘要任务的 RLHF 前驱工作  
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — 原始的基于偏好的强化学习表述  
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — Anthropic 对 InstructGPT 管线的 HH 扩展