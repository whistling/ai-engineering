# Reward Modeling & RLHF

> 人类无法为“好的助手回复”手写一个奖励函数，但他们可以比较两个回复并选择更好的那个。将这些比较拟合成奖励模型，然后用 RL 针对该模型对语言模型进行强化学习。Christiano 2017。InstructGPT 2022。这是把 GPT-3 变成 ChatGPT 的方法。在 2026 年它大多被 DPO 取代——但思维模型仍然存在。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 分钟

## 问题

你用下一个 token 预测目标训练了一个语言模型。它会写出语法正确的英语。但它也会撒谎、长篇大论、以及拒绝拒绝。仅靠更多的预训练无法修复——网页文本是问题所在，而非解药。

你想要一个标量奖励来表示“对于指令 X，回复 A 比回复 B 好”。手写该奖励函数不可能实现。“有帮助性”并不是一个基于 token 的封闭表达式。但人类可以比较两个输出并标出偏好。这种数据便宜且可扩展地收集。

RLHF（Christiano 等，2017；Ouyang 等，2022）把偏好数据转换为奖励模型，然后通过 PPO 针对该奖励优化语言模型。三步走：SFT → RM → PPO。这就是在 2023–2025 年交付 ChatGPT、Claude、Gemini 和其他所有对齐 LLM 的配方。

到 2026 年，PPO 步骤大多被 DPO（Phase 10 · 08）取代，因为它更便宜且在对齐微调上几乎同样有效。但*奖励模型*这块仍然支撑着每一个 Best-of-N 采样器、每一个基于可验证奖励的 RL 管道，以及每一个使用过程奖励模型的推理模型。理解 RLHF 就等于理解整个对齐栈。

## 概念

![三阶段 RLHF：SFT、基于成对偏好的 RM 训练、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**阶段 1：监督微调 (SFT)。** 从预训练的基础模型开始。在人工示例（指令跟随的回复、有帮助的回答等）上进行微调。结果：一个偏向良好行为的模型 `π_SFT`，但动作空间仍然是无界的。

**阶段 2：奖励模型训练。**

- 收集针对提示 `x` 的一对回复 `(y_+, y_-)`，由人类标注为“y_+ 优于 y_-”。
- 训练奖励模型 `R_φ(x, y)` 对 y_+ 给出更高分。
- 损失：**Bradley-Terry 成对逻辑回归**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 为 sigmoid。奖励差对应偏好的对数几率。BT 自 1952 年（Bradley-Terry）以来就是标准，并且是现代 RLHF 中的主流选择。

- `R_φ` 通常从 SFT 模型初始化，在顶部接一个标量头。保持相同的 transformer 骨干；用一层线性层输出奖励。

**阶段 3：用带 KL 惩罚的 PPO 针对 RM。**

- 用 `π_SFT` 初始化可训练策略 `π_θ`。保留冻结的*参考* `π_ref = π_SFT`。
- 对回复 `y` 的最终奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT`——它是一个*正则项*，而不是严格的信赖域。`β` 通常在 `0.01`-`0.05` 之间。
- 运行 PPO（Lesson 08）以该奖励进行训练。优势值在 token 级轨迹上计算，但 RM 只对完整回复打分。

为什么要加 KL？没有它，PPO 会很乐意找到奖励劫持的策略——RM 仅在分布内的完成上训练过。一个分布外的回复可能得分高于任何人写的回复。KL 将 `π_θ` 保持在 RM 训练的流形附近。这是 RLHF 中最重要的旋钮。

2026 年现状：

- **DPO**（Rafailov 2023）：封闭代数形式将阶段 2+3 合并为一个基于偏好数据的监督损失。无 RM，无 PPO。对齐基准上质量相同但计算成本更低。见 Phase 10 · 08。
- **GRPO**（DeepSeek 2024–2025）：用群体相对基线替代 critic 的 PPO，奖励由 *验证器*（代码运行 / 数学答案匹配）提供，而不是人类训练的 RM。推理模型领域的主流。见 Phase 9 · 12。
- **过程奖励模型（PRMs）：** 给部分解打分（每个推理步骤），在 RLHF 和 GRPO 的推理变体中使用。
- **Constitutional AI / RLAIF：** 使用已对齐的 LLM 生成偏好替代人类。扩展偏好预算。

## 实现

本课使用微小的合成“提示”和“回复”字符串表示。RM 是基于词袋表示的线性评分器。不是实际的 LLM——关注的是流程形状，而非规模。见 `code/main.py`。

### 步骤 1：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实的 RLHF 中，这由人工标注员替代。形状——`(prompt, preferred_response, rejected_response)`——完全相同。

### 步骤 2：Bradley-Terry 奖励模型

线性得分：`R(x, y) = w · bag(y)`。训练以最小化 BT 成对对数损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百次更新后，`w` 会对好词分配正权重，对坏词分配负权重。

### 步骤 3：在 RM 上做类 PPO 的策略训练

我们的玩具策略从词表中生成单个 token。我们用 RM 给该 token 打分，计算 `log π_θ(token | prompt)`，加上相对于参考的 KL 惩罚，然后应用裁剪的 PPO 代理目标。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # 对 theta 做 PPO 风格的更新，把 reward 当作回报
    ...
```

### 步骤 4：监控 KL

在每次更新时跟踪平均 `KL(π_θ || π_ref)`。如果它悄然超过 `~5-10`，说明策略已经远离 `π_SFT`——β 太小或奖励劫持开始发生。这是实际 RLHF 中的首要诊断指标。

### 步骤 5：使用 TRL 的生产级配方

一旦你理解了玩具流水线，下面是与真实库用户写的相同循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——Stage 2 使用 `RewardTrainer`，Stage 3 使用内置 KL 到参考的 `PPOTrainer`。

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

库为你做了三件事。`adap_kl_ctrl=True` 实现自适应 β 调度：如果观测到的 KL 超过 `target_kl`，β 加倍；若低于一半，β 减半。参考模型按惯例被冻结——你绝对不能不小心和 `policy` 共享参数。值头（value head）位于与策略相同的骨干上（`AutoModelForCausalLMWithValueHead` 附加一个标量 MLP 头），这就是为什么 TRL 会分别报告 `policy/kl` 与 `value/loss`。

## 陷阱

- **过度优化 / 奖励劫持。** RM 有缺陷；`π_θ` 会找到对抗性的完成，使得分很高但实际很糟。症状：奖励无限上升而人工评估分数停滞或下降。修复：提前停止、提高 `β`、扩展 RM 的训练数据。
- **长度劫持。** 在有用回复上训练的 RM 常隐式奖励长度。策略学会填充回复。补救：长度归一化的奖励，或使用对长度敏感的 RLAIF RM。
- **RM 太小。** RM 的规模需要至少和策略相当。微小的 RM 无法忠实地给策略输出评分。
- **KL 调参。** β 太低 → 漂移和奖励劫持。β 太高 → 策略几乎不变。常用技巧是目标为每步固定 KL 的*自适应* β。
- **偏好数据噪声。** 约 30% 的人工标注是有噪声或模糊的。通过在一致性过滤的数据上训练 RM 或在 BT 上使用温度来校准。
- **离策略问题。** 在第一个 epoch 后 PPO 的数据稍有离策略。像 Lesson 08 那样监控 clip fraction。

## 使用场景

到 2026 年 RLHF 是分层的：

| Layer | Target | Method |
|-------|--------|--------|
| Instruction following, helpfulness, harmlessness | Alignment | DPO (Phase 10 · 08) preferred over RLHF-PPO. |
| Reasoning correctness (math, code) | Capability | GRPO with verifier reward (Phase 9 · 12). |
| Long-horizon multi-step tasks | Agentic | PPO / GRPO with process reward models over steps. |
| Safety / refusal behavior | Safety | RLHF-PPO with separate safety RM, or Constitutional AI. |
| Best-of-N at inference | Fast alignment | Use RM at decode time; no policy training needed. |
| Reward distillation | Inference compute | Train a small "reward head" on top of a frozen LM. |

RLHF 曾是 2022–2024 年的*首选*方法。在 2026 年，生产级对齐流水线以 DPO 优先，PPO 仅用于 RM 密集或安全关键的步骤。

## 上线交付

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 练习

1. 简单：在 `code/main.py` 上用 500 个合成偏好对训练 Bradley-Terry 奖励模型。在保留的 100 个对上测量成对准确率。应超过 90%。
2. 中等：用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对于每个设置，绘制更新过程中的 RM 得分 vs 相对于参考的 KL。哪些运行出现奖励劫持？
3. 困难：在相同偏好数据上实现 DPO（闭式偏好似然损失），并与 RLHF-PPO 流水线比较所用算力与最终 RM 得分。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| RLHF | "Alignment RL" | Three-stage SFT + RM + PPO pipeline (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "The scoring net" | Learned scalar function fit to pairwise preferences via Bradley-Terry. |
| Bradley-Terry | "Pairwise logistic loss" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; the standard RM objective. |
| KL penalty | "Stay near the reference" | `β · KL(π_θ \|\| π_ref)` in the reward; the anti-reward-hacking regularizer. |
| Reward hacking | "Goodhart's law" | Policy exploits RM flaws; symptoms: reward up, human eval flat. |
| RLAIF | "AI-labeled preferences" | RLHF where labels come from another LM instead of humans. |
| PRM | "Process Reward Model" | Scores partial reasoning steps; used in reasoning pipelines. |
| Constitutional AI | "Anthropic's method" | AI-generated preferences guided by explicit rules. |

（注：表格中很多术语为专有或缩写，通常保留原文缩写并在上下文中解释。）

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 启动 RLHF 的论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — 支持 ChatGPT 的配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 较早期的摘要方向 RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年的后 RLHF 默认方法。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 与自我批判循环。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 生产级 `RewardTrainer` 与 `PPOTrainer` 文档。阅读 trainer 源码以了解自适应 KL 与值头的细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — 带图示的三阶段流水线权威演练。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 该库；`examples/` 有针对 Llama、Mistral、Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 奖励假设与设计；思考奖励劫持的基本先修读物。