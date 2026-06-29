# Proximal Policy Optimization (PPO)

> A2C 在一次更新后就丢弃每个 rollout。PPO 将策略梯度包裹在一个裁剪的重要性比率中，这样你就可以在同一数据上做 10+ 个 epoch 而不会导致策略爆炸。Schulman 等（2017）。截至 2026 年仍然是默认的策略梯度算法。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)  
**Time:** ~75 分钟

## 问题

A2C（第 07 课）是 on-policy 的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 需要从*当前* `π_θ` 采样的数据。做一次更新后，`π_θ` 发生变化；你用过的数据现在是 off-policy 的。重用它会使你的梯度有偏。

Rollouts 很昂贵。在 Atari 上，8 个环境 × 128 步的一个 rollout = 1024 个 transition，且需要十几秒的环境时间。一次梯度步就丢弃这些数据非常浪费。

Trust Region Policy Optimization（TRPO，Schulman 2015）是第一个修正：通过约束每次更新，使旧策略与新策略之间的 KL 散度低于 `δ`。理论上干净，但每次更新都需要做共轭梯度求解。到 2026 年几乎没人再用 TRPO。

PPO（Schulman 等，2017）用一个简单的裁剪目标替代了硬性的信赖域约束。只需一行代码。对每次 rollout 做十个 epoch。无共轭梯度。理论保证够用了。九年过去了，它仍然是从 MuJoCo 到 RLHF 的默认策略梯度算法。

## 概念

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略相对于收集数据时策略的似然比。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略比旧策略采样到 `a_t` 的概率高两倍。

**裁剪替代目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项含义：

- 如果优势 `A_t > 0` 且比率试图增长超过 `1 + ε`，裁剪会平坦化梯度 —— 不要把一个好的动作推得比旧概率高出超过 `+ε`。
- 如果优势 `A_t < 0` 且比率试图下降到 `1 - ε` 以下（意味着我们会相对旧策略使一个糟糕动作更可能），裁剪会限制梯度 —— 不要把一个坏的动作推得比旧策略低于 `-ε`。

`min` 处理另一方向的情况：如果比率已经向*有利*方向移动，你仍然可以得到梯度（不会在会伤害你的那一侧裁剪）。

典型 `ε = 0.2`。把目标作为 `r_t` 的函数画出来：分段线性函数，在“有利”一侧有平顶，在“不利”一侧有平底。

**完整的 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 actor-critic 结构。三项系数，通常 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 在 N 个并行环境中每个运行 T 步，收集 `N × T` transitions。
2. 计算优势（GAE），并将其冻为常数。
3. 将 `π_{θ_old}` 冻结为当前 `π_θ` 的快照。
4. 对于 K 个 epoch，对于每个 minibatch 的 `(s, a, A, V_target, log π_old(a|s))`：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵项。
   - 梯度下降一步。
5. 丢弃该 rollout。返回步骤 1。

`K = 10` 和 minibatch 大小 64 是标准超参数组合。PPO 很健壮：在 ±50% 范围内具体数字通常无关紧要。

**KL 惩罚变体。** 原始论文提出了一个替代方法，使用自适应 KL 惩罚：`L = L^{PG} - β · KL(π_θ || π_old)`，并根据观测到的 KL 调整 `β`。裁剪版本占据主导地位；KL 变体在 RLHF 中仍然保留（因为对参考策略的 KL 本身通常是你始终想要的一个单独约束）。

## 实现

### 步骤 1：在 rollout 时保存 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在 rollout 时一次性取得。在更新 epoch 期间它不会改变。

### 步骤 2：计算 GAE 优势（第 07 课）

与 A2C 相同。对整个 batch 做归一化。

### 步骤 3：裁剪替代目标更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # 反向传播 - 对 surrogate 取负，加入价值损失，减去熵项
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # 被裁剪（不更新）
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

“裁剪 → 零梯度”的模式是 PPO 的核心。如果新策略在有利方向上已经漂移得太远，更新就会停止。

### 步骤 4：价值与熵项

像 A2C 一样，为 critic 目标添加标准的 MSE，并在 actor 上加入熵奖励。

### 步骤 5：诊断指标

每次更新应关注三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]`。如果超过 `0.1`，降低 `K_EPOCHS` 或 `LR`。
- **裁剪比例（Clip fraction）** — 比率落在 `[1-ε, 1+ε]` 之外的样本比例。应为 `~0.1-0.3`。若接近 `~0`，裁剪从未触发 → 提高 `LR` 或 `K_EPOCHS`。若 `~0.5+`，说明你在过拟合该 rollout → 降低它们。
- **解释方差（Explained variance）** `1 - Var(V_target - V_pred) / Var(V_target)`。评价 critic 质量的指标。随着 critic 学习，应朝 1 上升。

## 陷阱

- **裁剪系数调参不当。** `ε = 0.2` 是事实标准。降到 `0.1` 会让更新过于谨慎；`0.3+` 会带来不稳定。
- **Epoch 太多。** `K > 20` 常常不稳定，因为策略会从 `π_old` 漂得太远。对于大网络尤其要限制 epoch 数。
- **没有奖励归一化。** 大尺度的 reward 会吃掉裁剪范围。在计算优势之前对奖励做归一化（running std）。
- **忘记优势归一化。** 按 batch 做均值为 0、方差为 1 的归一化是标准做法。跳过会让 PPO 在大多数基准上崩溃。
- **学习率不衰减。** PPO 从线性 LR 衰减到 0 中受益。恒定 LR 往往更差。
- **重要性比率计算错误。** 始终用 `exp(log_new - log_old)` 以保证数值稳定，而不是直接 `new / old`。
- **梯度符号错误。** 最大化替代目标 = *最小化* `-L^{CLIP}`。符号反了是最常见的 PPO 错误。

## 使用场景

PPO 是 2026 年在很多领域的默认 RL 算法：

| 用例 | PPO 变体 |
|------|----------|
| MuJoCo / 机器人控制 | 带高斯策略的 PPO，GAE(0.95) |
| Atari / 离散游戏 | 带类别策略的 PPO，滚动 128 步的 rollouts |
| LLM 的 RLHF | 带对参考模型 KL 惩罚的 PPO，奖励来自响应末端的 RM |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理类 LLM | GRPO（第 12 课）——无 critic 的 PPO 变体 |
| 仅偏好数据 | DPO —— PPO+KL 的闭式塌缩，无需在线采样 |

PPO 的*损失形状*——裁剪替代 + 价值 + 熵——构成了 DPO、GRPO 以及几乎所有 RLHF 流水线的框架。

## 发布（Ship It）

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: 生成给定环境的 PPO 训练配置和诊断计划。
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

给定一个环境和训练预算，输出：

1. Rollout 大小。`N` 个环境 × `T` 步。
2. 更新计划。`K` 个 epoch，minibatch 大小，学习率计划。
3. 替代参数。`ε`（裁剪），`c_v`，`c_e`，是否开启优势归一化。
4. 优势计算。GAE(`λ`)，并明确 `γ` 和 `λ`。
5. 诊断计划。KL、裁剪比例、解释方差的阈值和告警。

拒绝 `K > 30` 或 `ε > 0.3`（不安全的信赖域）。拒绝任何没有优势归一化或缺乏 KL/裁剪监控的 PPO 运行。若裁剪比例持续高于 0.4，则标记为策略漂移。
```

## 练习

1. 简单：在 4×4 GridWorld 上运行 PPO，`ε=0.2, K=4`。在相同环境步数下，与 A2C（每个 rollout 只做一个 epoch）的样本效率比较。
2. 中等：在 `K ∈ {1, 4, 10, 30}` 上做扫参。绘制回报随环境步数的曲线，并跟踪每次更新的平均 KL。在哪个 `K` 值上 KL 在该任务中爆炸？
3. 困难：用自适应 KL 惩罚替换裁剪替代（若 `KL > 2·target` 则 β 加倍，若 `KL < target/2` 则 β 减半）。比较最终回报、稳定性和是否免疫裁剪。

## 关键词

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Importance ratio | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`；相对于收集数据时的策略的偏离。 |
| Clipped surrogate | "PPO 的主要技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在裁剪之外的有利一侧梯度为平坦。 |
| Trust region | "TRPO / PPO 的意图" | 限制每次更新的 KL 以保证单调改进。 |
| KL penalty | "软信赖域" | PPO 的替代：`L - β · KL(π_θ \|\| π_old)`。自适应 β。 |
| Clip fraction | "裁剪触发频率" | 诊断项 —— 应为 0.1-0.3；超出表示调参不当。 |
| Multi-epoch training | "数据重用" | 对每个 rollout 做 K 个 epoch；用方差成本换取样本效率。 |
| On-policy-ish | "大体上是 on-policy" | PPO 名义上是 on-policy，但 K>1 时在稍微 off-policy 的数据上安全重用。 |
| PPO-KL | "另一个 PPO" | KL 惩罚变体；在 RLHF 中常用，因为对参考策略的 KL 已是一个存在的约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 原论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 大规模消融研究，几乎对每个 PPO 超参数都做了分析。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；RLHF 中的 PPO 配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 干净的现代阐述，带 PyTorch 示例。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 许多论文使用的参考单文件 PPO 实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 在语言模型上生产化使用 PPO 的说明；与第 09 课（RLHF）配合阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — “37 个代码层面的优化”论文；哪些 PPO 技巧是关键，哪些只是民间传说。