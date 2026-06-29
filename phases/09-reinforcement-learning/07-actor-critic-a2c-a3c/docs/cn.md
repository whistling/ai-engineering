# 演员-评论家 — A2C 与 A3C

> REINFORCE 噪声很大。加入一个学习 `V̂(s)` 的评论家，从回报中减去它，你就得到与原期望相同但方差远小的优势。这就是演员-评论家。A2C 同步运行它；A3C 在多线程下异步运行。两者都是每个现代深度强化学习方法的思维模型。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75 分钟

## 问题

原始的 REINFORCE 可行，但方差非常糟糕。蒙特卡洛回报 `G_t` 在不同回合之间可能波动十倍以上。将这些噪声乘以 `∇ log π` 并求平均会产生一个需要成千上万回合才能像少量 DQN 更新那样移动策略的梯度估计器。

方差来自于使用原始回报。如果你减去一个基线 `b(s_t)` —— 任何状态的函数，包括一个学习到的价值 —— 期望不变而方差下降。最可处理的最佳基线是 `V̂(s_t)`。现在乘以 `∇ log π` 的量就是 *优势*：

`A(s, a) = G - V̂(s)`

如果一个动作产生了高于平均的回报，则该动作是好的；反之则差。带学习评论家的 REINFORCE 就是 *演员-评论家*。评论家为演员提供了低方差的教师信号。这也是 2015 年后所有深度策略方法的基础（A2C、A3C、PPO、SAC、IMPALA）。

## 概念

![演员-评论家：策略网络加价值网络，TD 残差作为优势](../assets/actor-critic.svg)

**两个网络，共享损失：**

- **Actor** `π_θ(a | s)`：策略。用于采样执行动作。用策略梯度训练。
- **Critic** `V_φ(s)`：估计从状态出发的期望回报。通过最小化 `(V_φ(s) - target)²` 来训练。

**优势。** 两种常见形式：

- *MC 优势：* `A_t = G_t - V_φ(s_t)`。无偏，但方差高。
- *TD 优势：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用了 `V_φ`），但方差远低。也称为 *TD 残差* `δ_t`。

**n 步优势。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现对 Atari 使用 `n = 5`，对 MuJoCo 上的 PPO 使用 `n = 2048`。

**广义优势估计（GAE）。** Schulman 等人（2016）提出对所有 n 步优势做指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年的默认值 —— 根据偏差/方差权衡进行调优。

**A2C：同步优势演员-评论家。** 在 `N` 个并行环境中收集 `T` 步。为每一步计算优势。在组合批次上更新演员和评论家。重复。是 A3C 的更简单、更可扩展的姊妹方法。

**A3C：异步优势演员-评论家。** Mnih 等人（2016）。启动 `N` 个工作线程，每个线程运行一个环境。每个 worker 在自己的 rollout 上局部计算梯度，然后异步地应用到共享参数服务器。无需重放缓冲区 —— 工作线程通过运行不同轨迹实现去相关化。A3C 证明可以在 CPU 上大规模训练。到 2026 年，基于 GPU 的 A2C（批量并行环境）占主导，因为 GPU 需要大批次。

**组合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失、价值回归、熵奖励。`c_v ~ 0.5`、`c_e ~ 0.01` 是典型的起点。

## 实现

### 步骤 1：评论家

线性评论家 `V_φ(s) = w · features(s)` 用 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格化环境上，评论家在几百回合内收敛。在 Atari 上，将线性评论家替换为共享的 CNN 主干 + 价值头。

### 步骤 2：n 步优势

给定长度为 `T` 的 rollout 和引导的末端 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是评论家的目标。`advantages` 是乘以 `∇ log π` 的量。

### 步骤 3：组合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # 评论家
    critic_update(w, x, target_v, lr_v)

    # 演员
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

在策略式（on-policy）设置中，每次更新使用一个 rollout，演员和评论家使用不同的学习率。

### 步骤 4：并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程。每个线程运行自己的环境和前向传播。定期将梯度更新推送到共享主节点。主节点上不加锁 —— 竞争是可以接受的，它们只是增加噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观测堆叠成 `[N, obs_dim]` 的批次，做批量前向与反向传播。GPU 利用率更高，确定性更强，更易于分析。到 2026 年这是默认选择。

我们的示例代码为单线程以便于理解；改写为批量化的 A2C 只需三行 numpy 代码。

## 陷阱

- **评论家在演员梯度之前的偏差。** 如果评论家是随机的，它的基线没有信息，你就会在纯噪声上训练。先让评论家热身几百步再启用策略梯度，或使用较小的演员学习率。
- **优势归一化。** 在每个批次上将优势归一化为零均值/单位方差。几乎零成本却能极大稳定训练。
- **共享主干。** 对图像输入使用演员和评论家共享的特征提取器，单独的头。共享特征可以让两个损失共同受益。
- **在策略合约（on-policy）限制。** A2C 每个数据只用于一次更新。更多次使用会导致梯度有偏（PPO 添加重要性采样裁剪以修正这个问题）。
- **熵坍缩。** 如果 `c_e = 0`，策略在几百次更新内会变得近似确定，从而停止探索。
- **奖励尺度。** 优势幅度取决于奖励尺度。对奖励做归一化（例如用运行时标准差除）可以在不同任务之间保持一致的梯度幅度。

## 使用场景

A2C/A3C 在 2026 年很少是最终选择，但它们是后续所有改进的基础架构：

| Method | Relation to A2C |
|--------|----------------|
| PPO | A2C + 截断重要性比（clipped importance ratio）以支持多轮次更新 |
| IMPALA | A3C + V-trace 离策略修正 |
| SAC (Phase 9 · 07) | 带软值评论家的离策略 A2C（下一课） |
| GRPO (Phase 9 · 12) | 无评论家的 A2C — 群体相对优势 |
| DPO | 将 A2C 折叠为偏好排序损失，无需采样 |
| AlphaStar / OpenAI Five | 在联赛训练 + 模仿预训练下的 A2C |

如果你在 2026 年的论文里看到 “advantage”，就想到演员-评论家。

## 上线交付（Ship It）

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: 为给定环境生成 A2C / A3C / GAE 的配置，包含优势估计和损失权重的指定。
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

给定环境和计算预算，输出：

1. 并行度。A2C（GPU 批量）或 A3C（CPU 异步）以及工作线程数量。
2. Rollout 长度 T。每个环境每次更新的步数。
3. 优势估计器。n-step 或 GAE(λ)；注明 λ 值。
4. 损失权重。`c_v`（价值）、`c_e`（熵）、梯度裁剪。
5. 学习率。演员和评论家（若使用则分别指定）。

拒绝在 horizon > 1000 的环境上使用单 worker 的 A2C（过于在策略、过慢）。拒绝在未做优势归一化的情况下交付。若 `c_e = 0` 且观测到的熵 < 0.1，则标记为熵坍缩。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上训练带 MC 优势（`G_t - V(s_t)`）的演员-评论家。将样本效率与课题 06 中带运行均值基线的 REINFORCE 做比较。
2. **中等。** 切换到 TD 残差优势（`r + γ V(s') - V(s)`）。测量优势批次的方差。方差下降了多少？
3. **困难。** 实现 GAE(λ)。搜索 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终回报对样本效率的曲线。对于这个任务，偏差/方差的最优点在哪里？

## 关键术语

| Term | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Actor | "The policy net" | `π_θ(a\|s)`，通过策略梯度更新。 |
| Critic | "The value net" | `V_φ(s)`，通过对回报 / TD 目标做 MSE 回归来更新。 |
| Advantage | "How much better than average" | `A(s, a) = Q(s, a) - V(s)` 或其估计。乘以 `∇ log π` 的量。 |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`；一步优势估计。 |
| GAE | "The interpolation knob" | 参数为 `λ` 的 n 步优势的指数加权和。 |
| A2C | "Synchronous actor-critic" | 在多个环境上做批处理；每个 rollout 做一步梯度更新。 |
| A3C | "Async actor-critic" | 工作线程将梯度推送到共享参数服务器。原始论文方法；到 2026 年不常见。 |
| Bootstrap | "Use V at the horizon" | 截断 rollout，在末端加上 `γ^n V(s_{t+n})` 来闭合和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始的异步演员-评论家论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础理论；与第 9 章（函数逼近）配合阅读，当评论家为神经网络时尤其重要。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 可扩展的分布式演员-评论家，带 V-trace 离策略修正。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的生产级 A2C/PPO 实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 两时域演员-评论家分解的基础收敛结果。