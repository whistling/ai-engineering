# Policy Gradient — REINFORCE from Scratch

> 不再估计价值。直接对策略参数化，计算期望回报的梯度，向上步进。Williams (1992) 用一个定理写明了这一点。这就是为什么 PPO、GRPO 以及所有的 LLM RL 循环存在的原因。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 3 · 03 (反向传播), Phase 9 · 03 (蒙特卡洛), Phase 9 · 04 (时序差分学习)  
**Time:** ~75 分钟

## 问题

Q-learning 和 DQN 对*价值*函数进行参数化。你通过 `argmax Q` 选择动作。这在离散动作与离散状态下没问题。但在动作是连续的情况下就会崩溃（对 10 维力矩做 `argmax`？）或者当你想要一个随机策略时（`argmax` 从构造上是确定性的）。

策略梯度对*策略*本身进行参数化。`π_θ(a | s)` 是一个输出动作分布的神经网。通过从中采样来执行动作。计算期望回报关于 `θ` 的梯度。向上踏步。没有 `argmax`。没有贝尔曼递归。只是对 `J(θ) = E_{π_θ}[G]` 做梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。跑一条回合。计算回报。在每一步乘以 `∇ log π_θ(a | s)`。求平均。梯度上升。完成。

到 2026 年的每一个 LLM-RL 算法——PPO、DPO、GRPO——都是 REINFORCE 的改进。亲手掌握它是本阶段其余内容以及 Phase 10 · 07（RLHF 实现）和 Phase 10 · 08（DPO）的前提。

## 概念

![策略梯度：softmax 策略，log-π 梯度，回报加权更新](../assets/policy-gradient.svg)

**策略梯度定理。** 对任意由 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从时刻 `t` 开始的折扣回报。期望是对从 `π_θ` 采样得到的完整轨迹 `τ` 求的。

**证明很短。** 对期望下的 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数导数技巧）。把 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + 不依赖 θ 的环境项` 分解。环境项消掉。两行代数就得到了定理。

**方差降低技巧。** 朴素 REINFORCE 有致命的方差——回报有噪声，`∇ log π` 有噪声，它们的乘积非常嘈杂。两个常见修正：

1. **基线减法。** 将 `G_t` 替换为 `G_t - b(s_t)`，其中 `b(s_t)` 不依赖于 `a_t`。这是无偏的，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：令 `b(s_t) = V̂(s_t)`，由一个 critic 学习 → actor-critic（Lesson 07）。
2. **只计未来回报（reward-to-go）。** 用 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)` 替代 `Σ_t G_t · ∇ log π_θ(a_t | s_t)`。对于某个动作只关心未来的回报——过去的奖励对该动作贡献的是零均值噪声。

结合起来，你得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这是带基线的 REINFORCE —— A2C（Lesson 07）和 PPO（Lesson 08）的直接祖先。

**Softmax 策略参数化。** 对于离散动作，标准选择：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任意对每个动作输出得分的神经网。梯度有一个干净的形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即：所采取动作的得分减去该策略下的期望值。

**连续动作下的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式形式。这正是 Phase 9 · 07 的 SAC 所需的一切。

```figure
policy-gradient-landscape
```

## 实现

### 步骤 1：softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对于表格环境使用线性策略（每个动作一个权重向量）。对于 Atari，替换为 CNN 并保留 softmax 头。

### 步骤 2：采样与对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 步骤 3：在回合中记录对数概率

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 步骤 4：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（动作的 onehot 减去概率向量）是 softmax 策略梯度的核心。把它记到肌肉记忆里。

### 步骤 5：基线

用最近若干回合的 `G` 的运行均值作为基线就足以使 4×4 GridWorld 运行；收敛大约需要 ~500 个回合。将基线升级为学习到的 `V̂(s)`，即可得到 actor-critic。

## 陷阱

- **梯度爆炸。** 回报可能很大。在乘以 `∇ log π` 之前，总是在批次内对 `G` 做标准化到约 `~N(0, 1)`。
- **熵坍缩（Entropy collapse）。** 策略过早收敛为近似确定性，停止探索，陷入局部最优。修复方法：在目标中加入熵项 `β · H(π(·|s))`。
- **高方差。** 朴素 REINFORCE 需要成千上万集回合。一个 critic 基线（Lesson 07）或 TRPO/PPO 的信赖域（Lesson 08）是常用修复。
- **样本效率低。** On-policy 意味着每个转移在一次更新后就被丢弃。通过重要性采样进行离策略修正可以重用数据，但会带来方差（PPO 的比率就是被裁剪的 IS 权重）。
- **非平稳梯度。** 100 个回合之前得到的同样梯度对应的是旧的 `π`。所以 on-policy 方法通常每几次 rollout 更新一次以保证数据新鲜。
- **归因分配（Credit assignment）。** 如果不使用 reward-to-go，过去的奖励会成为噪声。始终使用 reward-to-go。

## 使用场景

到 2026 年，REINFORCE 很少直接单独运行，但它的梯度公式无处不在：

| Use case | Derived method |
|----------|---------------|
| Continuous control | PPO / SAC with Gaussian policy |
| LLM RLHF | PPO with KL penalty, running on token-level policy |
| LLM reasoning (DeepSeek) | GRPO — REINFORCE with group-relative baseline, no critic |
| Multi-agent | Centralized-critic REINFORCE (MADDPG, COMA) |
| Discrete action robotics | A2C, A3C, PPO |
| Preference-only settings | DPO — REINFORCE rewritten as a preference-likelihood loss, no sampling |

当你在 2026 年的训练脚本中看到 `loss = -advantage * log_prob`，那就是带基线的 REINFORCE。整篇论文（DPO、GRPO、RLOO）其实都在这一行上做方差降低的变体。

## 打包保存

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: 生成给定任务的 REINFORCE / actor-critic / PPO 训练配置，并诊断方差问题。
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 REINFORCE，使用线性 softmax 策略。训练 1,000 个回合，不使用基线。绘制学习曲线；测量回报的方差（回报的标准差）。
2. **中等。** 添加运行均值基线。重新训练。将样本效率和方差与朴素运行进行比较。基线将收敛所需步骤减少了多少？
3. **困难。** 添加熵奖励 `β · H(π)`。对 `β ∈ {0, 0.01, 0.1, 1.0}` 做参数搜索。绘制最终回报和策略熵。这个任务的最佳 β 值在哪儿？

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy gradient | "Train the policy directly" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; derived from the log-derivative trick. |
| REINFORCE | "The original PG algorithm" | Williams (1992); Monte Carlo returns multiplied by log-policy gradient. |
| Log-derivative trick | "Score function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; makes gradients of expectations tractable. |
| Baseline | "Variance reduction" | Any `b(s)` subtracted from `G`; unbiased because `E[b · ∇ log π] = 0`. |
| Reward-to-go | "Only future returns count" | `G_t^{from t}` instead of the full `G_0`; correct and lower-variance. |
| Entropy bonus | "Encourage exploration" | `+β · H(π(·\|s))` term keeps the policy from collapsing. |
| On-policy | "Train on what you just saw" | Gradient expectation is w.r.t. the current policy — cannot reuse old data directly. |
| Advantage | "How much better than average" | `A(s, a) = G(s, a) - V(s)`; the signed quantity REINFORCE-with-baseline multiplies. |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — 原始的 REINFORCE 论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 包含函数逼近的现代策略梯度定理。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书式讲解。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — 清晰的教学性阐述并附有 PyTorch 代码。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 关于方差降低和自然梯度视角，连接 REINFORCE 与信赖域家族（TRPO、PPO）。