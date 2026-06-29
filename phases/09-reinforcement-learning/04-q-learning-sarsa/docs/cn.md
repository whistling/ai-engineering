# 时序差分 — Q-Learning 与 SARSA

> 蒙特卡洛要等到一集结束才更新。时序差分（TD）通过对下一个值估计自举，从每一步后就进行更新。Q-learning 是离策略且乐观的；SARSA 是在策略且谨慎的。两者只需一行代码。它们构成了本阶段所有深度强化学习方法的基础。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 9 · 01 (MDPs -> 马尔可夫决策过程), Phase 9 · 02 (Dynamic Programming -> 动态规划), Phase 9 · 03 (Monte Carlo -> 蒙特卡洛)  
**Time:** ~75 分钟

## 问题

蒙特卡洛方法可行，但有两个代价高昂的要求。它需要可终止的回合，而且只有在最终回报到达后才更新。如果回合是 1,000 步，蒙特卡洛要等 1,000 步才更新任何东西。它在实践中表现为高方差、低偏差且收敛慢。

动态规划则相反 —— 零方差的自举备份，但需要已知模型。

时序差分（TD）学习折中这两者。从单个转移 `(s, a, r, s')` 中构建一步目标 `r + γ V(s')`，并把 `V(s)` 向它微调。无需模型。无需完整回合。在右侧使用近似 `V` 会引入偏差，但相比蒙特卡洛方差明显更低，并且从第一步起就能在线更新。

这是现代强化学习（DQN、A2C、PPO、SAC）赖以运转的枢纽。本章其余内容是建立在你将在本课编写的一步 TD 更新之上的函数近似与各类技巧。

## 概念

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号内的量就是 TD 误差 `δ = r + γ V(s') - V(s)`。它是蒙特卡洛中 `G_t - V(s_t)` 的在线对应。收敛需要 `α` 满足 Robbins–Monro 条件（`Σ α = ∞`, `Σ α² < ∞`）并且所有状态被无限次访问。

**Q-learning。** 一种用于控制的离策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始将采用贪婪策略，而不管智能体实际采取了什么动作。该解耦使得 Q-learning 在智能体通过 ε-贪婪探索时仍能学习到最优 `Q*`。Mnih 等人（2015）将其转换为在 Atari 上的深度 Q 学习（见 Lesson 05）。

**SARSA。** 一种在策略的 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名字来源于元组 `(s, a, r, s', a')`。SARSA 使用智能体实际采取的下一个动作 `a'`，而不是贪婪的 `argmax`。它收敛到当前 ε-贪婪策略 `π` 的 `Q^π`，当 `ε → 0` 时变为 `Q*`。

**悬崖漫步的差异。** 在经典的悬崖漫步任务（掉下悬崖 = 奖励 -100）中，Q-learning 学到沿着悬崖边缘的最优路径，但在探索时偶尔会遭遇惩罚。SARSA 学到更安全的一条路径，离悬崖远一步，因为它在 Q 值中考虑了探索噪声。随着训练，两者在 `ε → 0` 时都能达到最优。在实践中这很重要：如果在部署时仍在探索，SARSA 的行为更保守。

**Expected SARSA。** 将 `Q(s', a')` 用在策略 `π` 下的期望替换：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比 SARSA 方差更低（不对 `a'` 采样），但仍是同样的在策略目标。通常是现代教材中的默认选择。

**n 步 TD 与 TD(λ)。** 通过等待 `n` 步再自举，在 TD(0) 与蒙特卡洛之间插值。`n=1` 为 TD，`n=∞` 为蒙特卡洛。TD(λ) 对所有 `n` 以几何权 `(1-λ)λ^{n-1}` 进行加权平均。大多数深度 RL 在实务中使用 `n` 在 3 到 20 之间。

```figure
qlearning-gridworld
```

## 实现

### 步骤 1：在 ε-贪婪策略上实现 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行代码。与 Q-learning 唯一的区别就是目标行。

### 步骤 2：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。这个符号就是在策略内/外差异的关键。

### 步骤 3：学习曲线

跟踪每 100 集的平均回报。在简单确定性的 GridWorld 上 Q-learning 收敛更快；在悬崖漫步上 SARSA 更保守。在 `code/main.py` 中的 4×4 GridWorld 上，使用 `α=0.1, ε=0.1` 时，两者在 ~2,000 集后都接近最优。

### 步骤 4：与 DP 真值比较

运行值迭代（Lesson 02）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。在 4×4 GridWorld 上，健康的表格 TD 智能体在 10,000 集后通常会落在大约 `~0.5` 的误差范围内。

## 陷阱

- **初始 Q 值很重要。** 乐观初始化（在负奖励任务中将 `Q` 设为 0）会鼓励探索。悲观初始化可能使贪婪策略永远陷入局部最优。
- **α 的调度。** 对非平稳问题常数 `α` 就足够。衰减的 `α_n = 1/n` 从理论上保证收敛，但实践上太慢 —— 把 `α` 锁在 `[0.05, 0.3]` 并监控学习曲线。
- **ε 的调度。** 从高开始（`ε=1.0`），衰减到 `ε=0.05`。GLIE（Greedy in the Limit with Infinite Exploration）是收敛所需的条件。
- **Q-learning 的最大化偏差。** 当 `Q` 有噪声时，`max` 算子会向上偏置，导致过估计 —— Hasselt 的 Double Q-learning（在 Lesson 05 的 DDQN 中使用）用两个 Q 表修正此问题。
- **非终止回合。** TD 可以在无终止的情况下学习，但你需要对步数上限进行处理，或者在上限处正确自举。常见做法：将上限视为非终止，并继续自举。
- **状态哈希。** 如果状态是元组/张量，使用可哈希键（元组而不是列表；浮点数请先四舍五入，而不是直接使用原始浮点数组）。

## 使用场景

| 任务 | 方法 | 原因 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 在策略的安全关键任务 | SARSA / Expected SARSA | 在探索期间更保守。 |
| 高维状态 | DQN (Phase 9 · 05) | 使用带回放和目标网络的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3 (Phase 9 · 07) | 在 Q 网络上进行 TD 更新；策略网络输出动作。 |
| 基于奖励模型的 LLM 强化学习 | PPO / GRPO (Phase 9 · 08, 12) | 使用 GAE 的 TD 样式优势估计的演员-评论家方法。 |
| 离线 RL | CQL / IQL (Phase 9 · 08) | 对 Q-learning 加入保守正则以适应离线数据。 |

到 2026 年，你读到的大约 90% 的“强化学习”论文都是对 Q-learning 或 SARSA 的某种扩展。在深入阅读之前，先把表格更新写熟。

## 部署输出示例

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。绘制 2,000 集的学习曲线（每 100 集的平均回报）。谁收敛更快？
2. **中等。** 构建一个悬崖漫步环境（4×12，最后一行是悬崖，掉落奖励为 -100 并重置到起点）。比较 Q-learning 与 SARSA 的最终策略。截图各自的路径。哪一个更靠近悬崖？
3. **困难。** 实现 Double Q-learning。在有噪声奖励的 GridWorld（每步奖励加高斯噪声 σ=5）上展示 Q-learning 对 `V*(0,0)` 有显著过高估计，而 Double Q-learning 则不会。

## 术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| TD error | "The update signal" | `δ = r + γ V(s') - V(s)`，自举残差。 |
| TD(0) | "One-step TD" | 每个转移后使用下一状态估计进行更新。 |
| Q-learning | "Off-policy RL 101" | 对下一个状态动作取 `max` 的 TD 更新；无论行为策略如何，都学习 `Q*`。 |
| SARSA | "On-policy Q-learning" | 使用实际采样到的下一个动作的 TD 更新；对当前 ε-贪婪 π 学习 `Q^π`。 |
| Expected SARSA | "The low-variance SARSA" | 用策略下的期望替换采样到的 `a'`。 |
| GLIE | "Correct exploration schedule" | Greedy in the Limit with Infinite Exploration；Q-learning 收敛所需的探索调度。 |
| Bootstrapping | "Using current estimate in the target" | 区别 TD 与蒙特卡洛的关键。带来偏差但大幅减小方差。 |
| Maximization bias | "Q-learning overestimates" | 对有噪声的估计取 `max` 会造成向上偏置；Double Q-learning 可修正。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文与收敛性证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 解决最大化偏差的方法。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — Expected SARSA 的动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 首次提出 SARSA（当时称为“modified connectionist Q-learning”）的论文。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 推广到 TD(n)，从 Q-learning 到资格迹（eligibility traces）的路径，以及后来的 PPO 中的 GAE。