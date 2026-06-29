# 动态规划 — 策略迭代 与 价值迭代

> 动态规划就是有作弊手段的强化学习。你已经知道转移和回报函数；只需反复迭代贝尔曼方程直到 `V` 或 `π` 不再变化。它是每个基于采样的方法都试图逼近的基准。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 分钟

## 问题描述

你有一个已知模型的 MDP：可以查询任意状态-动作对的 `P(s' | s, a)` 和 `R(s, a, s')`。库存经理知道需求分布；桌面游戏有确定性转移；网格世界四行 Python 就能写完。你有一个*模型*。

无模型强化学习（Q-learning、PPO、REINFORCE）是在没有模型、只能从环境采样的情形下发明的。但当你有模型时，有更快、更好的方法：动态规划。Bellman 在1957年设计了这些方法。它们至今仍定义了“正确性”：当人们说“这个 MDP 的最优策略”时，指的就是 DP 会返回的策略。

到 2026 年你需要它们有三个原因。第一，强化学习研究中的每个表格化环境（GridWorld、FrozenLake、CliffWalking）都会用 DP 求得金标准策略。第二，精确值可以用来*调试*基于采样的方法：如果 Q-learning 对 `V*(s_0)` 的估计与 DP 答案偏差 30%，那你的 Q-learning 就有问题。第三，现代离线 RL 与规划方法（MCTS、AlphaZero 的搜索、基于模型的 RL）都在对学习到的或给定的模型上迭代贝尔曼备份。

## 概念

![策略迭代与价值迭代，并排展示](../assets/dp.svg)

**两种算法，都是对贝尔曼方程的定点迭代。**

**策略迭代。** 在两个步骤之间交替进行，直到策略不再改变。

1. 评估（Evaluation）：给定策略 `π`，通过反复应用
   `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`
   来计算 `V^π`，直到收敛。
2. 改进（Improvement）：给定 `V^π`，对 `V^π` 做贪婪改进：  
   `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛有保证，因为 (a) 每次改进要么使 `π` 保持不变，要么在某个状态上严格提高 `V^π`，(b) 确定性策略的空间是有限的。通常即使在较大的状态空间下，也会在约 5–20 次外层迭代内收敛。

**价值迭代。** 将评估和改进合并为一次遍历。应用贝尔曼*最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后通过贪婪选取动作来提取策略。每次迭代严格更快——没有内部评估循环——但通常需要更多次迭代才能收敛。

**广义策略迭代（GPI）。** 统一的框架。价值函数和策略锁定在一个双向改进循环中；任何将两者驱动到相互一致的方法（异步值迭代、修改的策略迭代、Q-learning、actor-critic、PPO）都是 GPI 的实例。

**为什么 `γ < 1` 很重要。** 贝尔曼算子在上确界范数（sup-norm）下是 `γ`-收缩：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。收缩意味着唯一的定点和几何收敛。放弃 `γ < 1` 的条件就失去保证——此时需要有限时域或吸收终止状态。

```figure
value-iteration-gamma
```

## 实现

### 第 1 步：构建 GridWorld MDP 模型

使用第 01 课的相同 4×4 GridWorld。我们加入一个随机版：以概率 `0.1`，智能体会滑向一个随机的垂直/水平相对方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回一个 `(s', r, p)` 的列表。这就是完整的模型。

### 第 2 步：策略评估

给定策略 `π(s) = {action: prob}`，反复迭代贝尔曼方程直到 `V` 不再变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 第 3 步：策略改进

用相对于 `V` 的贪婪策略替换 `π`。如果 `π` 没有改变，则返回——我们已达到最优。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 第 4 步：将它们拼接在一起

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

在 4×4 上典型收敛需要 4–6 次外层迭代。输出 `V*(0,0) ≈ -6`，以及严格减少步数的策略。

### 第 5 步：价值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同的定点，代码更少。

## 陷阱

- **忘记处理终止状态。** 如果你对吸收状态应用贝尔曼备份，它仍会选择一个“最佳动作”，虽然没任何改变。用 `if s == terminal: V[s] = 0` 来防护。
- **上确界范数 vs L2 收敛。** 使用 `max |V_new - V|`（上确界范数），而不是平均值。理论保证是基于上确界范数的。
- **就地更新 vs 同步更新。** 就地更新 `V[s]`（Gauss-Seidel）比使用单独的 `V_new` 字典（Jacobi）收敛更快。生产代码通常使用就地更新。
- **策略并列（ties）。** 如果两个动作的 Q 值相等，`argmax` 可能在每次迭代中不同地打破平局，导致“策略稳定”检查出现振荡。使用稳定的平局规则（固定顺序的第一个动作）。
- **状态空间爆炸。** DP 每次遍历的复杂度为 `O(|S| · |A|)`。在 ~10⁷ 个状态内可行。超过这个规模需要函数逼近（见 Phase 9 · 05 之后的内容）。

## 使用场景

到 2026 年，DP 是正确性的基准和规划器的内循环：

| 使用场景 | 方法 |
|----------|--------|
| 精确求解小型表格化 MDP | Value iteration（更简单）或 policy iteration（外层迭代更少） |
| 验证 Q-learning / PPO 实现 | 与玩具环境上的 DP 最优 V* 做比较 |
| 基于模型的 RL (Phase 9 · 10) | 在学习到的转移模型上做贝尔曼备份 |
| AlphaZero / MuZero 中的规划 | 蒙特卡洛树搜索 = 异步贝尔曼备份 |
| 离线 RL (CQL, IQL) | Conservative Q-iteration — 在 OOD 动作上加罚的 DP |

每当有人说“最优价值函数（the optimal value function）”，他们指的是“DP 的定点”。当你在论文里看到 `V*` 或 `Q*`，就想象这个循环。

## Ship It

Save as `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 练习

1. 简单：在 4×4 GridWorld 上对 `γ ∈ {0.9, 0.99}` 运行价值迭代。到 `max |ΔV| < 1e-6` 需要多少次遍历？将 `V*` 以 4×4 网格形式打印出来。
2. 中等：在有滑动概率 `0.1` 的随机 GridWorld 上比较策略迭代与价值迭代。统计：遍历次数、墙钟时间、最终 `V*(0,0)`。哪一个在迭代次数上更快？在墙钟时间上更快？
3. 困难：构建修改版策略迭代：在评估步骤中只运行 `k` 次遍历而不是收敛。绘制 `V*(0,0)` 误差相对于 `k` 的曲线，`k ∈ {1, 2, 5, 10, 50}`。曲线告诉你评估/改进之间的权衡是什么？

## 关键词

| 术语 | 大家如何说 | 实际含义 |
|------|-----------|---------|
| Policy iteration | "DP algorithm" | 交替进行评估（`V^π`）和改进（相对于 `V^π` 的贪婪 `π`），直到策略不再改变。 |
| Value iteration | "Faster DP" | 在一次遍历中应用贝尔曼最优性备份；以几何收敛性收敛到 `V*`。 |
| Bellman operator | "The recursion" | `(T V)(s) = max_a Σ P (r + γ V(s'))`；在上确界范数下是 `γ`-收缩映射。 |
| Contraction | "Why DP converges" | 任何满足 `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` 的算子都有唯一定点。 |
| GPI | "Everything is DP" | 广义策略迭代（Generalized Policy Iteration）：任何将 `V` 与 `π` 驱动到相互一致的方法。 |
| Synchronous update | "Jacobi-style" | 在一轮遍历中始终使用旧的 `V`；理论上更容易分析但较慢。 |
| In-place update | "Gauss-Seidel-style" | 在遍历过程中即时使用更新后的 `V`；实践中收敛更快。 |

## 深入阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 策略迭代与价值迭代的权威讲解。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — 关于收缩映射论证的严格处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 修改策略迭代及其收敛性分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 原始的策略迭代论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 从 DP 到近似 DP / 深度 RL 的桥梁，后续课程均在此基础上展开。