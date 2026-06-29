# MDPs, States, Actions & Rewards

> 马尔可夫决策过程由五部分构成：状态、动作、转移、奖励、折扣。强化学习中的一切 —— Q-learning、PPO、DPO、GRPO —— 都是在这个结构上进行优化。学会它一次，剩下的强化学习知识都会迎刃而解。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 1 · 06（概率与分布），Phase 2 · 01（机器学习分类）  
**Time:** ~45 分钟

## 问题

你正在编写一个棋类机器人。或者一个库存规划器。或者一个交易代理。或者训练推理模型的 PPO 循环。四个不同的领域，一个令人惊讶的事实：这四者都可以归结为同一个数学对象。

监督学习给你 `(x, y)` 对，并让你拟合一个函数。强化学习不给你标签 —— 只有一连串的状态、你采取的动作、以及一个标量奖励。这个走子赢了比赛吗？补货决策节省了成本吗？这笔交易赚了钱吗？LLM 刚输出的这个 token 是否让评判器的回报更高？

在你把它形式化之前，你无法从这一串数据中学习。“我看到了什么”、“我做了什么”、“接下来发生了什么”、“那有多好”——每一项都必须成为一个你可以推理的对象。这个形式化就是马尔可夫决策过程（MDP）。本阶段的每一个 RL 算法，包括最后的 RLHF 和 GRPO 循环，都是在这个结构上进行优化的。

## 概念

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**五个要素。**

- **States** `S`。代理决策所需的一切。在网格世界中，是格子；在国际象棋中，是棋盘；在 LLM 中，是上下文窗口以及任何记忆。
- **Actions** `A`。可选项。上/下/左/右 移动。落子。输出一个 token。
- **Transitions** `P(s' | s, a)`。给定状态 `s` 和动作 `a` 后，下一状态的分布。在国际象棋中是确定性的，在库存问题中是随机的，在 LLM 解码中几乎是确定性的但带一点随机性。
- **Rewards** `R(s, a, s')`。标量信号。赢 = +1，输 = -1。收入减去成本。GRPO 中的对数似然比项。
- **Discount** `γ ∈ [0, 1)`。未来奖励相对于当前奖励的重要程度。`γ = 0.99` 对应约 100 步的视野；`γ = 0.9` 对应约 10 步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只依赖于当前状态。如果不是这样，说明状态表示不完整 —— 不是方法失败，而是状态设计失败。

**策略与回报。** 策略 `π(a | s)` 将状态映射为动作分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣和。值函数 `V^π(s) = E[G_t | s_t = s]` 是在策略 `π` 下从状态 `s` 开始的期望回报。Q 值 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是以特定动作开始时的期望回报。每个 RL 算法要么估计这两者之一，然后相应地改进 `π`。

**贝尔曼方程。** 本阶段所有方法都使用的不动点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`  
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些方程将期望回报拆成“本步奖励”加上“到达下一状态的折扣值”。递归形式。本阶段的每个算法要么迭代此方程直到收敛（动态规划），要么从中采样（蒙特卡洛），要么进行一步自举（时序差分）。

```figure
discount-horizon
```

## 构建它

### 步骤 1：一个微小的确定性 MDP

一个 4×4 的网格世界。代理从左上角开始，终点在右下角，每步奖励 -1，动作集合 `{up, down, left, right}`。见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

仅五行。这就是整个环境。确定性转移、恒定步罚、吸收性终止状态。

### 步骤 2：执行策略并回放

策略是从状态到动作分布的函数。最简单的：均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略 1000 次。对于这个 4×4 棋盘，平均回报约为 -60 到 -80。最优回报是 -6（沿直线向右下走）。缩小这个差距就是 Phase 9 的全部工作。

### 步骤 3：通过贝尔曼方程精确计算 `V^π`

对于小型 MDP，贝尔曼方程是一个线性系统。枚举状态，应用期望，迭代直到值不再变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是迭代策略评估。它是 Sutton & Barto 中的第一个算法，也是后续所有 RL 方法的理论基础。

### 步骤 4：`γ` 是具有物理含义的超参数

有效视野大致为 `1 / (1 - γ)`。`γ = 0.9` → 10 步。`γ = 0.99` → 100 步。`γ = 0.999` → 1000 步。

γ 太小，代理行为会短视。γ 太大，归因（credit assignment）会变得非常噪声化，因为很多早期步骤都会对远期奖励共同负责。LLM 的 RLHF 通常使用 `γ = 1`，因为回合较短且有界。控制任务使用 `0.95–0.99`。长时域的策略游戏使用 `0.999`。

## 陷阱

- **非马尔可夫状态。** 如果你需要最近三次观测来决策，那么“状态”就不仅仅是当前观测。解决：堆叠帧（Atari 上的 DQN 堆叠了 4 帧）或使用递归状态（对观测做 LSTM/GRU）。
- **稀疏奖励。** 仅在获胜时才给予奖励的设计在大状态空间中几乎无法学习。构造形状化奖励（中间信号）或用模仿学习引导（Phase 9 · 09）。
- **奖励投机（Reward hacking）。** 优化代理对代理来说的代理目标常常会产生病态行为。OpenAI 的一位赛车代理为获取道具不断打圈而不是完成比赛。始终从目标结果定义奖励，而不是从代理能轻易优化的代理信号定义。
- **折扣设置错误。** 在无限时域任务上设置 `γ = 1` 会使得所有值发散为无穷。始终用有限视野或 `γ < 1` 来限定。
- **奖励尺度。** 奖励为 {+100, -100} 与 {+1, -1} 在最优策略上是相同的，但会导致梯度大小差异巨大。在把奖励输入 PPO/DQN 之前，将其归一化到大约 `[-1, 1]`。

## 使用场景

2026 年的栈将每个 RL 管道在接触代码前都先归约为一个 MDP：

| Situation | State | Action | Reward | γ |
|-----------|-------|--------|--------|---|
| 控制（行走、操纵） | 关节角度 + 速度 | 连续扭矩 | 任务相关的形状化奖励 | 0.99 |
| 游戏（国际象棋、围棋、扑克） | 棋盘 + 历史 | 合法落子 | 赢=+1 / 输=-1 | 1.0（有限回合） |
| 库存 / 定价 | 库存 + 需求 | 订货量 | 收入 - 成本 | 0.95 |
| LLM 的 RLHF | 上下文 tokens | 下一个 token | 终点处的奖励模型评分 | 1.0（回合约 ~200 token） |
| 用于推理的 GRPO | 提示 + 部分响应 | 下一个 token | 验证器在结束时给 0/1 | 1.0 |

在写任何训练循环之前先写出五元组。大多数“RL 不起作用”的 bug 报告最终都追溯到纸面上的 MDP 形式化就是错的。

## 交付

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: 给定任务描述，生成马尔可夫决策过程规范并在训练前标注形式化风险。
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

给定一个任务（控制 / 游戏 / 推荐 / LLM 微调），输出：

1. State。精确的特征向量或张量规格。说明马尔可夫性质的理由。
2. Action。离散集合或连续范围。维度说明。
3. Transition。确定性、已知模型的随机性，还是仅能采样。
4. Reward。函数与来源。稀疏还是形状化。终止时或逐步奖励。
5. Discount。数值与视野理由。

拒绝交付任何没有明确说明帧堆叠或递归状态而导致非马尔可夫的 MDP。拒绝任何不是以目标结果为定义的奖励。对无限时域任务标注并拒绝任何 `γ ≥ 1.0`。标注任何比典型步奖励大 100 倍以上的奖励范围为可能的梯度爆炸源。
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 4×4 网格世界和随机策略的回放。运行 10,000 个回合。报告回报的均值和标准差。与最优回报（-6）比较。
2. **中等。** 对均匀随机策略，用 `γ ∈ {0.5, 0.9, 0.99}` 运行 `policy_evaluation`。将 `V` 以 4×4 网格形式打印出来。解释为什么靠近终点的状态值在较大的 `γ` 下增长得更快。
3. **困难。** 将网格世界改为随机：每次动作以概率 `p = 0.1` 滑到相邻方向。重新评估均匀策略。`V[start]` 变好还是变差？为什么？

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MDP | "Reinforcement learning setup" | 元组 `(S, A, P, R, γ)`，满足马尔可夫性质。 |
| State | "What the agent sees" | 在所选策略类下对未来动力学而言的充分统计量。 |
| Policy | "Agent's behavior" | 条件分布 `π(a \| s)` 或确定性映射 `s → a`。 |
| Return | "Total reward" | 从当前步开始的折扣和 `Σ γ^t r_t`。 |
| Value | "How good a state is" | 从状态 `s` 在策略 `π` 下的期望回报。 |
| Q-value | "How good an action is" | 在策略 `π` 下以首步动作 `a` 开始时的期望回报。 |
| Bellman equation | "Dynamic programming recursion" | 将值 / Q 分解为一步奖励加上折扣后的后继值的不动点方程。 |
| Discount `γ` | "Future vs present" | 对远期奖励的几何衰减权重；有效视野约为 `~1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书。第 3 章涵盖 MDP 与贝尔曼方程；第 1 章阐述了支持后续所有课程的奖励假说。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — 贝尔曼方程的起源。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 来自深度 RL 角度的简明 MDP 入门。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 运筹学中关于 MDP 与精确求解方法的参考书。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 将 MDP 作为动态规划特例进行最清晰推导的论文。