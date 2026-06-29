# 多智能体强化学习

> 单智能体强化学习假设环境是平稳的。把两个学习中的智能体放在同一个世界里，这一假设就会被打破：每个智能体都是另一个智能体环境的一部分，并且两者都在变化。多智能体强化学习（Multi-Agent RL，MARL）是一系列在马尔可夫假设不再成立时使学习收敛的技巧。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)  
**Time:** ~45 分钟

## 问题

一个学习在房间中导航的机器人是单智能体 RL 问题。一支足球队不是。AlphaStar 对抗 StarCraft 对手不是。一个竞价代理的市场不是。两个汽车在四路停让处协商也不是。许多真实世界的多对多问题都不是单智能体问题。

在每个多智能体场景中，从任一智能体的角度看，其他智能体就是环境的一部分。随着它们学习并改变行为，环境变得非平稳。马尔可夫性——“下一个状态只依赖当前状态和我的动作”——被破坏，因为下一个状态还依赖于*其他*智能体选择了什么，而它们的策略是移动的目标。

这会打破表格 Q 学习的收敛性证明（Q-learning 的保证假设环境是平稳的）。它也会破坏简单的深度 RL：智能体互相追逐，形成循环，永远无法收敛到稳定策略。你需要针对多智能体的技术：集中训练/分散执行（CTDE），反事实基线，联赛训练，自我对弈等。

2026 年的应用：机器人蜂群、交通路由、自动驾驶车队、市场模拟、多智能体大型模型系统（Phase 16），以及任何有多个智能玩家的游戏。

## 概念

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**形式化：马尔可夫博弈（Markov Game）。** 是 MDP 的推广：状态集 `S`，联合动作 `a = (a_1, …, a_n)`，转移概率 `P(s' | s, a)`，以及每个智能体的奖励函数 `R_i(s, a, s')`。每个智能体 `i` 在其策略 `π_i` 下最大化自身回报。如果奖励相同，则属于**完全合作**。若为零和，则是**对抗性**。若混合，则为**一般和**（general-sum）。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的视角看，`P(s' | s, a_i)` 依赖于 `π_{-i}`，而后者在变化。
- **归因（Credit assignment）。** 在共享奖励下，哪个智能体造成了该奖励？
- **探索协调。** 智能体必须探索互补的策略，而不是重复性地探索相同状态。
- **可扩展性。** 联合动作空间随着 `n` 指数增长。
- **部分可观测性。** 每个智能体只能看到自身观察，整体全局状态被隐藏。

**四种主流范式：**

1. 独立 Q 学习 / 独立 PPO（IQL, IPPO）。每个智能体学习自己的 Q 或策略，将其他智能体视为环境的一部分。简单，有时有效（尤其是经验回放作为平滑代理-建模的技巧时）。理论收敛性：无。在实践中：对松耦合任务表现良好，对紧耦合任务表现糟糕。

2. 集中训练，分散执行（CTDE）。这是大多数现代范式。每个智能体有自己的策略 `π_i`，条件于本地观测 `o_i` —— 部署时的标准分散执行。在*训练*阶段，使用集中式 critic `Q(s, a_1, …, a_n)`，条件于完整全局状态和联合动作。示例：
   - **MADDPG**（Lowe 等，2017）：在每个智能体上使用集中 critic 的 DDPG。
   - **COMA**（Foerster 等，2017）：反事实基线 —— 问“如果我采取动作 `a'`，我的奖励会是多少？”——以此隔离我的贡献。
   - **MAPPO** / **IPPO**（Yu 等，2022）：带共享 critic 的 PPO；在 2026 年的合作 MARL 中占主导。
   - **QMIX**（Rashid 等，2018）：值分解 —— `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，采用单调混合。

3. 自我对弈（Self-play）。两个相同智能体的副本相互对弈。对手的策略就是我过去快照中的策略。AlphaGo / AlphaZero / MuZero、OpenAI Five。对零和游戏最有效；训练信号是对称的。

4. 联赛训练（League play）。将自我对弈扩展到一般和/对抗性环境：保持一组过去和当前策略，从联赛中抽样对手进行训练。加入 exploiters（专门针对当前最优者）和 main exploiters（专门针对 exploiters）。AlphaStar（星际争霸 II）。当游戏存在“石头剪刀布”式策略循环时，这是必须的。

**通信。** 允许智能体之间发送学习得到的消息 `m_i`。在合作场景中有效。Foerster 等（2016）展示了可微分的智能体间通信可以端到端训练。如今基于 LLM 的多智能体系统（Phase 16）本质上使用自然语言进行通信。

## 实现

本课使用一个 6×6 的 GridWorld，包含两个合作智能体。它们从相对的角落开始，必须到达共享目标。共享奖励：只要任一智能体仍在移动，每步 `-1`，两者都到达时 `+10`。见 `code/main.py`。

### 步骤 1：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态是两个位置的组合。

### 步骤 2：独立 Q 学习

每个智能体运行它自己的以联合状态为键的 Q 表。每一步：双方都按 ε-greedy 选择动作，收集联合转移，每个智能体用共享奖励更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在这个任务上可行，因为奖励稠密且对齐。在紧耦合任务（例如需要某一智能体*等待*另一智能体的情况）上则会失败。

### 步骤 3：集中 Q 与值分解式更新

使用一个关于联合动作的 Q：`Q(s, a_1, a_2)`。用共享奖励进行更新。执行时去中心化，通过边缘化来解耦：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。用正确的全局视角换取了指数级的联合动作空间代价。

### 步骤 4：简单自我对弈（对抗性两智能体）

同一智能体、两种角色。训练智能体 A 对抗智能体 B；每隔 `K` 集合，将 A 的权重复制给 B。对称训练，进展一致。是 AlphaZero 方案的缩影。

## 陷阱

- **非平稳的回放。** 对于独立智能体来说，经验回放比单智能体更糟，因为旧的转移是由现已过时的对手生成的。修复：重标记（relabel）或按新近程度加权。
- **归因模糊。** 长回合后的共享奖励；无法明确指出哪个智能体做出了贡献。修复：反事实基线（COMA），或对每个智能体进行奖励塑形。
- **策略漂移 / 互相追逐。** 每个智能体的最佳回应随着彼此更新而变化。修复：集中式 critic、降低学习率，或一次只冻结一个智能体。
- **通过协同进行奖励滥用。** 智能体发现设计者未预见的协同漏洞。拍卖中的智能体可能收敛于出价 0。修复：谨慎设计奖励、行为约束。
- **探索冗余。** 双方探索相同的状态-动作对。修复：对每个智能体添加熵奖励，或进行角色条件化（role-conditioning）。
- **联赛循环。** 纯自我对弈可能陷入支配循环。修复：通过多样化对手的联赛训练。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。用函数逼近近似；分解动作空间（为每个智能体使用独立的策略输出头）。

## 使用场景

2026 年 MARL 的应用地图：

| Domain | Method | Notes |
|--------|--------|-------|
| Cooperative navigation / manipulation | MAPPO / QMIX | CTDE；共享 critic + 分散 actor。 |
| Two-player games (chess, Go, poker) | Self-play with MCTS (AlphaZero) | 零和；对称训练。 |
| Complex multiplayer (Dota, StarCraft) | League play + imitation pretraining | OpenAI Five, AlphaStar。 |
| Autonomous-vehicle fleets | CTDE MAPPO / PPO with attention | 部分观测；队伍规模可变。 |
| Auction markets | Game-theoretic equilibrium + RL | 当 `n` → ∞ 时使用均值场 RL（Mean-field RL）。 |
| LLM multi-agent systems (Phase 16) | Natural-language comm + role conditioning | 在代理规划层面使用 RL 回路，轨迹级别的偏好优化，而非令牌级别（Phase 16 · 03）。 |

在 2026 年，MARL 最大的增长领域是基于 LLM 的系统：语言模型代理蜂群进行协商、辩论、构建软件。RL 出现在对轨迹级输出的偏好优化上，而不是对单个 token 的优化（Phase 16 · 03）。

## 部署

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. 简单题。 在 2 智能体合作 GridWorld 上训练独立 Q 学习。平均回报 > 0 需要多少集（episodes）？绘制联合学习曲线。
2. 中等题。 增加一个“协调”任务：只有当两名智能体在同一回合同时踏上目标时才算到达。独立 Q 仍然会收敛吗？问题出在哪里？
3. 困难题。 为 MAPPO 风格训练实现一个集中式 critic，并将其在协调任务上的收敛速度与独立 PPO 进行比较。

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Markov game | "Multi-agent MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有自己的奖励。 |
| CTDE | "Centralized training, decentralized execution" | 训练时使用联合 critic；每个智能体的策略仅使用本地观测。 |
| IPPO | "Independent PPO" | 每个智能体独立运行 PPO。简单的基线；常被低估。 |
| MAPPO | "Multi-agent PPO" | 带集中式值函数（基于全局状态）的 PPO。 |
| QMIX | "Monotonic value decomposition" | `Q_tot = f_monotone(Q_1, …, Q_n)`，允许分散的 argmax。 |
| COMA | "Counterfactual multi-agent" | 优势函数 = 我的 Q 减去对我的动作进行边缘化后的期望 Q（反事实基线）。 |
| Self-play | "Agent vs past self" | 单智能体、两种角色；零和游戏的标准做法。 |
| League play | "Population training" | 缓存过去的策略，从池中抽样对手；处理策略循环问题。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — CTDE，带集中式 critic。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 用于归因的反事实基线。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 具有单调性的值分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 中出乎意料地强大。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模的联赛训练。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 在零和游戏中纯自我对弈。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 包括教科书对多智能体场景和非平稳性问题的简短讨论，以及 CTDE 设计的动机。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 覆盖合作、竞争和混合 MARL 的综述，包含收敛性结果。