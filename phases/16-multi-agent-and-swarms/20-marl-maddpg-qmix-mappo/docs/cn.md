# MARL — MADDPG, QMIX, MAPPO

> 多智能体协作的强化学习传统仍在影响 2026 年的 LLM-agent 系统。**MADDPG**（Lowe 等，NeurIPS 2017，arXiv:1706.02275）引入了集中训练、去中心化执行（CTDE）：训练时每个 critic 可以看到所有智能体的状态和动作；测试时只有本地 actor 运行。适用于合作、竞争和混合场景。**QMIX**（Rashid 等，ICML 2018，arXiv:1803.11485）是带单调混合网络的价值分解；每个智能体的 Q 值组合成联合 Q，使得 `argmax` 能够干净地分布化 — 在 StarCraft Multi-Agent Challenge (SMAC) 上占主导地位。**MAPPO**（Yu 等，NeurIPS 2022，arXiv:2103.01955）是在中心化价值函数上的 PPO；在 particle-world、SMAC、Google Research Football、Hanabi 上“出乎意料地有效”，且只需最小调参。这些工作支撑了必须去中心化执行的智能体团队策略训练。MAPPO 是 **2026 年合作型 MARL 的默认基线**。本课通过一个小型网格世界玩具实现这三种模式，在进入 LLM-agent 训练前把思路吃透。

**Type:** 学习  
**Languages:** Python（标准库、小型无 NumPy 实现）  
**Prerequisites:** Phase 09（强化学习）, Phase 16 · 09（并行群体网络）  
**Time:** ~90 分钟

## 问题

LLM-agent 系统越来越多地训练用于智能体间协调的策略：何时委派、何时行动、调用哪个同伴。告诉你如何训练这类策略的文献是多智能体强化学习（MARL），它在 LLM 浪潮之前就存在，并且有一小套主流算法。

如果不了解这些模式词汇，阅读 MARL 论文会很痛苦。集中训练、去中心化执行（CTDE）、价值分解和中心化 critic 并不是流行词 — 它们是针对具体问题的具体答案：

- 独立强化学习（每个智能体单独学习）从单个智能体的视角看是非平稳的。很糟糕。
- 集中式强化学习（一个智能体控制全部）不具可扩展性并且违反执行约束。
- CTDE 拿到了两者的优点：训练时使用全局信息，部署时使用局部策略。

## 概念

### 论文常用的三个环境

- **Particle World（多智能体粒子环境）**。简单的二维物理，包含合作/竞争任务。MADDPG 的原始测试床。
- **StarCraft Multi-Agent Challenge (SMAC)。** 合作微观操作，部分观测。QMIX 的测试床。离散动作、连续状态。
- **Google Research Football、Hanabi、MPE。** MAPPO 的基线环境。

不同环境有不同的动作/观测类型，算法据此选择设计。

### MADDPG (2017) — CTDE 模式

每个智能体 `i` 有一个 actor `mu_i(o_i)`，将自己的观测映射到动作。每个智能体还有一个 critic `Q_i(x, a_1, ..., a_n)`，训练时可见所有观测和动作。actor 通过基于 critic 评估的策略梯度来更新。

```
actor 更新：    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) 在 a_i=mu_i(o_i) ]
critic 更新：   基于下一状态联合估计的 TD 更新，作用于 Q_i(x, a_1..n)
```

为什么用 CTDE：训练时我们知道所有人的动作；用它可以减少各个 critic 的方差。部署时，每个智能体只看到 `o_i` 并调用 `mu_i(o_i)`。

失败模式：critic 随着智能体数量 N 增长而膨胀（输入包含所有动作）。在没有近似的情况下，无法扩展到 ~10 个以上的智能体。

### QMIX (2018) — 价值分解

仅限合作。全局回报由每个智能体 Q 值的单调函数之和表示：

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

单调性保证了 `argmax_a Q_tot` 可以通过每个智能体独立选择 `argmax_{a_i} Q_i` 来计算。这正是你需要的**去中心化执行属性**。训练时，一个混合网络从每个智能体的 Q 生成 `Q_tot`。

QMIX 在 SMAC 上表现优异的原因：合作型 StarCraft 微操作具有同质智能体、局部观测、全局回报——非常契合价值分解假设。

失败模式：单调性约束比较严苛；有些任务的回报结构并不能单调分解（比如某个智能体为团队牺牲）。后续扩展（QTRAN、QPLEX）对其进行了放宽。

### MAPPO (2022) — 被低估的默认选择

Multi-Agent PPO：在中心化价值函数上的 PPO。每个智能体有自己的策略；所有智能体共享（或拥有各自的）价值函数，价值函数可见完整状态。Yu 等 2022 在五个基准上将 MAPPO 与 MADDPG、QMIX 及其扩展对比，发现：

- MAPPO 在 particle-world、SMAC、Google Research Football、Hanabi、MPE 上能匹配或超过离线（off-policy）MARL 方法。
- 需要的超参数调优很少。
- 训练稳定；在不同随机种子上可复现。

在这篇论文之前，社区低估了基于 on-policy 的 MARL。到 2026 年，MAPPO 成为合作 MARL 的默认基线；任何新方法都必须在它之上胜出。

### LLM-agent 工程师为什么关心

三个直接用途：

1. 路由器训练。一个元智能体选择哪个子智能体处理任务。这是一个有 N 个去中心化子智能体和一个中心化路由器的 MARL 问题。MAPPO 很适合。
2. 角色自发形成。在生成式智能体模拟中，让智能体随时间采取互补角色其实是一个隐蔽的 MARL 问题。QMIX 风格的价值分解通过构造强制互补性。
3. 多智能体工具使用。当智能体共享工具并竞争预算时，通过 CTDE 训练可以得到可部署的本地策略，尊重资源约束。

实际警告：到 2026 年，大多数生产级 LLM-agent 系统仍通过提示（prompting）来实现策略，而不是训练。当且仅当你拥有 (a) 大量交互数据、(b) 明确的回报信号、且 (c) 愿意投资训练基础设施时，才考虑 MARL。

### CTDE 作为超越强化学习的设计模式

即使不训练，CTDE 也是一个有用的架构模式：

- 在设计阶段，假定团队可见全部信息。
- 运行时，强制去中心化执行：每个智能体只看到 `o_i`。

该模式强制你显式维护每个智能体的状态，并提前考虑部分可观测性。许多生产多智能体系统在内部默认共享状态——CTDE 的纪律可以防止这种隐患。

### 非平稳性问题

当多个智能体同时学习时，每个智能体的环境（包括其他智能体的策略）是非平稳的。经典的单智能体 RL 证明不再成立。本课中的 MARL 算法都在应对这一点：

- MADDPG：全局 critic 能看到所有动作，使其值估计具有平稳性。
- QMIX：价值分解把学习移动到联合 Q 空间，在该空间中最优性有定义。
- MAPPO：中心化价值函数减缓了其他策略变化带来的方差。

在 LLM-agent 系统中，非平稳性表现为“我的智能体上个月还能正常工作，但上游的另一个智能体改了之后，我的就出问题了”。用 CTDE 训练 MARL 是有原则的解决方案；提示级修补更快但不够持久。

### 本课不涵盖的内容

实际网络训练属于 Phase 09 主题。本课构建了带脚本策略的版本，演示 CTDE、价值分解和中心化价值函数的模式，但不做梯度更新。目标是在你开始使用完整 MARL 库（PyMARL、MARLlib、RLlib multi-agent）之前把这些模式记住。

## 实现

`code/main.py` 实现了三种模式演示，全部基于一个非常小的 2 智能体合作网格世界：

- 环境：4x4 网格上有 2 个智能体和一个奖励颗粒（pellet）。如果任意一个智能体到达颗粒，回报 = 1；任务结束。
- `IndependentAgents` — 每个智能体将其他智能体视为环境。基线。
- `MADDPGStyle` — 中心化 critic 计算联合值；actor 策略基于其进行脚本化的策略改进。
- `QMIXStyle` — 使用单调混合器的价值分解。
- `MAPPOStyle` — 中心化价值函数；策略基于共享基线进行脚本式改进。

所有四种设置运行相同的回合并报告平均到达目标的步数。CTDE 变种比独立基线收敛到更短的路径。

运行：

```
python3 code/main.py
```

期望输出：独立智能体平均需要 ~6 步；CTDE 变种收敛到 ~3.5 步（4x4 网格的最优为 3 步）。即使策略是脚本化的，模式差异仍然显现。

## 使用

`outputs/skill-marl-picker.md` 是一个技能（skill），用于根据给定的多智能体任务选择 MARL 算法：合作 vs 竞争、同质 vs 异质、动作空间类型、规模、回报信号等。

## 投产建议

生产环境中很少使用 MARL。若确实要用：

- **从 MAPPO 开始。** 2022 年的论文把它确立为基线；先复现它可以省下数周时间。
- **记录每个智能体的观测和动作流。** 没有逐智能体的追踪日志，调试 MARL 几乎不可能。
- **把训练代码和执行代码分离。** CTDE 是一种纪律；让执行路径真正只看到 `o_i`。
- **奖励塑形警告。** MARL 对奖励设计极其敏感。一个协调上的错误塑形会被智能体学会利用。进行对抗性测试。
- **对于 LLM 智能体**，优先考虑提示级策略。只有在交互数据 + 回报信号 + 基础设施都具备时，才投入 MARL 训练。

## 练习

1. 运行 `code/main.py`。测量独立智能体与 MAPPO 风格智能体在到达目标步数上的差距。在 6x6 网格上该差距是增大还是缩小？
2. 实现一个竞争性变体：两名智能体、一个颗粒，只有先到者获得回报。哪种模式能更好地处理竞争？历史上 MADDPG 擅长此类问题。
3. 阅读 MADDPG（arXiv:1706.02275）第 3 节。用伪代码按你自己的语言实现精确的 critic 更新规则。
4. 阅读 MAPPO（arXiv:2103.01955）。作者为何认为带中心化价值的 PPO 在他们的基准上能击败离线 MARL？列出三条最有力的论点。
5. 将 CTDE 作为设计模式应用到一个假设的 LLM-agent 系统（例如：研究智能体 + 摘要器 + 编码器）。在设计阶段可以获得哪些联合信息，而这些信息在运行时不可获得？

## 术语表

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MARL | "Multi-Agent RL" | 多智能体系统的强化学习。 |
| CTDE | "Centralized Training, Decentralized Execution" | 使用全局信息训练；部署时使用本地策略。 |
| MADDPG | "Multi-Agent DDPG" | CTDE，per-agent critic 可见所有观测与动作。 |
| QMIX | "Value decomposition" | 对每个智能体 Q 的单调混合。仅限合作。 |
| MAPPO | "Multi-Agent PPO" | 在中心化价值函数上的 PPO。2026 年的默认基线。 |
| Value decomposition | "Sum of individual Qs" | 联合 Q 被表示为 per-agent Q 的单调函数。 |
| Non-stationarity | "Moving targets" | 随着其他智能体学习，每个智能体的环境在变化。MARL 的核心问题。 |
| On-policy / off-policy | "Learn from current / replay" | PPO 是 on-policy（MAPPO）；DDPG 和 Q-learning 是 off-policy。 |
| SMAC | "StarCraft Multi-Agent Challenge" | 合作微观操作基准；QMIX 的试验场。 |

## 延伸阅读

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG；NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX；ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO；NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — 对 MAPPO 结果的易读解读
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge 仓库