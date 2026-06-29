# Deep Q-Networks (DQN)

> 2013：Mnih 在原始像素上训练了一个 Q-learning 网络，在七个 Atari 游戏上击败了所有经典 RL 算法。2015：扩展到 49 个游戏，并发表在 Nature，掀起了深度强化学习时代。DQN = Q-learning + 三个工程技巧，使得函数逼近稳定。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 3 · 03 (反向传播), Phase 9 · 04 (Q-learning, SARSA)  
**Time:** ~75 分钟

## 问题

表格化的 Q-learning 需要为每个 (state, action) 对维护一个单独的 Q 值。一个棋盘大约有 ~10⁴³ 个状态。一个 Atari 帧是 210×160×3 = 100,800 个特征。表格化 RL 在几千个状态时就崩溃了，更别说数十亿了。

事后看来修复很明显：用神经网络替代 Q 表，即 `Q(s, a; θ)`。但显而易见的问题花了几十年才解决。带函数逼近的朴素 Q-learning 在“致命三角”——函数逼近 + 引导 (bootstrapping) + 离策略学习 下会发散。Mnih 等人（2013，2015）确定了三项工程技巧来稳定训练：

1. **Experience replay（经验回放）** 去相关化转移样本。
2. **Target network（目标网络）** 冻结引导目标。
3. **Reward clipping（回报裁剪）** 归一化梯度幅度。

在 Atari 上的 DQN 是第一次用单一架构和单一超参集合直接从原始像素解决数十个控制问题。从此之后所有“深度 RL”的工作——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都是在这个三技巧基础上堆叠而来。

## 概念

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**目标。** DQN 在神经 Q 函数上最小化一步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络（online network），每步通过梯度下降更新。`θ^-` = 目标网络（target network），周期性从 `θ` 复制（大约每 ~10,000 步）。`D` = 存储过去转移的回放缓存（replay buffer）。

**三项技巧，按重要性排序：**

**Experience replay（经验回放）。** 一个容量为 `~10⁶` 的环形缓冲区。每次训练步骤从中均匀随机采样一个小批量。这打破了时间相关性（连续帧几乎相同），让网络可以多次从稀有的有奖励转移中学习，并去相关连续的梯度更新。没有它，带神经网的 on-policy TD 在 Atari 上会发散。

**Target network（目标网络）。** 在贝尔曼方程的两边使用相同网络 `Q(·; θ)` 会使目标在每次更新时变化 —— “追着自己的尾巴”。解决办法：保留第二个网络 `Q(·; θ^-)`，其权重被冻结。每隔 `C` 步，将 `θ → θ^-` 复制一次。这能让回归目标在成千上万步的梯度更新期间保持稳定。软更新 `θ^- ← τ θ + (1-τ) θ^-`（在 DDPG、SAC 中使用）是更平滑的变体。

**Reward clipping（回报裁剪）。** Atari 的奖励幅度从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 可以防止任何单一游戏支配梯度。当奖励幅度很重要时这是错误的；但在 Atari 中通常只在意符号，所以可行。

**Double DQN。** Hasselt（2016）修复了最大化偏差：使用在线网络来*选择*动作，使用目标网络来*评估*该动作。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

这是一个可直接替换的改进，效果稳定更好。默认启用。

**其他改进（Rainbow，2017）：** 优先回放（优先采样高 TD-error 的转换）、dueling 架构（分离 `V(s)` 与优势头）、noisy networks（可学习探测）、n-step returns、多步引导、分布式 Q（C51/QR-DQN）。每项都能带来几个百分点的提升；总体增益大致可加。

## 实现

这里的代码只使用标准库、无 numpy —— 我们在一个微小的连续 GridWorld 上用手工实现的单隐藏层 MLP，因此每个训练步骤只需微秒。算法在原则上与大规模 Atari DQN 相同。

### 第一步：回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 推荐容量 ~50,000；我们的玩具环境 5,000 就足够。

### 第二步：微小的 Q 网络（手写 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。整个网络就这么简单。

### 第三步：DQN 的更新步骤

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形式上相当于第 04 课的 Q-learning，两处不同： (a) 我们通过可微的 `Q(·; θ)` 反向传播而不是索引表格，(b) 目标使用 `Q(·; θ^-)` 。

### 第四步：外层循环

每个 episode 中，在 `Q(·; θ)` 上执行 ε-greedy 策略，向缓冲区推入转移，采样小批量，进行一次梯度更新，周期性地同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们的小型 GridWorld（16 维 one-hot 状态）上，智能体在 ~500 个 episode 内学到接近最优的策略。在 Atari 上，把规模放大到 2 亿帧并加入 CNN 特征提取器。

## 陷阱

- **Deadly triad（致命三角）。** 函数逼近 + 离策略 + 引导可能发散。DQN 用目标网络 + 回放缓冲来缓解；不要移除任意一项。
- **探索。** ε 必须衰减，通常在前 ~10% 的训练中从 1.0 下降到 0.01。早期探索不足会使 Q 网收敛到局部盆地。
- **过度估计。** 对嘈杂 Q 取 `max` 会产生向上的偏差。生产环境中务必使用 Double DQN。
- **回报尺度。** 裁剪或归一化回报；梯度幅度与回报幅度成正比。
- **回放缓冲冷启动（coldstart）。** 在缓冲区有几千个转移之前不要训练。早期在 ~20 条样本上得到的梯度会过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太稀疏 ≈ 目标过时。Atari DQN 使用 10,000 个环境步骤。经验法则：每次训练轨迹长度的 ~1/100 同步一次。
- **观测预处理。** Atari DQN 堆叠 4 帧以使状态具马尔可夫性。任何有速度信息的环境都需要帧堆叠或循环状态（recurrent state）。

## 使用场景

到 2026 年，DQN 已很少是最先进算法，但仍然是参考的离策略算法：

| Task | Method of choice | Why not DQN? |
|------|------------------|--------------|
| Discrete-action Atari-like | Rainbow DQN or Muesli | 同一框架，更多技巧。 |
| Continuous control | SAC / TD3 (Phase 9 · 07) | DQN 没有策略网络。 |
| On-policy / high-throughput | PPO (Phase 9 · 08) | 无回放缓冲；更易扩展。 |
| Offline RL | CQL / IQL / Decision Transformer | 更保守的 Q 目标，避免引导爆炸。 |
| Large discrete action spaces (recommender) | DQN with action embedding, or IMPALA | 可行；实现细节很重要。 |
| LLM RL | PPO / GRPO | 序列级别而非步级别；损失不同。 |

这些经验仍然通用。回放与目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自对弈缓冲，以及所有离线 RL 方法中。回报裁剪以优势归一化（advantage normalization）的形式在 PPO 中延续。这个架构是一个蓝图。

## 部署

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **简单。** 运行 `code/main.py`。绘制每集回报曲线。多少个 episode 后运行平均值超过 -10？
2. **中等。** 关闭目标网络（在贝尔曼目标的两边都使用在线网络）。测量训练不稳定性 —— 回报是否振荡或发散？
3. **困难。** 添加 Double DQN：用在线网络选择 `argmax a'`，用目标网络评估。比较在带噪声回报的 GridWorld 中，1,000 个 episode 后有无 Double DQN 时 `Q(s_0, best_a)` 相对于真实最优值 `V*(s_0)` 的偏差。

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| DQN | "Deep Q-learning" | 使用神经 Q 函数、回放缓冲和目标网络的 Q-learning。 |
| Experience replay | "Shuffled transitions" | 环形缓冲区；每次梯度步骤均匀采样；去相关化数据。 |
| Target network | "Frozen bootstrap" | 用于贝尔曼目标的周期性复制的 Q；稳定训练。 |
| Deadly triad | "Why RL diverges" | 函数逼近 + 引导 + 离策略 = 无收敛性保证。 |
| Double DQN | "Fix for maximization bias" | 在线网选择动作，目标网评估该动作。 |
| Dueling DQN | "V and A heads" | 将 Q 分解为 V + A - mean(A)；相同输出，但梯度流更好。 |
| Rainbow | "All the tricks" | DDQN + PER + dueling + n-step + noisy + distributional 的合成。 |
| PER | "Prioritized Replay" | 按 TD-error 大小的比例采样转移。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 2013 年的 NeurIPS 研讨会论文，掀起深度 RL 热潮。  
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 论文，49 个游戏的 DQN。  
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。  
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN。  
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 堆叠技巧的论文。  
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代讲解。  
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 关于“致命三角”（函数逼近 + 引导 + 离策略）的教材章节，说明了目标网络和回放缓冲的设计动机。  
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 用于消融研究的参考单文件 DQN 实现；与本课的从头实现一起阅读效果更好。