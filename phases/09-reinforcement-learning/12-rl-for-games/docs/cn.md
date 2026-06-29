# RL for Games — AlphaZero, MuZero, and the LLM-Reasoning Era

> 1992: TD-Gammon beat human champions at backgammon with pure TD. 2016: AlphaGo beat Lee Sedol. 2017: AlphaZero dominated chess, shogi, and Go from scratch. 2024: DeepSeek-R1 proved the same recipe, with GRPO replacing PPO, works on reasoning. Games are the benchmark that drives every breakthrough in this phase.

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** 阶段 9 · 05 (DQN), 阶段 9 · 08 (PPO), 阶段 9 · 09 (RLHF), 阶段 9 · 10 (MARL)  
**Time:** ~120 分钟

## 问题概述

游戏具备强化学习所需的一切。清晰的奖励（胜/负）。无限的回合（自对弈可以重置）。完美的模拟（游戏本身即模拟器）。离散或小规模连续动作空间。多智能体结构促使对抗性鲁棒性。

而且每一次主要的 RL 突破几乎都在游戏上验证过。TD-Gammon（双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（星际争霸 II，2019）。MuZero（学习模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）——最新的展示表明游戏类 RL 技术同样适用于文本任务。

本结业项目通过一个统一的视角审视三个里程碑式架构——AlphaZero、MuZero 和 GRPO：自对弈 + 搜索 + 策略改进。每个都对前者进行推广；尤其是 GRPO，将 AlphaZero 的配方应用到 LLM 推理中，把 token 当作动作，把数学验证当作胜利信号。

## 概念

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017)。** Silver 等。给定一个已知规则的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：一个塔式网络 `f_θ(s) → (p, v)`。`p` 是对合法走子的先验分布。`v` 是预期的游戏结果值。
- 蒙特卡洛树搜索 (MCTS)：在每一步扩展可能延续的树。用 `(p, v)` 作为先验并做引导（bootstrap）。按 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自对弈：代理彼此对弈。在第 t 步，MCTS 的访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人工知识。零手工启发式。单一配方在各自数千万次自对弈后征服了棋类世界。

**MuZero (2019)。** Schrittwieser 等。移除了已知规则的要求。

- 不再依赖固定环境，改为学习一个*潜在动力学模型*`(h, g, f)`：
  - `h(s)`：将观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 值。
- MCTS 在*学习到的潜在空间*中运行。搜索、训练循环同上。
- 在围棋、国际象棋、将棋和 Atari 上都有效——同一算法，无需规则知识。

**随机 MuZero (2022)。** 增加了随机动力学和机会节点；扩展到类似双陆棋的带随机性的游戏。

**Muesli、Gumbel MuZero (2022-2024)。** 在样本效率和确定性搜索上做了改进。

**GRPO (2024-2025)。** DeepSeek-R1 的配方。相同的 AlphaZero 形循环，但应用于语言模型推理：

- “游戏”：回答数学/编程/推理题目。“赢” = 验证器（单元测试通过或数值答案匹配）返回 1。
- 策略：LLM。动作：tokens。状态：提示词 + 当前已生成的回应。
- 没有 critic（PPO 式的 V_φ）。对于每个提示，从策略中抽样 G 个完成样本。对每个样本计算奖励。使用**组相对优势**（group-relative advantage）`A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 式更新的信号。
- 加入对参考策略的 KL 惩罚以防止策略漂移（类似 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

没有奖励模型、没有 critic、没有 MCTS。组相对基线替代了三者。在推理基准上以远低于 PPO 的计算成本匹配或超过了 PPO-RLHF 的质量。

**完整的 R1 配方。** DeepSeek-R1（DeepSeek 2025）在论文中包含两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基础模型开始。不做 SFT。直接用 GRPO，两个奖励分量：*准确性奖励*（基于规则——最终答案解析为正确数值 / 代码通过单元测试）和*格式奖励*（完成是否用 `<think>…</think>` 标签包裹思维链）。经过数千步后，平均响应长度从 ~100 增长到 ~10,000 token，数学基准分数达到接近 o1-preview 的水平。模型学会了从零开始推理。缺点是其思维链常常难以阅读，混杂语言，缺乏风格打磨。
- **R1。** 通过四阶段流水线修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集几千条格式良好的长 CoT（长思维链）示例。对基础模型做监督微调，得到可读的起点。
  2. **面向推理的 GRPO。** 用准确性+格式奖励加上*语言一致性*奖励以防止语言切换，应用 GRPO。
  3. **拒绝采样 + 二轮 SFT。** 从 RL 检查点采样 ~60 万条推理轨迹，仅保留最终答案正确且 CoT 可读的样本，并与 ~20 万条非推理 SFT 示例（写作、问答、自我认知）合并。再次微调基础模型。
  4. **全谱 GRPO。** 再做一轮 RL，覆盖推理（基于规则的奖励）和一般对齐（有益性/无害性偏好式奖励）。

结果在 AIME 和 MATH-500 上与 o1 匹配并开源权重，且体积足够小以便蒸馏。论文还发布了六个通过 SFT 从 R1 推理轨迹蒸馏得到的密集模型（Qwen-1.5B 到 Llama-70B）——学生模型没有用 RL。强 RL 教师的蒸馏在学生规模上持续优于从头开始的 RL。

**为什么在推理上用 GRPO 而不是 PPO。** DeepSeekMath 论文（2024 年 2 月）给出三点理由：(1) 不需要训练值网络，内存减半；(2) 组基线自然处理推理任务中稀疏的回合末奖励；(3) per-prompt 归一化使得不同难度问题上的优势可比，而 PPO 的单一 critic 做不到这一点。

**无搜索 vs 有搜索。** 游戏类方向分叉：

- *完美信息、长时程游戏*（围棋、国际象棋）：仍然依赖搜索。AlphaZero / MuZero 主导。
- *LLM 推理*：生产环境中尚未看到 MCTS；采用 GRPO 的整段 rollout、best-of-N 推理计算。过程奖励模型（PRMs）提示步骤级的搜索可能会被重新加入。

## 实现

`code/main.py` 中的代码实现了一个“微型 GRPO”——一个带有多个样本组的带臂问题（bandit）。算法与 LLM 上的完全相同；唯一不同的是策略和环境更简单。它演示了*损失*和*组相对优势*，这是 2025 年的创新点。

### 第 1 步：一个微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实的 GRPO 中，验证器会运行单元测试或检查数学等式是否成立。

### 第 2 步：策略：对每个提示在 K 个答案 token 上做 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于 LLM 在给定提示下的最终层输出。

### 第 3 步：组采样与组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 2024 年 DeepSeek 的技巧。不需要 critic。“基线”是组均值，归一化使用组标准差。

### 第 4 步：与无值的 REINFORCE 基线比较

相同设置、相同计算，使用普通 REINFORCE。GRPO 在收敛速度和稳定性上更优。

### 第 5 步：观察熵和 KL

与 RLHF 相同的诊断：到参考策略的平均 KL、策略熵、随时间的奖励。一旦这些指标稳定，训练就可以结束。

## 陷阱

- **通过欺骗验证器获利（Reward hacking）。** GRPO 继承了 RLHF 的风险：如果验证器有误或可被利用，LLM 会找到利用方法。需要稳健的验证器（多组测试用例、形式化证明）。
- **组大小过小。** 组基线的方差随 `1/√G` 变化。低于 `G = 4` 时优势信号会很嘈杂；常用选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的 LLM 完成项具有不同的对数概率。按 token 数归一化、或使用序列级别对数概率、或截断到最大长度。
- **纯自对弈循环。** AlphaZero 式训练在一般和棋（general-sum）游戏上可能陷入支配循环。通过多样化对手池（league play，见 Lesson 10）缓解。
- **搜索-策略错配。** AlphaZero 将策略训练为模仿搜索输出。如果策略网络太小无法表示搜索分布，训练会停滞。
- **计算门槛。** MuZero / AlphaZero 需要海量计算。单次消融实验通常需数百 GPU 小时。存在微型演示（例如在四连棋上运行 AlphaZero）以做学习示例。
- **验证器覆盖率。** 对有漏洞的解法通过的单元测试会强化该漏洞。设计能捕捉边缘情况的验证器非常重要。

## 使用场景

到 2026 年，按领域划分的主导方法：

| Domain | Dominant method |
|--------|-----------------|
| Two-player zero-sum board games (Go, chess, shogi) | AlphaZero / MuZero / KataGo |
| Imperfect info card games (poker) | CFR + deep learning (DeepStack, Libratus, Pluribus) |
| Atari / pixel games | Muesli / MuZero / IMPALA-PPO |
| Large multiplayer strategy (Dota, StarCraft) | PPO + self-play + league (OpenAI Five, AlphaStar) |
| LLM math/code reasoning | GRPO (DeepSeek-R1, Qwen-RL, open replications) |
| LLM alignment | DPO / RLHF-PPO (not GRPO; verifier is preference not verifiable) |
| Robotics | PPO + DR (not game-RL, but uses same policy-gradient tools) |
| Combinatorial problems | AlphaZero variants (AlphaTensor, AlphaDev) |

这套*配方*——自对弈、基于搜索的改进、策略蒸馏——横跨文本、像素与物理控制。GRPO 是最新的实例；未来还会有更多类似方法出现。

## 发布（Ship It）

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO bandit。针对 2 个提示 × 每个提示 4 个答案 token 进行训练。在 `G=8` 的设置下于 < 1000 次更新内收敛。
2. **中等。** 接入 PPO（裁剪式）和普通 REINFORCE。比较在相同 bandit 上 GRPO 的样本效率和奖励方差。
3. **困难。** 扩展到长度为 2 的“推理链”：智能体输出两 token，验证器对这个 token 对给予奖励。衡量 GRPO 如何处理两步序列的归因问题。（提示：对*完整序列*计算组优势，并将该优势传播到两个 token 位置。）

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MCTS | "Tree search with learned net" | 蒙特卡洛树搜索；用学习到的 `(p, v)` 先验的 UCB1/PUCT 选择。 |
| AlphaZero | "Self-play + MCTS" | 策略-价值网络被训练为匹配 MCTS 访问分布和游戏结果。 |
| MuZero | "Learned-model AlphaZero" | 同一循环，但通过学习到的动力学在潜在空间中运行。 |
| GRPO | "Critic-free PPO" | Group Relative Policy Optimization；带组均值基线 + KL 的 REINFORCE。 |
| PUCT | "AlphaZero's UCB" | `Q + c · p · √N / (1 + N_a)` — 在价值估计与先验之间平衡。 |
| Self-play | "Agent vs past self" | 零和问题的标准做法；对称的训练信号。 |
| League play | "Population-based self-play" | 将过去的、当前的与专门的 exploiters 作为对手的采样策略。 |
| Verifier reward | "Verifiable RL" | 奖励来自确定性的检查器（测试通过、答案匹配）。 |
| Process reward | "PRM" | 为每个推理步骤评分，而不仅仅评价最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 引入 GRPO 与组相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整的四阶段 R1 配方以及 R1-Zero 消融实验。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + 大规模深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开始这一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 在自定义奖励下应用 GRPO 的生产参考实现。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 在多个规模上复现 R1 配方的开源实现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书中关于自对弈、搜索和“设计奖励”的框架，R1 在 LLM 规模上实现了这些思想。