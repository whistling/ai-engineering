# Sim-to-Real Transfer

> 在模拟器中训练而在硬件上失败的策略，是记住了模拟器的策略。域随机化、域适配与系统识别是让学习到的控制器跨越现实差距的三把工具。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)  
**Time:** ~45 分钟

## 问题

在真实机器人上训练既慢又危险且昂贵。双足机器人学会走路可能需要数百万次训练回合；真实双足一旦跌倒就可能损坏硬件。模拟提供了无限重置、确定性可复现、并行环境以及零物理损坏的优势。

但模拟器是有误差的。轴承的摩擦比 MuJoCo 模型更大；相机有镜头畸变而模拟器未建模；马达有延迟、齿隙和饱和，这 99% 的仿真模型都省略了。风、灰尘和变化的光照会破坏在无菌渲染上训练出的策略。现实差距（reality gap）——模拟分布与真实分布之间的系统性差异——是机器人部署强化学习的核心问题。

你需要一个对 sim-to-real 分布漂移鲁棒的策略。有三种历史方法：对模拟器做随机化（域随机化），用少量真实数据对策略做适配（域适配 / 微调），或识别真实系统参数并匹配（系统识别）。到 2026 年，主流方案是把三者结合并配合大规模并行模拟（Isaac Sim、Isaac Lab、Mujoco MJX 在 GPU 上运行）。

## 概念

![三种 sim-to-real 机制：域随机化、域适配、系统识别](../assets/sim-to-real.svg)

**域随机化 (DR)。** Tobin 等，2017；Peng 等，2018。在训练期间随机化所有可能与真实机器人不同的模拟参数：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位置、照明、纹理、碰撞模型。策略学习到一个关于“今天处于哪个模拟”的条件分布，并在整个跨度上泛化。如果真实机器人落在训练包络内，策略就能工作。

- 优点：不需要真实数据。一套配方，多种机器人可用。
- 缺点：过度随机化会产生“通用”但过于谨慎的策略。噪声过多 ≈ 正则化过强。

**系统识别 (SI)。** 在训练前用真实世界数据拟合模拟器参数。如果你能测量真实机器人的臂关节摩擦，就把它塞入模拟器，然后训练期望这些值的策略。需要访问真实系统，但能直接减少现实差距。

- 优点：目标精确、噪声低。
- 缺点：残余的模型误差对策略不可见；未识别的小效应（例如马达死区）仍会破坏部署。

**域适配。** 在模拟中训练，然后用少量真实数据微调。两种形式：

- **Real2Sim2Real：** 使用真实 rollout 学习残差模拟 `f(s, a, z) - f_sim(s, a)`，在校正后的模拟中训练。用不多的真实数据就能缩小差距。
- **观测适配：** 训练一个把真实观测映射为类似模拟观测的策略（例如，GAN 像素到像素）。控制器仍在模拟空间运行。

**特权学习 / 教师-学生。** Miki 等，2022（ANYmal 四足机器人）。在模拟中训练一个拥有特权信息（真实摩擦、地形高度、IMU 漂移）的教师，然后蒸馏出只看到真实传感器观测的学生。学生学会从历史中推断特权特征，从而在物理参数变化下鲁棒。

**大规模并行模拟。** 2024–2026。Isaac Lab、Mujoco MJX、Brax 在单个 GPU 上运行成千上万并行机器人。使用 4,096 个并行人形的 PPO 在数小时内收集多年经验。随着训练分布的扩展，现实差距缩小；当每个 4,096 个环境都有不同随机参数时，DR 的代价几乎为零。

**2026 年现实配方（四足行走示例）：**

1. 大规模并行模拟，域随机化重力、摩擦、电机增益、载荷等参数。
2. 用特权信息（地形图、身体速度真值）训练教师策略。
3. 用仅含本体感知（腿关节编码器）的学生策略从教师蒸馏而来。
4. 可选：在真实 IMU 上用自编码器做观测适配。
5. 部署。对 10+ 环境零样本适配。如失败，用受安全约束的 PPO 做几分钟的真实世界微调。

## 构建示例

本课代码是一个在有“噪声”转移的 GridWorld 上演示域随机化的小示例。我们训练一个在“模拟”中经历随机化滑移概率（slip）的策略，并在“真实”上用它从未见过的滑移水平进行评估。该结构与 MuJoCo 到硬件的迁移直接对应。

### 步骤 1：参数化模拟

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是模拟器暴露的一个参数。在真实机器人中它可以是摩擦、质量、电机增益 —— 任何在模拟与真实间改变的量。

### 步骤 2：用 DR 训练

在每个回合开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q-learning / 任意算法。进行很多回合。

### 步骤 3：在“真实”滑移上零样本评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个值在训练支持内；`0.5` 和 `0.7` 在支持之外。DR 训练出的策略应在支持内保持近最优，并在支持外平滑退化。单一滑移训练的策略在训练滑移之外会变得脆弱。

### 步骤 4：与窄训练比较

训练第二个仅在 `slip = 0.0` 下的策略。在相同的滑移序列上评估。你应当看到一旦真实滑移 > 0 就出现灾难性下降。

## 陷阱

- 太多随机化。若在 `slip ∈ [0, 0.9]` 上训练，策略会变得过于保守而永远不尝试最优路径。匹配预期的真实世界分布，而不是“任何事情都有可能发生”。
- 太少随机化。在狭窄区间训练，策略根本无法泛化。使用自适应课程（Automatic Domain Randomization）在策略提升时扩展分布范围。
- 参数空间识别错误。随机化错误的东西（例如在真实差距是马达延迟时随机化相机色相）不会起作用。先对真实机器人做剖析（profiling）。
- 特权信息泄露。一个使用全局状态来决定动作的教师，可能会产生学生无法追赶的行为。确保教师的策略在学生给定观测历史下是可实现的。
- sim-to-sim 转移失败。如果策略对更难的模拟变体都不鲁棒，那么对真实世界也不会鲁棒。在部署前始终在保留的模拟变体上测试。
- 缺乏真实世界安全包。一个在模拟中工作并且“在真实中也工作”的策略，若没有低级安全屏障，仍可能损坏硬件。增加速率限制、扭矩限制、关节限制等非学习控制器。

## 应用场景

2026 年的 sim-to-real 技术栈：

| Domain | Stack |
|--------|-------|
| 腿式行走 (ANYmal, Spot, humanoid) | Isaac Lab + DR + 特权教师/学生 |
| 操作（灵巧手、抓取放置） | Isaac Lab + DR + 用于视觉的 DR-GAN |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线自适应 |
| 手指 / 手内操作 | OpenAI Dactyl（前所未有规模的 DR） |
| 工业机械臂 | MuJoCo-Warp + SI + 少量真实微调 |

在各类控制任务中，工作流一致：尽可能拟合模拟；对无法拟合的部分做随机化；训练超大策略；蒸馏；以安全护盾部署。

## 部署清单

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. 简单：在固定滑移的 GridWorld（slip=0.0）上训练 Q-learning 代理。在 `slip ∈ {0.0, 0.1, 0.3, 0.5}` 上评估。绘制 return vs slip 曲线。
2. 中等：训练一个 DR Q-learning 代理，采样 `slip ~ Uniform[0, 0.3]`。评估相同滑移序列。DR 在滑移=0.5（分布外）处带来了多少性能提升？
3. 困难：实现课程学习：从 slip=0.0 开始，每当策略达到最优的 90% 时就扩大 DR 范围。比较到达对 slip=0.3 的零样本能力所需的总环境步数与固定 DR 基线。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 现实差距 (Reality gap) | “sim-to-real difference” | 训练与部署物理/感知之间的分布漂移。 |
| 域随机化 (Domain randomization, DR) | “在随机模拟中训练” | 在训练期间随机化模拟参数以使策略泛化。 |
| 系统识别 (System identification, SI) | “测量真实并拟合模拟” | 估计真实物理参数并让模拟匹配它们。 |
| 域适配 (Domain adaptation) | “在真实数据上微调” | 在模拟训练后用少量真实数据微调；可适配观测或动力学。 |
| 特权信息 (Privileged info) | “教师的真值信息” | 模拟独有的信息；学生必须从观测历史中推断它。 |
| 教师/学生 (Teacher/student) | “把特权信息蒸馏给学生” | 教师使用捷径训练；学生学会在无捷径情况下模仿。 |
| ADR | “Automatic Domain Randomization” | 随着策略改进而扩展 DR 范围的课程方法。 |
| Real2Sim | “用真实数据缩小差距” | 学习残差以使模拟更像真实 rollout。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 原始的 DR 论文（机器人视觉方向）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 动力学领域的 DR，四足行走。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl，规模化 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal 的教师-学生方法。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 推动 2025–2026 部署的大规模并行模拟。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR 的课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 支撑现代 sim-to-real 流程的 Dyna 框架（用模型进行规划 + rollout）。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — 关于机器人深度强化学习中 sim-to-real 方法的综述与基准结果。