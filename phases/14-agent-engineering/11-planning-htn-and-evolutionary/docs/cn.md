# 使用 HTN 与进化搜索进行规划

> 符号化规划解决那些计划可被证明正确的情况。进化代码搜索解决那些适用机器可检验的适应度函数的情况。ChatHTN (2025) 和 AlphaEvolve (2025) 展示了将两者与 LLM 配合时各自能释放的能力。

**Type:** 构建  
**Languages:** Python（stdlib）  
**Prerequisites:** 第14阶段 · 02（ReWOO 和 Plan-and-Execute）  
**Time:** ~75 分钟

## 学习目标

- 解释层次任务网络（Hierarchical Task Networks）：任务、方法、操作符、前置条件、效果。
- 描述 ChatHTN 的混合循环 —— 符号化搜索并在无方法时回退到 LLM 解构。
- 解释 AlphaEvolve 的进化循环以及为什么它只在存在程序化评估器时有效。
- 使用标准库实现一个玩具 HTN 规划器以及一个玩具的进化搜索。

## 问题背景

ReWOO（Lesson 02）、Plan-and-Execute 和 ReAct 覆盖了大多数智能体规划场景。但有两类情况它们不太适合：

1. 可证明正确的计划。调度、航线规划、合规工作流 —— 计划必须从构造上保证正确。一个有时会产生幻觉步骤的流式 LLM 计划是不可接受的。
2. 具有机器可检验适应度函数的优化问题。矩阵乘法、调度启发式、编译器传递 —— 目标不是“一个正确的计划”，而是“最优的计划”。

HTN 规划和 AlphaEvolve 分别解决这两类不同的问题。两者都将 LLM 用作放大器而非替代品。

## 概念

### 层次任务网络（Hierarchical Task Networks）

HTN 包含：

- **Tasks** — 复合任务（需要被分解）和原子任务（可直接执行）。
- **Methods** — 将复合任务分解为子任务的方式，并带有前置条件。
- **Operators** — 带有前置条件和效果的原子动作。
- **State** — 一组事实。

规划问题：给定一个目标任务和初始状态，找到一个分解，使之最终成为前置条件在顺序执行时都被满足的原子操作序列。

HTN 比 LLM 出现得更早，仍然是可证明正确计划的参考方法。

### ChatHTN（Gopalakrishnan 等，2025）

ChatHTN（arXiv:2505.11814）将符号化 HTN 与 LLM 查询交错使用：

1. 尝试用现有方法分解当前的复合任务。
2. 如果没有方法适用，则询问 LLM：“在状态 `s` 下你会如何分解 `task`？”
3. 将 LLM 的响应翻译成候选子任务。
4. 根据操作符模式验证；拒绝无效的分解。
5. 递归进行。

论文的核心论点：每个生成的计划都是可证明正确的，因为 LLM 的建议仅以候选分解的形式进入，从不直接修改计划。符号层负责正确性；LLM 拓展方法库。

在线方法学习（OpenReview `gwYEDY9j2x`，2025 后续工作）引入了一个学习器，通过回归将 LLM 产生的分解进行泛化 —— 将 LLM 查询频率削减高达 75%。

### AlphaEvolve（Novikov 等，2025）

AlphaEvolve（arXiv:2506.13131，DeepMind，2025 年 6 月）是另一种范式：由 Gemini 2.0 Flash/Pro 集群协同的进化代码搜索。

循环：

1. 从种子程序和一个程序化评估器（返回适应度分数）开始。
2. LLM 集群提出变异。
3. 用评估器运行这些变异。
4. 保留表现最好的；继续变异。

已发布的成果：

- 在 56 年来首次在 4x4 复数矩阵乘法上改进了 Strassen（48 次标量乘法）。
- 在 Borg 调度启发式上恢复了 0.7% 的 Google 计算效率。
- 在一条前沿工作负载上对 FlashAttention 实现了 32% 的加速。

硬约束：适应度函数必须是机器可检验的。对散文答案进行进化搜索无法收敛。

### 何时使用哪种方法

| Problem class | Use | Why |
|---------------|-----|-----|
| 有严格约束的调度 | HTN + ChatHTN | 可证明的正确性 |
| 编译器优化 | AlphaEvolve | 机器可检验的适应度 |
| 多步任务执行 | ReAct / ReWOO | LLM 在环，没有形式保证 |
| 通过测试改进代码 | AlphaEvolve | 测试用例作为评估器 |
| 受策略约束的自动化 | HTN | 前置条件编码策略 |

### 该模式的失败点

- **没有操作符的 HTN。** 如果没有前置条件/效果模式，正确性主张就崩塌。ChatHTN 的“LLM 建议分解”依赖于模式来拒绝无效操作。
- **没有真实评估器的 AlphaEvolve。** “问 LLM 哪个代码更好”并不是适应度函数。评估器必须是确定性的且快速的。
- **过度工程化。** 大多数智能体任务不需要这两个方案。优先考虑 ReAct 或 ReWOO。

## 实现它

`code/main.py` 实现了两个玩具：

- 一个使用标准库的 HTN 规划器，包含操作符、方法、前置条件、效果，以及一个在没有方法匹配时触发的 `LLMFallback`。这里的 “LLM” 是一个脚本化的分解器，以便规划器可以离线运行。
- 一个基于标准库的算术表达式进化搜索：增长表达式，使其在测试集上最小化 `|f(x) - target|`。评估器是确定性的。

运行它：

```
python3 code/main.py
```

运行跟踪会展示 HTN 规划器如何分解一个复合任务（中途使用 LLM 回退），以及进化循环如何收敛到目标表达式。

## 使用建议

- **HTN 规划器** — 使用 `pyhop`、`SHOP3`，或为领域特定的策略执法构建自己的实现。
- **ChatHTN** — 研究代码；该模式（符号层 + LLM 回退）可以无缝移植到任何 HTN 规划器。
- **AlphaEvolve** — DeepMind 的论文；该模式（集群 + 评估器）是可复现的。OpenEvolve 与类似的开源分支正在出现。
- **智能体框架** — 目前没有框架原生提供一流的 HTN 或 AlphaEvolve。将其构建为子智能体或后台任务通常更合适。

## 部署产物

`outputs/skill-hybrid-planner.md` 会生成一个混合规划器骨架（HTN 或进化式），并明确界定 LLM 的角色范围。

## 练习

1. 将 HTN 规划器扩展为支持回溯：当某个操作符的后置条件在运行时失败时，回滚并尝试下一个方法。
2. 为 ChatHTN 添加 LLM-方法缓存：当 LLM 在状态模式 `P` 下分解任务 `T` 时，存储结果。下次调用时先检查方法库。
3. 将进化搜索的评估器换成真正的测试套件。进化一个通过 20 个测试用例的排序函数；报告收敛所需的代数。
4. 阅读 AlphaEvolve 的评估器设计说明。为你关心的领域设计一个评估器（SQL 查询优化、测试套件最小化、部署 YAML 优化等）。
5. 结合两者：使用 HTN 将复合任务分解为子任务，然后对每个子任务的原子操作使用进化搜索。它在哪些场景表现出色，在哪些场景又是过度工程？

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| HTN | "Hierarchical planner" | 任务分解，包含操作符、前置条件、效果 |
| Method | "Decomposition rule" | 将复合任务拆分为子任务的方式 |
| Operator | "Primitive action" | 具有前置条件和效果的具体步骤 |
| ChatHTN | "LLM + HTN" | 在没有方法匹配时，符号化规划器向 LLM 询问分解 |
| AlphaEvolve | "Evolutionary code search" | LLM 集群变异代码；确定性评估器进行筛选 |
| Fitness function | "Evaluator" | 对输出进行确定性、机器可检验的评分 |
| Online method learning | "Cached LLM decomposition" | 存储并泛化 LLM 的分解以降低查询成本 |

## 延伸阅读

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — 符号化 + LLM 的混合规划器  
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — 具有 LLM 变异器的进化代码搜索  
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 何时使用规划器与何时采用简单循环的指南