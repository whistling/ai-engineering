# AlphaEvolve — 进化式代码代理

> 将一个前沿的编程模型与一个进化循环和可机器检验的评估器配对。让循环运行足够久。它发现了一个用于 4x4 复数矩阵乘法的过程，使用 48 次标量乘法——这是 56 年来对 Strassen 的首次改进。它还找到了一条 Google 级的 Borg 调度启发式，在生产环境中恢复了约 0.7% 的集群计算资源。该架构故意保持朴素。真正的胜利来自评估器的严谨性。

**Type:** 学习  
**Languages:** Python（stdlib，进化循环示例）  
**Prerequisites:** Phase 15 · 01（长时程任务构架），Phase 15 · 02（自学式推理）  
**Time:** ~60 分钟

## 问题

大型语言模型可以编写代码。进化算法可以在代码空间中搜索。两者在过去几十年里分别被尝试过；两者都遇到了天花板。LLM 的天花板是虚构（confabulation）：模型会写出看似合理但并不按其所述方式工作的代码。进化算法的天花板是搜索成本：语法层面的随机变异很少能产生可编译的程序，更别提更优的程序了。

AlphaEvolve（Novikov 等，DeepMind，arXiv:2506.13131，2025 年 6 月）将两者结合。LLM 对程序库提出有针对性的修改；自动评估器为每个变体打分；得分高的变体成为下一代的父代。LLM 负责那个昂贵的部分——写出看起来合理的代码；评估器抓住虚构。循环运行数小时到数周。

报告的结果：用于 4x4 复数矩阵乘法的 48 次标量乘法（Strassen 在 1969 年的下界是 49）、在 Google 生产环境中恢复约 0.7% 集群计算的 Borg 调度启发式、FlashAttention 内核加速 32.5%、Gemini 训练吞吐量提升等。

该架构之所以有效，是因为评估器是可机器检验的。在评估器不存在或不严谨的领域，它就无效。这种不对称是本工作的教训。

## 概念

### 循环

1. 从一个正确但不理想的种子程序 `P_0` 开始。
2. 维护一个变体程序数据库，每个变体由评估器评分。
3. 从数据库中采样一个或多个父代（MAP-elites 风格或岛模型）。
4. 提示 LLM（对大量候选使用 Gemini Flash，对难题使用 Gemini Pro）生成父代的修改变体。
5. 编译、运行并在预留评估器上评估该变体。
6. 根据其分数与特征向量将其插入数据库。
7. 重复。

两个细节很重要。首先，提示 LLM 时不仅给出父程序——通常还会提供数据库中的若干顶级变体、评估器签名以及简短的任务描述。模型的任务是提出一个可能提高得分的有针对性修改。其次，数据库是结构化的（MAP-elites 网格、岛模型），以便循环探索多样性，而不是只追随当前的 leader。

### 为什么评估器不可或缺

AlphaEvolve 的胜利都来自评估器快速、确定且难以被规避的领域：

- **矩阵乘法算法**：单元测试对矩阵进行乘法并逐位比对相等性。
- **Borg 调度启发式**：生产级模拟器重放历史集群负载并衡量浪费的计算量。
- **FlashAttention 内核**：正确性测试 + 在真实硬件上的壁钟基准。
- **Gemini 训练吞吐量**：以每步 GPU-秒数测量。

在每种情况下，评估器都能捕获那些会主导问题的 LLM 错误类别：虚构的正确性声明、在硬件上消失的性能声称以及边缘情况失败。去掉评估器，循环就会优化“好看”的代码。

### 奖励劫持是这句话的另一面

进化会优化评估器所衡量的东西。如果评估器不完善，循环会找到并利用不完善之处。在未验证领域，循环会优化表面特征而不是目标行为。DeepMind 在论文中明确指出：AlphaEvolve 的成功仅能迁移到评估器严谨性与搜索野心相匹配的领域。

2025–2026 年间代码搜索循环中具体的奖励劫持例子：

- 将“完成时间”作为目标的优化会鼓励提交空解法以获得高分。
- 奖励“通过测试”的基准会鼓励记忆测试并过拟合。
- 将“代码质量”代理化后，代理可能通过删除注释、重写变量名（不改变语义）来提高分数。

AlphaEvolve 的修复办法：交付一个 LLM 从未见过的预留评估器，评估输入在评估时生成。即便如此，DeepMind 建议对任何拟部署方案进行严格审查。

### 为什么 LLM + 搜索胜过单独使用任一者

LLM 能生成可编译、语义上看得通的修改。对一个 2000 行 Python 文件做随机变异的遗传算法几乎总是产生语法错误。LLM 还会把搜索集中在合理的邻域（改一个函数，而不是随机改字节），这大幅减少了无用评估调用。

评估器则能捕捉 LLM 的虚构错误。LLM 可能会自信地声称某函数“在极限下是 O(n log n)”而实际上是 O(n^2)；一个壁钟基准会把这个问题解决清楚。

### AlphaEvolve 在前沿技术栈中的位置

| System | Generator | Evaluator | Domain | Example win |
|---|---|---|---|---|
| AlphaEvolve | Gemini | correctness + benchmark | algorithms, kernels, schedulers | 48-mul 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | correctness | combinatorial math | cap-set lower bounds |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM critique + experiment | ML research | ICLR workshop paper |
| Darwin Godel Machine (L4) | agent scaffolding | SWE-bench / Polyglot | agent code | 20% → 50% SWE-bench |

这四种系统都是同一配方的变体：生成器 + 评估器，形成循环。不同之处在于评估器打分的对象以及它有多严格。

## 使用方法

`code/main.py` 实现了一个最小的 AlphaEvolve 风格循环，针对一个玩具符号回归问题。这里的 “LLM” 是一个 stdlib 代理，提出对计算目标函数的程序的小型语法变异。 “评估器” 在保留的测试点上测量均方误差（MSE）。

观察要点：

- 最佳得分随世代的改进情况。
- MAP-elites 网格如何保持多样解，使循环不会收敛到局部最优。
- 去掉预留测试（仅用训练集评估，使用 `--no-holdout`）会如何导致严重过拟合。

## 投产条件

`outputs/skill-evaluator-rigor-audit.md` 是在一个新领域考虑采用 AlphaEvolve 风格循环的前置条件：你的评估器是否确实能发现你关心的失败模式？

## 练习

1. 运行 `code/main.py`。记录最佳得分轨迹。禁用预留评估器（标志 `--no-holdout`）并重跑。量化过拟合现象。

2. 阅读 AlphaEvolve 论文第 3 节关于 MAP-elites 网格的内容。为一个新问题（例如编译器优化通道）设计一个特征向量描述符，使搜索保持多样性。

3. 那个把 4x4 乘法从 49 次乘法改进到 48 次乘法的结果在 56 年后才出现。阅读论文附录 F，用三句话解释为什么这个问题的评估器特别容易做到严谨，以及为什么大多数领域不像它那样容易。

4. 提出一个 AlphaEvolve 会失败的领域。精确指出评估器在哪儿失效以及原因。

5. 对一个你熟悉的领域，写出你会使用的评估器签名。包括：(a) 正确性条件，(b) 性能度量，(c) 预留输入生成规则，(d) 至少一项反奖励劫持检查。

## 关键词

| Term | What people say | What it actually means |
|---|---|---|
| AlphaEvolve | "DeepMind's evolutionary coding agent" | Gemini + 程序数据库 + 可机器检验的评估器 |
| MAP-elites | "Diversity-preserving archive" | 由特征向量键控的格子；每个格子保存具有该描述符的最佳变体 |
| Island model | "Parallel evolution subpopulations" | 独立子种群定期迁移；防止过早收敛 |
| Machine-checkable evaluator | "Deterministic oracle" | 单元测试、模拟器或基准，LLM 无法伪造——这是该循环的前提条件 |
| Reward hacking / 奖励劫持 | "Optimizing the measure, not the goal" | 循环找到最大化分数但不实现目标的方式 |
| Seed program | "The starting point" | 初始的正确但不理想的程序，循环从此进化 |
| Held-out evaluator | "Evaluation data the LLM never saw" | 在评估时生成的输入，以防止记忆化 |

## 延伸阅读

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — 完整论文。  
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — 厂商的结果说明与总结。  
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — 发现的算法仓库，包括 48 次乘法的 4x4 矩阵乘法实现。  
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — 前驱系统。  
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 将评估器限定的自治作为关键研究方向进行讨论。