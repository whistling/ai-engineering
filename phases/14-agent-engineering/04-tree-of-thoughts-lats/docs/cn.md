# Tree of Thoughts and LATS: Deliberate Search

> 单一的思维链轨迹没有回溯的余地。ToT（Yao 等，2023）将推理转为一棵树，每个节点上都有自我评估。LATS（Zhou 等，2024）在蒙特卡洛树搜索下将 ToT、ReAct 与 Reflexion 统一。Game of 24 从 4%（CoT）提升到 74%（ToT）；LATS 在 HumanEval 上达到 92.7% pass@1。

**Type:** 构建  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 14 · 01（Agent Loop），Phase 14 · 03（Reflexion）  
**Time:** ~75 分钟

## 学习目标

- 将推理框作搜索来构建：节点是“想法”，边是“扩展”，值是“多有希望”。
- 实现一个基于标准库的 ToT 风格的广度优先树搜索，并带自我评估评分。
- 扩展为一个玩具版 LATS MCTS 循环，包含选择 / 扩展 / 模拟 / 反向传播。
- 决定何时值得为搜索付出代价（代币乘数）：例如 Game of 24、代码生成；以及何时一条轨迹就足够（简单问答）。

## 问题描述

思维链是线性的行走。如果第一步错了，后续每一步都基于错误前提。在 Game of 24（用四个数字和 + − × ÷ 得 24）上，GPT-4 的 CoT 只有 4% 的准确率。模型早期选择了错误的子表达式，无法恢复。

推理需要提出多个候选、评估它们、挑选有前景的，并在遇到死胡同时回溯。这就是搜索。Tree of Thoughts 和 LATS 是两种典型的表述方式。

## 概念

### Tree of Thoughts（Yao 等，NeurIPS 2023）

每个节点是一个连贯的中间步骤（“一个想法”）。每个节点可以扩展出 K 个子想法。LLM 对每个节点进行自我评估并给出评分。搜索遍历这棵树——可用 BFS、DFS 或 beam search。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

自我评估是承重的部分。论文展示了三种变体：`sure / likely / impossible` 的分类、`1..10` 的数值评分，以及候选间的投票。三者在 Game of 24 上都显著优于 CoT（使用 GPT-4 时从 4% 提升到 74%）。

### LATS（Zhou 等，ICML 2024）

LATS 在 MCTS 下统一了 ToT、ReAct 和 Reflexion。LLM 扮演三种角色：

- **Policy（策略）**：提出候选下一步动作（ReAct 风格）。
- **Value function（价值函数）**：对部分轨迹打分（ToT 风格的自评）。
- **Self-reflector（自我反思器）**：在失败时写出自然语言的反思（Reflexion 风格），并用它来重新种子后续的 rollout。

环境反馈（观测）混入价值函数中，使搜索能依据真实工具结果而非仅凭模型意见进行判断。论文当时的结果：使用 GPT-4 在 HumanEval 上 pass@1 达到 92.7%（SOTA），使用 GPT-3.5 在 WebShop 上平均 75.9（接近基于梯度的微调表现）。

### 最小化解释的 MCTS

每次迭代四个阶段：

1. **Select（选择）** — 使用 UCT（树的上置信界）从根走到一个叶子。
2. **Expand（扩展）** — 通过策略生成 K 个子节点。
3. **Simulate（模拟）** — 从子节点开始使用策略进行 rollout，用价值函数（或环境奖励）对叶子打分。
4. **Backpropagate（反向传播）** — 更新路径上的访问计数和价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是利用；第二项是探索。根据任务调整 `c`。

### 成本现实

搜索会导致代币激增。ToT 在 Game of 24 上使用的代币是 CoT 的 100–1000 倍。LATS 类似。这不是免费的；把搜索保留给：

- 单条轨迹明显不足以解决的任务（Game of 24、复杂代码）。
- 墙钟时间不重要但正确性重要的任务。
- 拥有廉价且可靠的价值函数的任务（代码的单元测试、有明确目标的数学题）。

如果任务只有一个正确答案但评估器噪声很大，搜索往往会让情况更糟——它会找到一个“高分”的错误答案。

### 2026 年的定位

大多数生产智能体不运行 LATS。它们运行 ReAct 并结合工具驱动的验证（CRITIC，Lesson 05）。搜索出现在专业化的利基场景中：

- 使用测试作为价值函数的代码智能体（HumanEval 风格）。
- 在多个查询路径上进行探索的深度研究智能体。
- LangGraph 子图内部的重规划密集型工作流。

AlphaEvolve（Lesson 11）是 2025 年的极端例子：对代码进行进化搜索、以程序化可检验的适应度作为衡量，带来边界改进（例如 56 年来首次改进的 4x4 矩阵乘加速）。

## 实现

`code/main.py` 实现了：

- 一个在风格化“选择算术运算”任务上的微型 ToT BFS。
- 在相同任务上的玩具 LATS MCTS 循环（选择 / 扩展 / 模拟 / 反向传播），采用 UCT 选择策略。
- 一个将符号评分与自我评估分数组合的价值函数。

运行：

```
python3 code/main.py
```

运行轨迹会显示 ToT 在每个节点扩展 3 个候选的 BFS，与 LATS 通过 MCTS 收敛到最佳 rollout 的过程相比。两者还会打印代币计数。

## 使用建议

LangGraph 将 ToT 风格的探索作为子图模式；LangChain 团队关于 LATS（2024 年 5 月）的博客是参考教程。LlamaIndex 提供一个 `TreeOfThoughts` agent。在多数 2026 年的生产环境中，这一模式被放在 `if task_complexity > threshold: use_search()` 的分支后面——见 Lesson 05 中的 evaluator-optimizer 模式。

## 部署

`outputs/skill-search-policy.md` 在任务形态、预算和评估器可信度之间选择线性 ReAct、ToT、LATS 和进化搜索。

## 练习

1. 在 UCT c=0.1 与 c=2.0 下运行玩具 LATS。轨迹有什么变化？  
2. 将价值函数替换为噪声更大的评分器（加入随机扰动）。MCTS 仍然能找到最优叶子吗？它能容忍的最小信噪比是多少？  
3. 实现 beam-search 风格的 ToT（在每一层保留 top-k），并与 BFS 比较。在紧张的代币预算下哪个更好？  
4. 阅读 LATS 第 5.1 节。复现 HumanEval 的轨迹计数：需要多少次 rollout 才能达到论文报道的 pass@1？  
5. 阅读 LATS 论文关于“何时 LATS 帮助较少”的讨论。写一段决策规则（一段话），将任务形态映射到搜索策略。

## 关键术语

| 术语 | 人们如何称呼 | 实际含义 |
|------|--------------|----------|
| Tree of Thoughts | “分支的思维链” | Yao 等——有自我评估的思维树节点 |
| LATS | “针对 LLM 的 MCTS” | Zhou 等——在 MCTS 下统一 ToT + ReAct + Reflexion |
| UCT | “上置信界” | 选择公式，平衡利用（Q）和探索（ln N / n） |
| Value function | “这个状态有多好” | 由提示词驱动的 LLM 评分或环境奖励；用于反向传播 |
| Policy | “动作提议者” | ReAct 风格的生成器；输出候选下一步想法/动作 |
| Rollout | “模拟轨迹” | 从某节点沿策略行走到叶子，并用价值函数评分 |
| Backpropagate | “更新祖先” | 将叶子的奖励向上传递，更新访问计数和 Q 值 |
| Search cost | “代币爆炸” | 在 Game of 24 上比 CoT 多 100–1000 倍；在采用前要评估预算 |

## 深入阅读

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — 经典论文  
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — 带 Reflexion 反馈的 MCTS  
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 搜索的子图模式  
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — 以程序化评估器为适应度的进化搜索