# Theory of Mind and Emergent Coordination

> Li et al. (arXiv:2310.10701) 显示，在一个协作文字游戏中，LLM 代理会出现 **高阶心智理论（ToM）** 的涌现 —— 推理另一个代理对第三方信念的看法 —— 但由于上下文管理和幻觉问题，在长时程规划上失败。Riedl (arXiv:2510.05174) 在群体尺度上测量了高阶协同并发现，**只有** 带有 ToM 提示词的条件会产生与身份关联的分化和目标导向的互补性；低容量的 LLM 仅表现出伪性涌现。也就是说，协调的涌现依赖于提示词并依赖模型，而非自发出现。本课实现了一个最小的 ToM 感知代理，运行带和不带 ToM 提示词的协作任务，并按照 Riedl 2025 的协议衡量协调差异。

**Type:** 学习 + 构建  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 16 · 07（心智社会与辩论），Phase 16 · 17（生成式代理）  
**Time:** ~75 分钟

## 问题

多智能体协调看起来常常很神奇：代理分工、互相预测、避免冗余。通常这种“涌现”是提示词工程的产物——有人告诉代理“要协作”。移除提示词，协调就消失。

Riedl 2025 的发现更严格：在受控条件下，只有当代理被提示去推理“其他代理的心智”（ToM）时，协调才会涌现。没有 ToM 提示词，即便是强模型也会表现出在统计控制下无法存活的协调模式。这对生产非常重要：很多团队发布“多代理协调”功能，其实是高度依赖提示词且脆弱的。

本课将 ToM 视为一种特定能力（即关于信念的信念的推理），构建最小的 ToM 感知代理，衡量真实协调与提示词包装的区别。

## 概念

### ToM 的含义

发展心理学：3 岁儿童认为任何人的内心世界都与自己相同。5 岁儿童理解他人可能有不同的信念。7 岁儿童能推理关于信念的信念（“她认为我认为球在杯子下”）。这些分别对应零阶、一次和二阶 ToM。

对于 LLM 代理，ToM 的阶数对应为：

- **Zeroth-order（零阶）:** 没有对他人的模型。代理只根据自身观察行动。
- **First-order（一次）:** 代理有对每个其他代理信念的模型。“Alice 相信 X。”
- **Second-order（二阶）:** 代理建模递归信念。“Alice 相信 Bob 相信 X。”

Li et al. 2023 发现，在协作游戏中，一阶和二阶 ToM 会在 LLM 代理中涌现，但在长时程和通信不可靠时会退化。

### 简述 Sally-Anne 测试

1985 年的错误信念测试：Sally 把弹珠放在 A 篮子里并离开。Anne 把它移到 B。Sally 回来时会去哪儿找弹珠？具有一阶 ToM 的孩子会说 A（Sally 的信念与现实不同）。没有一阶 ToM 的孩子会说 B。

GPT-4 时代的 LLM 在简单呈现的 Sally-Anne 风格测试中能通过。但当叙事很长、场景多次变化或问题间接表述时，会失败。这就是截至 2026 年生产级 LLM 在 ToM 方面的实际状况。

### Riedl 的协调度量

Riedl (arXiv:2510.05174) 构建了群体规模的测试：N 个代理、一个协作目标、可变的提示词条件。测量指标：

1. **Identity-linked differentiation（与身份关联的分化）**：代理随时间是否发展出稳定的角色区分？
2. **Goal-directed complementarity（目标导向的互补性）**：代理的行动是否相互补充（不同子任务）而非重复？
3. **Higher-order synergy（高阶协同）**：统计度量，判断群体是否实现了任何子集无法实现的结果。

结果：只有在 ToM 提示条件下，这三个指标才会高于基线。没有 ToM 提示，指标在中等容量模型中徘徊于随机水平。大型模型在没有显式 ToM 提示时也能表现出一定的协调，但效果比显式提示时小。

### 协调幻象

在没有统计控制时，演示中的“涌现协调”往往反映的是：

- 把协调烘托进提示词（系统提示写明“协作”）。
- 观察者偏见（我们看到期望看到的模式）。
- 事后挑选成功的运行结果。

那些在没有可测信号下宣传“涌现协调”的生产系统应被视为营销行为。声称前，请先衡量。

### 一个最小的 ToM 感知代理

结构：

```
agent state:
  own_beliefs:    {代理自身相信的事实}
  other_models:   {other_agent_id -> {代理归因于他们的信念}}
  actions_last_N: [其他代理动作的历史]
```

在上面结构中，`other_models` 属性就是 ToM 状态。一阶 ToM 仅保留一层。二阶则在 `other_models[i][other_models_of_j]` 中添加——我认为代理 i 认为代理 j 的信念。

更新观测：
  - 从直接观测更新 own_beliefs
  - 根据他人的动作 + 先验信念更新 other_models[agent_id]

动作选择：
  - 枚举候选动作
  - 对于每个动作，基于建模的信念预测每个其他代理下一步会做什么
  - 选取在这些预测下最大化联合结果的动作

### 为什么长时程会受损

Li et al. 指出：上下文限制导致代理忘记哪个信念属于谁。幻觉则会向他人模型中添加虚假信念。两者都会产生“我以为他以为 X” 的错误，并随着时间复合放大。

论文及 2024–2026 年的后续工作列举了缓解方法：

- **在提示词中显式 ToM 状态。** 结构化格式如 `{agent_id: belief_list}`。强制检索以保持身份—信念绑定。
- **缩短推理链条。** 每回合更少的 ToM 更新以减少复合幻觉。
- **外部 ToM 存储。** 将模型保存在 LLM 上下文之外；每回合注入仅相关部分。

### ToM 在生产中的失败点

- **对抗环境。** 具有良好 ToM 的代理更容易被操纵（你可以预测他们对你的模型，然后利用之）。
- **异构团队。** 当模型不同，适用于某一对手的 ToM 模型不一定泛化。
- **依赖真实事实的任务。** ToM 关注的是信念；若正确性依赖于事实，ToM 可能成为干扰。

### 可测量的协调信号

三个实际信号可以区分真实的团队协调与提示词包装：

1. **随时间的互补性。** 在多回合任务中，代理的动作是否覆盖互不重叠的子任务？
2. **预期性。** 代理 A 在 T+1 的动作是否依赖于对 B 在 T+2 动作的预测，且该预测被证实？
3. **纠正。** 当 A 在回合 T 误解 B 的信念时，是否在 T+2 前后进行纠正？

这些都可以在日志化的多代理系统中测量。它们是“协调”叙述的实质版。

## 构建它

`code/main.py` 实现：

- `ToMAgent` — 跟踪自身信念和针对每个其他代理的信念模型。
- 一个协作任务：三个代理必须从三个箱子里分别收集三个代币；每个箱子最多容纳一个代币。代理不能通信；他们从彼此的动作推断意图。
- 两种配置：`zeroth_order`（无 ToM）和 `first_order`（带一层 ToM 的模型）。
- 在 200 次随机试验上测量：完成率、重复率（两个代理瞄准同一箱子的情况）、完成所需平均回合数。

运行：

```
python3 code/main.py
```

预期输出：零阶代理在 10 回合内重复劳动率约为 ~35%，完成率约为 ~60%。一阶 ToM 代理重复率约为 ~5%，完成率约为 ~95%。该差值即为可测的协调效应。

## 使用它

`outputs/skill-tom-auditor.md` 是一个技能脚本，用于审计多代理系统关于“涌现协调”的声明。检查提示词包装、与对照组的统计显著性，以及测得的互补性。

## 交付标准

协调宣称核对清单：

- **Control condition（对照条件）。** 系统中不包含协调提示词的版本。对两个版本都进行衡量。
- **Statistical test（统计检验）。** 在你的指标上，系统与对照的差异在 `p < 0.05` 上是否显著？
- **Complementarity measure（互补性度量）。** 基于随时间的动作不重叠性，而非仅仅看最终成功率。
- **Failure-case log（失败案例日志）。** 当代理失协调时，ToM 状态是什么样？
- **Model-capacity disclosure（模型容量披露）。** 若效果在较小模型上消失，要说明。

## 练习

1. 运行 `code/main.py`。确认一阶 ToM 将重复率降低约 7 倍。当你扩展到 5 个代理和 5 个箱子时，这一差距是否持续？
2. 实现二阶 ToM（代理 A 建模 B 对 C 的看法）。它相对于一阶有改进吗？在哪些任务上有效？
3. 在 ToM 状态中注入一个 **幻觉**：每回合随机翻转一个信念。这会在多大程度上降低一阶性能？
4. 阅读 Li et al. (arXiv:2310.10701)。复现“长时程退化”发现：当回合数从 10 增加到 30 时，你的一阶 ToM 性能如何变化？
5. 阅读 Riedl 2025 (arXiv:2510.05174)。在你的仿真日志上实现高阶协同统计量。没有 ToM 提示条件时，这种效应是否存在？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Theory of Mind | "Understanding others' minds" | 模拟另一代理信念的能力。按阶数分级（0、1、2+）。 |
| Sally-Anne test | "The false-belief test" | 1985 年的发展心理学测试；LLM 能通过简单版本，但在复杂版本中失败。 |
| First-order ToM | "A believes X" | 建模某一他人的关于事实的信念。 |
| Second-order ToM | "A believes B believes X" | 递归建模，多一层深度。 |
| Identity-linked differentiation | "Stable roles over time" | Riedl 的度量：角色随时间保持，不是随机的。 |
| Goal-directed complementarity | "Disjoint actions" | 代理瞄准不同子任务，而非相同目标。 |
| Higher-order synergy | "Group exceeds any subset" | Riedl 的统计度量，用以判断真实协调。 |
| Coordination illusion | "It looks coordinated" | 在没有可测信号的情况下，提示词包装的协调表象。 |

## 延伸阅读

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — 协作游戏中 ToM 的涌现；长时程失败模式  
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — 群体尺度测量；ToM 提示是承重条件  
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — 1978 年 ToM 概念的起源  
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — Sally-Anne 论文（1985）