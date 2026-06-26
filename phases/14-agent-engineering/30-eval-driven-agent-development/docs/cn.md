# 基于评估的智能体开发

> Anthropic 的指导：“从简单的提示词开始，使用全面的评估进行优化，只有在需要时才添加多步的智能体系统。”评估不是最后一步。它是驱动第 14 阶段中每一个其他选择的外层循环。

**Type:** 学习 + 构建  
**Languages:** Python（stdlib）  
**Prerequisites:** Phase 14 的全部内容。  
**Time:** ~60 分钟

## 学习目标

- 说出三层评估：静态基准、自定义离线、在线生产，以及每一层的用途。
- 解释评估器-优化器（evaluator-optimizer）紧闭环的工作原理。
- 描述 2026 年最佳实践：评估与代码并存、在 CI 中运行、作为 PR 的门禁。
- 将第 14 阶段的每一课与它产生的评估用例建立关联。

## 问题

智能体能通过演示，但在生产中以演示无法预测的方式失败。基准回答的是“这个模型是否具有广泛能力？”，而不是“这个智能体是否为我的产品生成了正确的补丁？”答案是：在三层进行持续评估，并将每一条护栏和学习到的规则映射到一个评估用例。

## 概念

### 三层评估

1. **静态基准（Static benchmarks）** — 例如用于代码的 SWE-bench Verified（Lesson 19）、用于浏览 / 桌面的 WebArena/OSWorld（Lesson 20）、通用能力的 GAIA（Lesson 19）、用于工具使用的 BFCL V4（Lesson 06）。用于跨模型比较和回归门控。污染（contamination）是真实存在的：SWE-bench+ 发现了 32.67% 的解答泄露。始终报告 Verified / ±审计的分数。

2. **自定义离线评估（Custom offline evals）** — 针对你的产品形态：
   - LLM-as-judge（Langfuse、Phoenix、Opik — Lesson 24）。
   - 基于执行的评估（运行补丁，检查测试）。
   - 基于轨迹的评估（将动作序列与黄金轨迹比较；OSWorld-Human 显示顶级智能体比黄金轨迹高出 1.4–2.7 倍）。

3. **在线评估（Online evals）** — 生产环境：
   - 会话回放（Langfuse）。
   - 由护栏触发的告警（Lesson 16、21）。
   - 每步成本 / 延迟跟踪（Lesson 23 的 OTel spans）。

### 评估器-优化器（Anthropic）

紧闭环流程：

1. Proposer 生成输出。  
2. Evaluator 判定。  
3. 反复改进直到 evaluator 通过。

这就是 Self-Refine（Lesson 05）的广义化。你关心的任何智能体流程都可以用评估器-优化器进行包裹以提高可靠性。

### 2026 年最佳实践

- 评估与代码并存（Evals live next to code）。
- 在每次 PR 的 CI 中运行。
- 基于评估分数来门控合并（例如：“相较 main 不得出现 >5% 的回归”）。
- 每一条护栏都映射到一个评估用例。
- 每一条学习到的规则（Reflexion、pro-workflow learn-rule）都映射到一个失败用例。

### 将 Phase 14 关联起来

第 14 阶段中的每一课都会生成评估用例：

| Lesson | 它生成的评估用例 |
|--------|------------------|
| 01 Agent Loop | 预算耗尽、无限循环护栏 |
| 02 ReWOO | 当工具失败时规划器正确重新规划 |
| 03 Reflexion | 学到的反思在重试时生效 |
| 05 Self-Refine/CRITIC | 判定器通过精炼后的输出 |
| 06 Tool Use | 参数强制有效；未知工具被拒绝 |
| 07-10 Memory | 检索引用与来源匹配；陈旧事实导致失效 |
| 12 Workflow Patterns | 每种工作流模式产生正确输出 |
| 13 LangGraph | 恢复（resume）能精确重建状态 |
| 14 AutoGen Actors | DLQ 捕获崩溃的处理程序 |
| 16 OpenAI Agents SDK | 护栏在正确的输入上触发 |
| 17 Claude Agent SDK | 子智能体的结果返回给编排器 |
| 19-20 Benchmarks | SWE-bench Verified 分数、WebArena 成功率、OSWorld 效率 |
| 21 Computer Use | 每一步的安全检查捕获注入的 DOM |
| 23 OTel | spans 发出必需的属性 |
| 26 Failure Modes | 检测器标记已知故障 |
| 27 Prompt Injection | PVE 拒绝被投毒的检索结果 |
| 28 Orchestration | 监督者将请求路由到合适的专家 |
| 29 Runtime Shapes | DLQ 处理 N% 的故障 |

如果你的评估套件包含这些用例，你就覆盖了第 14 阶段的要点。

### 基于评估开发失败的常见原因

- 无基线（No baseline）。缺少最后已知良好版本的评估无法解读。务必存储基线。
- 使用 LLM 作为裁判而没有落地依据（LLM-judge without grounding）。裁判也会产生幻觉。采用 CRITIC 模式（Lesson 05）——让裁判基于外部工具落地判断。
- 对评估过拟合（Over-fitting to evals）。为了评估而优化会偏离生产实用性。轮换用例。
- 评估不稳定（Flaky evals）。非确定性的用例会产生误报。锁定随机种子，快照状态。

## 构建它

`code/main.py` 是一个仅使用 stdlib 的评估框架：

- 用例注册表，含类别（benchmark、custom、online）。
- 被测的脚本化智能体。
- 评估器-优化器循环：propose、judge、refine，直到通过或达到最大轮次。
- CI 门控：汇总通过率 + 与基线的回归比较。

运行方法：

```
python3 code/main.py
```

输出：每个用例的通过/失败、回归标志、CI 门控判定。

## 使用它

- 在与智能体代码相同的仓库中编写评估用例。
- 通过 CI 在每次 PR 上运行它们。
- 在出现回归时使构建失败。
- 跟踪随时间的通过率。
- 将每一起生产故障映射为一个新的用例。

## 部署它

`outputs/skill-eval-suite.md` 为一个智能体产品构建了三层评估套件，包含 CI 门控和回归跟踪。

## 练习

1. 选取一个你的生产故障。编写一个能复现该故障的评估用例。现在你的智能体能通过它吗？  
2. 为你的领域构建一个 LLM-judge 评分规则（rubric），包含三个维度（事实性、语气、范围）。对 50 次会话进行打分。  
3. 将评估套件接入 CI。在 >=5% 的回归时使构建失败。  
4. 添加一个轨迹效率指标：智能体所用步骤数与黄金轨迹的比值。  
5. 将第 14 阶段的每一课映射到你套件中的评估用例。有没有遗漏？那就是需要补上的空白。

## 关键词

| 术语 | 常有人怎么说 | 实际含义 |
|------|--------------|---------|
| Static benchmark | “现成的评估” | SWE-bench、GAIA、AgentBench、WebArena、OSWorld |
| Custom offline eval | “领域评估” | LLM-as-judge / 执行型 / 轨迹型，针对你的产品形态 |
| Online eval | “生产评估” | 会话回放、护栏告警、成本/延迟跟踪 |
| Evaluator-optimizer | “提议-判定-精炼” | 迭代直到判定器通过 |
| CI gate | “合并拦截器” | 在评估回归时使构建失败 |
| Baseline | “最后已知良好版本” | 用于检测回归的参考分数 |
| Trajectory efficiency | “相对于黄金轨迹的步骤数” | 智能体步骤数除以人类专家的最小步骤数 |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — “从简单开始，用评估优化”  
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 精选基准  
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — 工具使用基准  
- [Langfuse docs](https://langfuse.com/) — 实践中的评估 + 会话回放