# 基准测试：SWE-bench、GAIA、AgentBench

> 三个基准在 2026 年成为评估智能体的锚点。SWE-bench 测试代码修补能力。GAIA 测试通用工具使用能力。AgentBench 测试多环境推理能力。了解它们的构成、污染（contamination）问题，以及它们未衡量的方面。

**Type:** 学习  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 14 · 06（工具使用）  
**Time:** ~60 分钟

## 学习目标

- 说出 SWE-bench 的测试框架名称（`FAIL_TO_PASS`），并解释为什么以单元测试为门控。
- 解释为什么存在 SWE-bench Verified（OpenAI，500 个任务）以及它移除了什么内容。
- 描述 GAIA 的设计理念：对人类简单、对 AI 困难；三档难度。
- 说出 AgentBench 的八个环境以及其对开源 LLM 的主要阻碍因素。
- 总结 SWE-bench+ 的污染发现及其含义。

## 问题

排行榜只能告诉你哪个模型在某个基准上得分更高。它们不能告诉你：

- 基准是否被污染（训练数据中存在解答、测试泄露）。
- 基准是否衡量了你关心的能力（代码、浏览、还是通用能力）。
- 评估器是否健壮（AST 匹配、状态检查、人工复核）。

在引用一个数字之前，先了解这三个锚定基准及它们的失效模式。

## 概念

### SWE-bench（Jimenez 等，ICLR 2024 口头报告）

- 来自 12 个流行 Python 仓库的 2,294 个真实 GitHub issue。
- 智能体获得：修复前提交的代码库 + 自然语言的 issue 描述。
- 智能体输出：一个补丁（patch）。
- 评估器：应用补丁并运行仓库的测试套件。补丁必须使 `FAIL_TO_PASS` 测试（原先失败、现在通过）翻转为通过，同时不破坏 `PASS_TO_PASS` 测试。

SWE-agent（Yang 等，2024）在发布时达到 12.5% 的通过率，强调智能体—计算机接口（模型能理解的文件编辑命令、搜索语法）。

### SWE-bench Verified

OpenAI，2024 年 8 月发布。由人工策划的 500 任务子集。移除了歧义性问题、不可靠的测试、以及修复不明确的任务。是衡量“你的智能体能否产出真实补丁”的主要基准。

### 污染（Contamination）

- 超过 94% 的 SWE-bench issue 比大多数模型的截止日期更早。
- **SWE-bench+** 发现成功补丁中有 32.67% 的案例在 issue 文本中泄露了解法（模型在描述中看到了修复），另有 31.08% 的案例因测试覆盖薄弱而可疑。
- Verified 更干净，但并非完全无污染。

实际影响：一个在 SWE-bench 得分 50% 的模型，在 SWE-bench+ 上可能只有 35%。如果声称 SWE-bench 性能，务必同时报告 Verified 或 SWE-bench+ 的结果。

### GAIA（Mialon 等，2023 年 11 月）

- 466 道问题；用于私有排行榜的保留题为 300 题（见 huggingface.co/gaia-benchmark）。
- 设计哲学：对人类概念上简单（92% 的人类正确），但对 AI 困难（带插件的 GPT-4：15%）。
- 测试推理、多模态、网页与工具使用。
- 三个难度等级；第三级需要跨模态的长链工具调用。

GAIA 用于衡量“通用能力”。不要与代码专用基准混淆。

### AgentBench（Liu 等，ICLR 2024）

- 覆盖 8 个环境，跨代码（Bash、DB、KG）、游戏（Alfworld、LTP）、网页（WebShop、Mind2Web）以及开放式生成。
- 多回合，每个分割约 4k–13k 次交互回合。
- 主要发现：长期推理、决策制定与指令追踪是开源 LLM 赶上商用模型的主要阻碍。

### 这些基准未衡量的内容

- 真实世界的运营成本（token 数量、真实时延）。
- 在对抗条件下的安全行为。
- 你领域内的性能（请使用你自己的评估，见 Lesson 30）。
- 边缘失败情况（基准通常取平均；生产场景更关心最差的 1%）。

### 评测常见误区

- 单一数字的迷信。SWE-bench 的 50% 比不上 P50/P75/P95 成本 + 步骤分布的重要性。
- 被污染的声明。在报告 SWE-bench 时不说明是否用了 Verified 或 SWE-bench+ 就具有误导性。
- 以基准作为开发目标。为基准优化会偏离生产可用性。

## 实践：构建它

`code/main.py` 实现了一个玩具级的 SWE-bench 风格测试框架：

- 合成的 bug 修复任务（3 个任务）。
- 一个脚本化的“智能体”来提出补丁。
- 一个测试运行器，检查 `FAIL_TO_PASS`（缺陷已修复）和 `PASS_TO_PASS`（无回归）。
- 一个基于问题分解深度的 GAIA 风格难度分类器。

运行它：

```
python3 code/main.py
```

输出将显示每个任务与每个难度的解决率，并使评估规则具体化。

## 使用建议

- 对代码智能体使用 **SWE-bench Verified**。务必报告 Verified 得分。
- 对通用智能体使用 **GAIA**。使用私有排行榜的切分数据。
- 对多环境比较使用 **AgentBench**。
- 对你的产品形态使用 **自定义评估**（见 Lesson 30）。

## 部署（Ship It）

`outputs/skill-benchmark-harness.md` 构建了一个针对任意代码库-任务对、以 `FAIL_TO_PASS` / `PASS_TO_PASS` 为门控的 SWE-bench 风格评估框架。

## 练习

1. 将玩具评估框架移植到真实仓库（选一个你自己的）。为已知 bug 写 3 个 `FAIL_TO_PASS` 测试。
2. 增加步数统计指标。在你的 3 个任务上，每次成功平均需要多少智能体步骤？
3. 阅读 SWE-bench+ 论文。实现一个解法泄露检测（将 issue 文本与 diff 做模式匹配）。
4. 从 GAIA 的公开切分下载一道题目。追踪一个 GPT-4 级别的智能体会怎么做。它需要哪些工具？
5. 阅读 AgentBench 的按环境细分结果。哪个环境最像你的产品表面？那里的“SOTA” 看起来如何？

## 关键术语

| 术语 | 大家如何称呼 | 实际含义 |
|------|--------------|----------|
| SWE-bench | “代码智能体基准” | 2,294 个 GitHub issue；补丁必须翻转 `FAIL_TO_PASS` 测试 |
| SWE-bench Verified | “干净的 SWE-bench” | 由 OpenAI 人工策划的 500 个任务 |
| `FAIL_TO_PASS` | “修复门” | 修复后必须通过的、之前失败的测试 |
| `PASS_TO_PASS` | “无回归门” | 原本通过的测试，修补后仍必须通过 |
| GAIA | “通用基准” | 466 道人类容易 / AI 困难的多工具问题 |
| AgentBench | “多环境基准” | 8 个环境；长时程多回合评估 |
| 污染（Contamination） | “训练集泄露” | 基准任务出现在模型训练数据中 |
| SWE-bench+ | “污染审计” | 在成功的 SWE-bench 补丁中发现 32.67% 的解法泄露 |

## 延伸阅读

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — 原始基准
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 人工策划子集
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — 通用基准
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — 多环境套件