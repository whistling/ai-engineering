# A/B 测试 LLM 功能 — GrowthBook、Statsig 与「感觉问题」

> 传统的 A/B 测试并不是为非确定性的 LLM 设计的。关键区别：评估（evals）回答“模型能否完成任务？”，A/B 测试回答“用户是否在意？”。两者都必需；仅凭直觉上线已经不够。2026 年要测试的方向：提示词工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs 开源；准确性 vs 成本 vs 延迟）、生成参数（temperature、top-p）。真实案例：一个聊天机器人奖励模型变体带来了 +70% 的对话长度和 +30% 的留存；Nextdoor 的 AI 主题行实验在奖励函数优化后带来 +1% CTR；可汗学院的 Khanmigo 在延迟与数学准确性之间反复权衡。平台分裂：**Statsig**（2025 年 9 月被 OpenAI 以 11 亿美元收购）——顺序测试、CUPED、全能一体化。**GrowthBook**——开源、仓库为中心、贝叶斯 + 频率学派 + 顺序引擎、CUPED、SRM 检查、Benjamini-Hochberg 与 Bonferroni 校正。根据你对仓库-SQL 的偏好以及“被 OpenAI 收购”是否重要来选择。

**Type:** 学习  
**Languages:** Python（标准库，玩具级顺序测试模拟器）  
**Prerequisites:** 阶段 17 · 13（可观测性）、阶段 17 · 20（渐进部署）  
**Time:** ~60 分钟

## 学习目标

- 区分评估（“模型能否完成任务”）与 A/B 测试（“用户是否在意”）。
- 枚举三条可测试轴（提示词、模型、参数）并为每条选择指标。
- 解释 CUPED、顺序测试与 Benjamini-Hochberg 多重比较校正。
- 根据仓库-SQL 的姿态与企业对收购的态度选择 Statsig 或 GrowthBook。

## 问题背景

你手工调优了一个系统提示。感觉更好了。你把它上线了。转化率在噪声中波动。你责怪指标。或者你上线了一个新模型但转化率没变——是模型退化了还是变化太小无法检测？你不知道，因为你没有做 A/B 测试。

评估（evals）回答模型在标注集上是否能完成任务。它们不能回答用户是否偏好输出。只有受控的在线实验才能回答这个问题，而且只有当实验有足够的检验力、控制了非确定性并对多重比较做了校正时才可靠。

## 概念

### Evals vs A/B 测试

**Evals** — 离线、标注集、评判（评分规则、LLM 作为评判器或人工）。回答：“在这个固定分布上，输出是否正确 / 有帮助 / 安全？”

**A/B 测试** — 在线、真实用户、随机分配。回答：“新变体是否能显著提升关键的用户级指标？”

两者都必须做。Evals 在暴露前捕获回归；A/B 在产品层面确认影响。

### 要测试的内容

1. **提示词工程** — 措辞、系统提示结构、示例。指标：任务成功率、用户留存、每次请求成本。  
2. **模型选择** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：任务准确率 + 每次请求成本 + P99 延迟（多目标）。  
3. **生成参数** — temperature、top-p、max_tokens。指标：与任务相关（输出多样性 vs 决定性）。

### CUPED — 方差降低

Controlled-experiments Using Pre-Experiment Data。通过在比较后期结果前先回归出前期方差来降低方差。典型的方差降低：30%–70%。等效样本量“免费”上升。

实现：Statsig 与 GrowthBook 均有实现。

### 顺序测试

经典 A/B 假设固定样本量。顺序测试（“边看边决定”）在多次中期检验下控制假阳性率。始终有效的顺序方法（如 mSPRT、Howard 的置信序列）允许在明显胜出时提前终止。

### 多重比较校正

同时运行 20 个 A/B 测试在 95% 置信度下会产生一个偶然的假阳性。Bonferroni 校正会收紧每次检验的 α；Benjamini-Hochberg 控制假发现率（FDR）。GrowthBook 实现了两者。

### SRM — 样本比例不匹配（Sample Ratio Mismatch）

分配哈希将用户随机分配到变体。如果 50/50 的拆分实际是 47/53，则说明有问题——SRM 检查会标记。两个平台都实现了该检查。

### Statsig vs GrowthBook

**Statsig**:
- 于 2025 年 9 月被 OpenAI 以 11 亿美元收购。托管型 SaaS。
- 支持顺序测试、CUPED、保留人群（held-out populations）。
- 一体化：功能开关 + 实验 + 可观测性。
- 最适合：想要捆绑式产品且不在意 OpenAI 收购身份的团队。

**GrowthBook**:
- 开源（MIT）；仓库为中心（直接从 Snowflake/BigQuery/Redshift 读取）。
- 多种引擎：贝叶斯、频率学派、顺序测试。
- 支持 CUPED、SRM、Bonferroni、BH 校正。
- 可自托管或使用托管云。
- 最适合：以仓库-SQL 为主、数据团队控制指标层、希望使用 OSS 的组织。

### 非确定性使得检验力复杂化

相同提示会产生不同输出。传统的功效（power）计算假设观测独立同分布（IID）。在 LLM 的非确定性下，有效样本量比名义值更低。安全起见，将所需样本量放大约 1.3–1.5 倍。

### 真实案例结果

- 聊天机器人奖励模型变体：+70% 对话长度，+30% 留存。  
- Nextdoor 主题行：在奖励函数改进后 +1% CTR。  
- 可汗学院 Khanmigo：在延迟与数学准确性之间反复迭代。

### 反模式：凭感觉上线

每位资深工程师都能举出一个因为“感觉更好”而上线的功能。大多数人在几个月内没有注意到产品指标的倒退。A/B 测试是强制机制。

### 你应该记住的数字

- Statsig 被 OpenAI 收购：11 亿美元，2025 年 9 月。  
- GrowthBook：开源 MIT；贝叶斯 + 频率学派 + 顺序测试。  
- CUPED 方差降低：30%–70%。  
- LLM 非确定性 → 需要 +30%–50% 的样本量缓冲。

## 使用方法

`code/main.py` 模拟了一个顺序 A/B 测试，包含固定边界与顺序边界。展示了顺序方法如何让你提前停止。

## 交付成果

本课产出 `outputs/skill-ab-plan.md`。给定功能变更、工作负载、基线，选择平台、门控、样本量。

## 练习

1. 运行 `code/main.py`。在基线转化率为 3%、预期提升 5% 的情况下，达到 80% 功效需要多少样本量？  
2. 为一个受医疗监管、需本地部署的客户选择 Statsig 还是 GrowthBook？  
3. 设计一个 A/B 实验，测试 GPT-4 vs GPT-3.5 在“每次解决工单成本（cost-per-resolved-ticket）”上的差异。主要指标、护栏指标、次要指标分别是什么？  
4. 你的金丝雀（canary）通过但 A/B 显示 -1.2% 转化率。你会上线吗？写出升级（escalation）标准。  
5. 对一个前期方差为后期 60% 的情形应用 CUPED，计算等效样本量提升。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着 |
|------|-----------|--------------|
| Eval | “离线测试” | 模型能力的标注集评估 |
| A/B 测试 | “实验” | 针对真实用户的在线随机比较 |
| CUPED | “方差降低” | 使用前期数据回归以降低方差 |
| 顺序测试 | “允许边看边停” | 始终有效的程序，允许提前停止 |
| 多重比较 | “家族错误” | 同时运行多个测试会放大假阳性 |
| Bonferroni | “严格校正” | 将 α 除以测试数量 |
| Benjamini-Hochberg | “BH FDR” | 控制假发现率，比 Bonferroni 保守性低 |
| SRM | “错误拆分” | 样本比例不匹配；分配出错的信号 |
| Statsig | “OpenAI 持有” | 商业一体化产品，2025 年被收购 |
| GrowthBook | “那个 OSS” | MIT 许可、仓库为中心的平台 |
| mSPRT | “顺序概率比检验” | 经典的顺序检验方法 |

## 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/) — GrowthBook：如何对 AI 做 A/B 测试（实用指南）  
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation) — Statsig：超越提示词：基于数据的 LLM 优化与在线实验  
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools) — Statsig vs GrowthBook 的比较视角  
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf) — Deng 等人：CUPED 原始论文  
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240) — Howard：置信序列（顺序统计的理论基础）