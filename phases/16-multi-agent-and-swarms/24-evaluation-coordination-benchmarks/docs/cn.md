# Evaluation and Coordination Benchmarks

> Five 2025-2026 benchmarks cover the multi-agent evaluation space. **MultiAgentBench / MARBLE** (ACL 2025, arXiv:2503.01935) evaluates star/chain/tree/graph topologies with milestone KPIs; **graph is best for research**, cognitive planning adds ~3% milestone achievement. **COMMA** evaluates multimodal asymmetric-information coordination; state-of-the-art models including GPT-4o struggle to beat a random baseline. **MedAgentBoard** (arXiv:2505.12371) covers four medical task categories and often finds multi-agent does not dominate single-LLM. **AgentArch** (arXiv:2509.10769) benchmarks enterprise agent architectures combining tool-use + memory + orchestration. **SWE-bench Pro** ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)) has 1865 problems across 41 repos spanning business apps, B2B services, and developer tools; frontier models score ~23% on Pro vs 70%+ on Verified — a reality check on contamination. Claude Opus 4.7 (April 2026) is reported at **64.3%** on Pro with explicit agent-teams coordination (no Anthropic primary source published yet — treat as preliminary); Verdent (agent scaffold) hits **76.1% pass@1** on Verified ([Verdent technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)). **AAAI 2026 Bridge Program WMAC** (https://multiagents.org/2026/) is the 2026 community focal point. This lesson builds on MARBLE's metrics, runs a topology-vs-metric sweep, and pins the "just passing SWE-bench Verified is not evidence of generalization" rule.

**Type:** 学习  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 16 · 15 (Voting and Debate Topology), Phase 16 · 23 (Failure Modes)  
**Time:** ~75 分钟

## Problem

When a paper claims "our multi-agent system is better," the question is: better than what, on what, measured how? The 2023-2024 era of multi-agent evaluation was chaos — everyone picked their own metrics, their own baselines, and their own task sets. The 2025-2026 benchmarks imposed structure.

Without shared benchmarks, you cannot compare two multi-agent systems meaningfully. Worse, without hold-out benchmarks, frontier models can contaminate. SWE-bench Verified became partially contaminated in training corpora by mid-2025; frontier scores inflated; Pro was designed as an uncontaminated reality check.

This lesson enumerates the five canonical 2026 benchmarks, names what each measures, and teaches you to read benchmark claims skeptically.

## Concept

### MultiAgentBench (MARBLE) — ACL 2025

arXiv:2503.01935。评估四种协调拓扑（star、chain、tree、graph）在研究、编码与规划任务上的表现。基于里程碑的 KPI 跟踪部分进展，而不仅仅是最终成功。

测得结论：

- **Graph** 拓扑在研究场景中表现最佳；支持任意到任意（any-to-any）批评/审阅。
- **Chain** 最适合逐步细化的编码任务。
- **Star** 最适合快速事实整合。
- 在 graph 拓扑上，超过约 4 个智能体之后会出现 **协调税**（coordination tax）。
- **认知规划**（cognitive planning）在各拓扑上平均增加约 3% 的里程碑达成率。

何时使用：当你想对不同协调拓扑做苹果对苹果的比较时。MARBLE 仓库（https://github.com/ulab-uiuc/MARBLE）提供评估器。

### COMMA — 多模态不对称信息

涵盖智能体拥有不同观测模态且必须在不完全信息共享下进行协调的任务。报告的结果令人不安：包括 GPT-4o 在内的前沿模型在 COMMA 的智能体-智能体协作任务上都难以打败一个**随机基线**。信号是多模态智能体协调在训练与评估上被严重忽视——大模型在单一模态合作上表现尚可，但多模态协调会崩溃。

何时使用：当你的系统涉及多模态或不对称信息协调时。COMMA 的零结果提醒你在声称性能之前必须先测量。

### MedAgentBoard — 领域压力测试

arXiv:2505.12371。四类医疗任务：诊断、治疗规划、报告生成、病患沟通。比较多智能体、单一大模型与传统规则系统的表现。

结论：在大多数类别上，多智能体并不压倒性优于单一大模型。多智能体优势较窄——当子任务可清晰分解（例如诊断 + 治疗）时分解有利；当协调开销超过专业化收益（例如报告生成）时反而不利。

何时使用：当你的领域有明确的单一大模型基线时。如果 MedAgentBoard 的结论可推广，许多被提议的多智能体系统其实是过度工程化的。

### AgentArch — 企业架构基准

arXiv:2509.10769。覆盖企业场景，工具使用、记忆与编排层叠在一起。基准可以孤立评估每一层的贡献：加入工具有多大帮助？加入记忆？加入多智能体编排？

何时使用：当你在设计企业 Agent 栈并需要为每一层的引入做价值论证时。AgentArch 帮助避免为无法衡量价值的特性付费。

### SWE-bench Pro — 现实检验

arXiv:2509.16941。1865 道题目，来自 41 个仓库，覆盖商务应用、B2B 服务与开发者工具。设计目标是与后续训练截点相互独立，尽量保持**未污染**。前沿模型在 Pro 上得分约为 23%，而在 Verified 上为 70%+。这一差距即为污染信号。

2026 年 4 月分数：
- Claude Opus 4.7 在 Pro 上：**64.3%**（报告使用了显式 agent-teams 协调；Anthropic 尚未发布一手来源——视为初步数据）。
- Verdent（代理脚手架）在 Verified 上：**76.1% pass@1**（[技术报告](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。
- 未使用 agent 脚手架的前沿原始模型在 Pro 上的得分：约 **23–35%**（[SWE-bench Pro 论文](https://arxiv.org/abs/2509.16941)）。

要点："我们击败了 SWE-bench Verified" 已不足以作为能力证明。Pro 是当前的门槛测试。Agent-team 脚手架在 Pro 上带来了可测量的提升（大约 30–40 个百分点差异），这是 2026 年为数不多的支持多智能体协调的强实证论据之一。

### AAAI 2026 WMAC

AAAI 2026 Bridge Program —— Workshop on Multi-Agent Coordination（https://multiagents.org/2026/）。这是 2026 年多智能体 AI 研究的社区焦点。被 WMAC 接收的论文和研讨会论文集是评估新方法的规范会场；在生产决策时应优先考虑 WMAC 接收的结论，而不是仅仅依赖 arXiv 预印本。

### 以怀疑的眼光阅读基准声明 —— 2026 年检查清单

当有人宣称多智能体结果时：

1. **哪个基准，哪个拆分（split）？** SWE-bench Verified 与 Pro 差别巨大。报告在错误拆分上的数字毫无价值。  
2. **污染检查。** 基准是否在模型训练截止后发布？若不是，则需谨慎对待。  
3. **基线比较。** 与单一大模型基线比较，或与随机、或与先前多智能体工作比较。不要只是“与未经调优的同一系统版本比较”。  
4. **统计显著性。** 试验次数 N、p 值、置信区间。前沿模型方差高；单次运行会误导。  
5. **任务多样性。** 是单一任务还是多任务？泛化对生产很重要。  
6. **成本披露。** 每任务的 token、时延。一个 90% 的解法但成本是 20× 时，应被视为商业决策而非能力声明。

### 基准未能良好衡量的方面

- **长时程协调。** 持续数天的交互。目前所有基准都偏短。  
- **对抗鲁棒性。** 当某个智能体恶意或被攻陷时会怎样？  
- **部署下的漂移。** 基准是静态的；生产分布会变化。  
- **成本归一化的性能。** 大多数基准报告原始准确率，而非按成本归一化的准确率（accuracy-per-dollar）。

为你真正关心的维度构建内部基准往往是更正确的做法。

## Build It

`code/main.py` 是一个非交互式演示：

- 在一个玩具任务上模拟 3 个多智能体系统。
- 为每个系统计算 MARBLE 风格的里程碑指标。
- 通过从“训练”集保留任务来运行污染检查。
- 明确与随机基线比较。
- 打印基准声明记分卡。

运行：

```bash
python3 code/main.py
```

预计输出：系统记分卡，包含原始准确率、里程碑达成、每任务成本、与随机基线的差异，以及污染检查备注。

## Use It

`outputs/skill-benchmark-reader.md` 会读取任何多智能体基准声明并应用审查清单。输出：一个等级和若干注意事项。

## Ship It

生产评估纪律：

- **构建一个反映实际生产分布的内部基准。** 公共基准有参考价值，但不能替代。  
- **在每次比较中包括随机基线。** 如果在协调任务上无法大幅超越随机，任务可能本身就是不合适的。  
- **与准确率一并报告成本。** 包括 token 成本与墙钟时间。运维团队需要两者。  
- **每季度重建基准。** 生产分布会变化；过时的基准会误导。  
- **避免针对已发布基准的过拟合。** 如果团队专门优化 SWE-bench Pro 的指标，生产环境上会退化。

## Exercises

1. Run `code/main.py`。识别三个模拟系统中哪个在每里程碑成本（cost-per-milestone）上最好。它是否与原始准确率最高的系统一致？  
2. 阅读 MultiAgentBench (arXiv:2503.01935)。对于你自己的任务领域，决定 MARBLE 会推荐哪四种拓扑中的哪一种。根据论文结果进行论证。  
3. 阅读 SWE-bench Pro 论文。它具体采用了哪些手段使其对污染具有抗性？这些技术能否应用到你关心的其他基准上？  
4. 阅读 COMMA 关于多模态协调的发现。设计一个你可以添加到内部基准中的简单多模态协调任务。什么样的指标会是有用的信号？  
5. 将基准声明检查清单应用到一篇最近的多智能体论文的头条结果。你会给该声明打几分？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MARBLE | "MultiAgentBench" | ACL 2025；star/chain/tree/graph 拓扑，带里程碑 KPI。 |
| COMMA | "Multimodal benchmark" | 多模态不对称信息协调；前沿模型在该任务上难以超越随机。 |
| MedAgentBoard | "Domain stress test" | 四类医疗任务；常发现多智能体并不压倒单一大模型。 |
| AgentArch | "Enterprise benchmark" | 工具 + 记忆 + 编排 的分层评估。 |
| SWE-bench Pro | "Contamination-resistant" | 1865 道题、41 个仓库；Pro ~23% vs Verified 70%+（污染信号）。 |
| Milestone achievement | "Partial credit" | 对进展给予部分分数，而非仅衡量最终成功。 |
| Contamination | "Benchmark leaked into training" | 基准发布后被纳入训练语料；分数因而被夸大。 |
| WMAC | "AAAI 2026 Bridge Program" | 多智能体协调研讨会；社区焦点。 |

## Further Reading

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — 拓扑基准，带里程碑 KPI  
- [MARBLE repository](https://github.com/ulab-uiuc/MARBLE) — 参考实现  
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — 领域压力测试；多智能体常不占优  
- [AgentArch](https://arxiv.org/abs/2509.10769) — 企业 Agent 架构  
- [SWE-bench leaderboards](https://www.swebench.com/) — Verified 与 Pro 的前沿模型排行榜  
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — 2026 年社区焦点