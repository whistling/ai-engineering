# 自动化对齐研究（Anthropic AAR）

> Anthropic 在独立沙箱中并行运行 Claude Opus 4.6 的自主对齐研究者（Automated Alignment Researchers），各队在独立沙箱内操作，并通过一个共享论坛协调，该论坛的日志存储在所有沙箱之外（因此代理无法删除自己的记录）。在弱到强训练问题上，AARs 的表现优于人类研究者。Anthropic 的总结指出，规定性的工作流程常常限制 AAR 的灵活性并降低性能。自动化对齐研究是压缩步骤，它将时间线压缩到 RSP 旨在检测的精确失配风险。

**Type:** 学习  
**Languages:** Python（stdlib，parallel-research-forum 模拟器）  
**Prerequisites:** Phase 15 · 05（AI Scientist v2），Phase 15 · 04（DGM）  
**Time:** ~60 分钟

## 问题

对齐研究在人工研究者时间上代价高昂。像可扩展监督、奖励规范或弱到强训练这样的课题需要每次迭代数周的实验。随着前沿能力的进展，对齐工作量增长速度超过合格研究者的供给。

自动化对齐研究（AAR）在问：能否让同样那些能力超过对齐进度的前沿模型，参与缩小这一区别。Anthropic 2026 年关于 AAR 运行的弱到强训练研究报告（https://alignment.anthropic.com/2026/automated-w2s-researcher/）是此类部署系统的首批公开结果之一。

结果是真正正面的：AARs 在某个研究问题上比人类基线做得更好。该结果也将本阶段发展出的每一个关切集中于一个系统中。如果对齐研究能被自动化，那么同样也可能自动化那些破坏护栏的部分。RSP 为 AI 研发能力设定的阈值就是以此循环为背景写就的。

## 概念

### 架构（公开描述）

- **并行代理。** 多个 Claude Opus 4.6 实例作为 AARs 运行，每个在独立沙箱中。沙箱限制文件系统、网络和进程面。
- **共享论坛。** 发现结果发布到一个共享论坛，其存储后端位于每个沙箱之外。代理可以读取论坛；代理不能在各自沙箱内删除或编辑论坛记录。
- **任务分配。** 人类设计的任务队列向 AARs 分配研究问题。AARs 提出分解方案、运行实验、撰写发现报告。
- **弱到强训练聚焦。** 公开研究关注弱到强训练 —— 使用较弱的监督模型训练更强的学生模型。AARs 在该任务上优于人类研究者。

### 为什么沙箱外日志很重要

日志完整性属性，是受控实验与可能被破坏的实验之间的区别。如果 AARs 能编辑自己的日志，它们就能掩盖失败或可疑的实验。Anthropic 直接指出：该属性是信任研究输出的必要条件，而且验证它并非易事。

在基础设施术语里的类似做法是追加式日志并写穿到外部存储（例如 S3 object-lock，不可变的 Datadog 日志）。代理没有修改过去记录的凭证。这与 2026 年企业安全团队用来构建抵抗勒索软件的审计链路的模式相同。

### 规定性工作流程的权衡

Anthropic 的报告指出，规定性工作流程常常限制 AAR 的灵活性并降低性能。换言之：如果人类设计者逐步写出研究计划，AARs 的表现不如当它们被赋予高层目标并自由分解时。这与代理式研究（LangGraph、微软的 Agent Framework）中的结果一致 —— 过度指定的脚手架会伤害有能力的代理。

安全含义较为微妙。能提升 AAR 性能的灵活性，也是允许未经授权分解、创新工具使用和目标漂移的灵活性。权衡是一个设计空间决策：在不让代理有机会把目标泛化到分布外的前提下，给予多少灵活性才能获得足够的能力以击败人类？

### 压缩风险

RSP v3.0（Lesson 19）引入了一个 AI 研发能力阈值：能够以与人类+AI 工具具有竞争力的成本完全自动化 AI 研发流水线。DeepMind 的 FSF v3 包含一个类似的 ML 研发自治水平。两个框架都将该阈值视为需要升级控制的触发点。

AAR 比该阈值还差一步：它自动化了流水线的一部分（针对特定、范围明确任务的对齐研究），但没有实现端到端的能力开发闭环。时间线问题是差距收窄的速度有多快。

压缩的时间线是复合失败的担忧。如果对齐研究和能力研究以相似速度复合，失配风险面至少以与能力相当的速度增长。如果能力复合得更快（历史趋势），差距会扩大。这是认为 AAR 在有条件下是有益的论点：只有当研究过程值得信任时，每一个额外的对齐结果才会减少差距。

### AAR 不会替代的部分

人类研究者设定任务队列、审查结果，并保有宪法性权威。AARs 加速流水线的中间环节，而非始终端。Anthropic 的公开产出包含了 AAR 的贡献以及人类研究者关于发布、撤回和精炼的判断。

这与 Lesson 15 中的 propose-then-commit 模式一致地应用到研究本身：AARs 提出；人类决定。

## 使用方法

`code/main.py` 模拟一个多 AAR 的研究论坛。三个独立的“研究代理”并行运行，向一个追加式论坛发布发现，并可以选择性尝试篡改日志。

注意观察：

- 沙箱外日志如何捕捉日志篡改尝试。
- 在更难的问题上，具有自由分解的任务分配如何优于固定工作流程。
- 如果日志不是篡改可检测的，单个被攻破的 AAR 如何污染论坛的信号。

## 部署材料

`outputs/skill-aar-deployment-review.md` 是一份关于自动化对齐研究流水线的预部署评审。

## 练习

1. 运行 `code/main.py`。比较 “fixed-workflow” 与 “free-decomposition” 设置。自由分解总是获胜吗，还是存在某些问题类别使得固定工作流程具有竞争力？

2. 修改模拟器，使其中一个代理尝试篡改日志。确认追加式日志能检测到这一点。写一段话精确描述日志中检测签名的样子。

3. 阅读 Anthropic 的弱到强 AAR 报告。指出 AARs 在哪个具体子任务上击败了人类研究者。是什么使该子任务适合自动化？

4. 设计一个任务队列分配策略，在 AAR 灵活性（更好结果）与规定性工作流程约束（更易审计）之间取得平衡。描述你如何对两种策略做 A/B 测试。

5. 阅读 RSP v3.0 的 AI R&D-4 阶段。在一段话中描述你认为会越过该阈值，而当前 AAR 尚未达到的能力点。

## 关键词

| Term | What people say | What it actually means |
|---|---|---|
| AAR | "Automated Alignment Researcher" | 以自治方式在对齐问题上运行的 Claude Opus 4.6 实例 |
| Weak-to-strong training | "Training a stronger model with a weaker supervisor" | 经典的可扩展监督基准：AARs 在此项上优于人类 |
| Shared forum | "Where agents publish findings" | 追加式（append-only）、沙箱外存储的论坛 |
| Out-of-sandbox log | "Agent cannot edit its own record" | 可检测篡改并写穿到外部存储 |
| Prescribed workflow | "Step-by-step plan from human designer" | 约束 AAR；相比自由分解常常降低性能 |
| Free decomposition | "Agent decides how to break the task" | 更有能力，但更难审计 |
| AI R&D threshold | "RSP/FSF capability level" | 能以具有竞争力的成本实现研发流水线的完全自动化 |
| Compressed timeline | "Alignment vs capability race" | 如果能力比对齐增长得更快，失配风险会增加 |

## 延伸阅读

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — 主要来源。  
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — AI 研发阈值框架。  
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 更广的代理自治框架。  
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 与 RSP 类似的 ML 研发自治水平。  
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — AARs 攻克问题的底层问题。