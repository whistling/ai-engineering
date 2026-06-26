# Failure Modes — MAST, Groupthink, Monoculture, Cascading Errors

> The reference taxonomy for 2026 is **MAST** (Cemri et al., NeurIPS 2025, arXiv:2503.13657), derived from 1642 execution traces across 7 state-of-the-art open-source MAS showing **41–86.7% failure rate**. Three root categories: **Specification Problems** (41.77%) — role ambiguity, unclear task definitions; **Coordination Failures** (36.94%) — communication breakdowns, state desync; **Verification Gaps** (21.30%) — missing validation, absent quality checks. The **Groupthink** family (arXiv:2508.05687) adds: monoculture collapse (same base model → correlated failures), conformity bias (agents reinforce each other's errors), deficient theory of mind, mixed-motive dynamics, cascading reliability failures. Cascading example: retry storms where a payment failure triggers order retries, which trigger inventory retries, which overwhelm inventory service (10x load in seconds — needs circuit breakers). Memory poisoning: one agent's hallucination enters shared memory, downstream agents treat it as fact; accuracy decays gradually, making root-cause diagnosis painful. **STRATUS** (NeurIPS 2025) reports 1.5x mitigation-success improvement via specialized detection / diagnosis / validation agents. This lesson treats failure modes as first-class engineering targets.

**Type:** 学习  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 16 · 13（共享内存）, Phase 16 · 14（共识与 BFT）, Phase 16 · 15（投票与辩论拓扑）  
**Time:** ~75 分钟

## 问题

多智能体系统在真实任务上的失败率为 41–86.7%（Cemri 等，2025 年在 7 个开源 MAS 上的测量）。单靠“再加几个 agent”并不能调试这些失败。失败存在结构性原因。MAST 分类法给出了这些类别。本课将每个类别映射到具体的检测、诊断和缓解模式，以便这些数字不再显得任意。

到 2026 年的生产实践是将失败模式视为设计输入。在你能指出每个 MAST 类别并说明已部署的缓解措施之前，你的架构就还“不够好”。

## 概念

### MAST 类别

**Specification Problems（Specification 问题，41.77% 的失败）。** 任务定义不够严格。示例：

- 角色模糊：两个 agent 都认为自己是审阅者。
- 任务欠说明：“总结这个”但用户想要特定角度。
- 成功判定隐式：agent 无法判断是否完成。

缓解措施：
- 编写明确的角色合约。每个 agent 的 prompt 要说明它做什么以及不做什么。
- 为每项任务定义验收测试。在 agent 开始前定义“完成看起来像 X”。
- 起飞前规范检查：由单独的 agent 审查任务定义再派发。

**Coordination Failures（协调失败，36.94%）。** 通信或状态断裂。

示例：
- 两个 agent 在没有同步的情况下更新共享状态。
- agent 之间的消息丢失（队列故障、超时）。
- 状态漂移：agent A 认为任务已完成；agent B 仍在执行。

缓解措施：
- 版本化共享状态并使用乐观并发控制。
- 对关键消息使用显式确认（重试直到收到 ack）。
- 周期性状态同步检查点；及早检测漂移。

**Verification Gaps（验证缺口，21.30%）。** 输出没有独立校验。

示例：
- 一个 agent 声称成功；无人验证。
- 一串 agent 互相信任前一环的输出。
- 对组合行为的测试覆盖不足。

缓解措施：
- 独立验证 agent（见第 13 课）。只读、独立的数据访问。
- 显式交接合约：“A 的输出必须通过检查器 C 才能由 B 开始处理”。
- 结果日志用于事后分析。

### Groupthink 家族（arXiv:2508.05687）

当 agents 同质化或模仿彼此时，会出现五类相关失败：

**Monoculture collapse（单一文化崩溃）。** 相同的基础模型或训练数据 → 相关错误。当三个 agent 共享同一 LLM 时，它们共享同样的幻觉模式。

**Conformity bias（从众偏差）。** agents 向最响亮或最自信的同伴靠拢，即使该同伴是错的。

**Deficient ToM（心智理论缺失）。** agents 未能建模彼此的信念；协调失败（见第 18 课）。

**Mixed-motive dynamics（动机混合动力学）。** 部分一致的激励导致各方朝妥协中间漂移，结果谁都不满意。

**Cascading reliability failures（级联可靠性故障）。** 一个组件的错误模式触发依赖组件的错误模式。

### 级联示例 — retry storm（重试风暴）

一个经典的 2026 事件模式：

```
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

修复是经典的：**电路断路器（circuit breakers）**。当下游错误率超过阈值时，短路并使用缓存或默认结果。此外为每个请求限定重试预算。

电路断路器是少数可以直接从分布式系统借用到多智能体失败缓解策略之一，几乎不需修改。

### Memory poisoning（记忆中毒，重述）

来自第 13 课：一个 agent 的幻觉进入共享内存，后续 agents 将其当作事实；在 MAST 术语中，这是发生在共享内存层面的验证缺口。

渐进的准确性衰减是其症状。你不会得到崩溃；你会得到缓慢漂移，溯源诊断非常痛苦。

缓解：追加写入日志、数据溯源、不可写的验证器。见第 13 课已覆盖的内容。

### STRATUS — 专门化 agent 用于故障检测

STRATUS（NeurIPS 2025）报告称，当部署以下 agent 时，缓解成功率提高 1.5 倍：

- **Detection agent（检测 agent）。** 监视症状模式（高不一致率、重试峰值、准确率漂移）。
- **Diagnosis agent（诊断 agent）。** 给定症状，从 MAST 分类法中推断可能根因。
- **Validation agent（验证 agent）。** 缓解施行后检查症状是否消失。

这是 SRE 式的事故响应在 agent 系统上的应用。三种角色都可以是具有专门 prompt 的 LLM agent。

### 故障模式审计

2026 年的最佳实践是按年（或每次大版本）做故障模式审计：

1. **采样追踪。** 收集 ~1000 条真实执行追踪。
2. **分类。** 将每条失败追踪映射到 MAST + Groupthink 类别。
3. **计算按类别失败率。** 哪些类别在你的系统中占主导？
4. **为缓解措施排序。** 哪个修复可以消除最多失败？
5. **挑选 2–3 个缓解措施。** 实施；下季度重新审计。

这一纪律比具体选择更重要。没有审计，失败会混成噪声，永远得不到系统性处理。

### 当系统无声失败时

最危险的失败类别是无声的正确性失败。会大声失败（崩溃、异常、告警）的系统可以被监控。产生貌似合理但错误输出的系统无法通过异常日志检测到。这就是为什么验证缺口在每次失败的成本上最昂贵，尽管按次数它们只占 21.30%。

要投资以下内容：
- 抽样的人为复核。
- 金标准数据集回归测试。
- 对重要输出的跨 agent 交叉校验。

### 失败 vs 缓慢失败

有些失败是即时发生的；有些是缓慢的。即时失败（超时、模式不匹配、认证错误）易于检测且代价低。缓慢失败（记忆中毒、单一文化漂移、角色模糊）检测与预防成本高。

2026 年的工程手段：为缓慢失败设计代理度量，以便在漂移成为可见错误前捕捉。协议覆盖率、重试率、输出长度分布，以及连续 agent 版本间的编辑距离都是有用的代理指标。

## 搭建

`code/main.py` 实现：

- `FailureTaxonomy` — 将模拟事件分类到 MAST + Groupthink 类别。
- `CircuitBreaker` — 经典模式；当错误率超过阈值时打开。
- `RetryStormSimulator` — 展示级联失败；可切换电路断路器开/关。
- `DetectionAgent` — 脚本化的 STRATUS 风格症状匹配器。

运行：

```
python3 code/main.py
```

预期输出：
- 无电路断路器时的重试风暴：库存错误激增（模拟）。
- 使用电路断路器时：在阈值处被限制；提供降级模式响应。
- 检测 agent 标记该模式并指出 MAST 类别。

## 使用

`outputs/skill-mast-auditor.md` 对一个多智能体系统运行 MAST 风格的故障模式审计。追踪 → 分类 → 缓解排序。

## 投产

生产中的故障模式纪律：

- **每季度进行 MAST 审计。** 不是每年。随着系统增长，类别会发生变化。
- **到处都要有电路断路器。** 对任何外部依赖的出站调用都应加上。默认打开阈值设在 5–10% 的错误率。
- **金标准数据集。** 小型、高质量、人工审定。每周回归测试。
- **STRATUS 三人组。** 检测 + 诊断 + 验证 agent 监控生产。先从检测 agent 开始；当症状噪声大时再加上诊断 agent。
- **失败预算。** 按类别明确 SLO。超额触发暂停发版讨论。

## 练习

1. 运行 `code/main.py`。确认电路断路器能限制重试风暴。改变失败阈值并观察权衡。
2. 实现一个 **缓慢失败代理度量**：三个并行 agent 之间的一致率。当它急剧下降时触发警报。通过逐渐相关化 agent 输出来模拟单一文化漂移。
3. 阅读 Cemri 等（arXiv:2503.13657）。选取他们的 7 个 MAS 系统中的一个，并映射其前三大失败类别。这些结果与 MAST 预测有何异同？
4. 阅读 Groupthink 论文（arXiv:2508.05687）。识别五种模式中在生产环境中最难检测的一种。提出一个代理度量。
5. 为你熟悉的某个具体多智能体系统设计一个 STRATUS 风格的检测-诊断-验证三人组。检测监视哪些症状？诊断建议哪些缓解措施？验证如何确认它们有效？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MAST | "The 2026 taxonomy" | Cemri 2025；3 个根类 + 14 个子类的失败类型。 |
| Specification Problem | "Role ambiguity" | 任务或角色定义不足；agents 不知道该做什么。 |
| Coordination Failure | "State drift" | agents 之间的通信或同步断裂。 |
| Verification Gap | "No one checked" | 输出在没有独立验证的情况下被接受。 |
| Groupthink family | "Homogeneity failures" | 单一文化、从众、心智理论缺失、动机混合、级联故障。 |
| Monoculture collapse | "Same model, same hallucinations" | 共享基础模型或训练数据导致的相关错误。 |
| Retry storm | "Cascading error amplification" | 一个失败触发重试，进而放大下游负载。 |
| Circuit breaker | "Fail fast on error rate" | 当错误率超阈值时打开；使用默认值或缓存短路。 |
| STRATUS | "Incident response trio" | 检测 + 诊断 + 验证 agents。缓解成功率 1.5x。 |
| Memory poisoning | "Hallucinations propagate" | 共享内存事实被污染；后续 agents 在受污染数据上推理。 |

## 拓展阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分类法，NeurIPS 2025  
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) — 单一文化、从众及五族分类法  
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) — NeurIPS 2025 会议论文集条目（检测 + 诊断 + 验证）  
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — 经典电路断路器参考书  
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 生产故障模式笔记