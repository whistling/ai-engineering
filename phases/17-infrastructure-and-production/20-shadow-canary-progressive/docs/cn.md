# Shadow Traffic, Canary Rollout, and Progressive Deployment for LLMs

> LLM 发布结合了软件部署中最困难的部分：没有单元测试、故障模式分散、信号滞后。流程是 (1) 影子模式（shadow mode）— 将生产请求复制到候选模型，记录并比较，对用户零影响；可以捕获明显的分布问题，但不是质量保证；(2) 金丝雀发布（canary rollout）— 渐进性流量迁移 10% → 25% → 50% → 75% → 100%，每一步都有闸门；跟踪延迟分位数、每次请求成本、错误/拒绝率、输出长度分布、用户反馈率；(3) A/B 测试在稳定性确认后用于不同替代方案的比较。非确定性是不可约的 — 在相同输入的多次运行中因 GPU 浮点运算非结合性和批量大小差异可能导致高达 15% 的准确率波动。成本是可变的，不是常数 — 一个性能提升 20% 的模型每次调用可能贵 3 倍。回滚速度决定成败：如果回滚需要重新部署，你就太慢了。策略存在于配置/开关；模型存在于注册表并固定摘要；回滚 = 翻转策略 + 恢复阈值 + 在几秒内将旧模型固定回去。

**Type:** 学习  
**Languages:** Python (stdlib, 玩具 canary 进度 模拟器)  
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 21 (A/B Testing)  
**Time:** ~60 分钟

## 学习目标

- 区分影子模式（零影响比较）、金丝雀发布（在线渐进）和 A/B（稳定后对比）。
- 列举五个 LLM 特有的金丝雀指标（延迟、每次请求成本、错误/拒绝、输出长度、用户反馈）。
- 解释为什么 LLM 的非确定性（高达 15%）会改变“稳定”在发布中的含义。
- 设计一个以秒为级别（策略翻转）而不是小时级别（重新部署）的回滚路径。

## 问题场景

你发布了一个新模型。离线评估显示准确率提升 3%。你在生产中启用了它。24 小时内，成本上升 40%，用户点踩增加 8%，有三个客户工单报告“奇怪的回答”。你回滚。重新部署需要 3 小时。你的周末被毁掉了。

上述每一项都是可避免的。影子模式本可以在任何用户感知到之前发现 40% 的成本飙升。金丝雀在点踩上升时会在 10% 就停下。策略标志回滚只需 30 秒。纪律性弥合了“离线评估看起来不错”和“真实用户满意”之间的鸿沟。

## 概念

### 影子模式

将候选模型置于与生产相同的请求流；输出被记录，但不返回给用户。对用户零影响。记录内容：

- 输出内容（与生产的 diff）。
- Token 计数（成本差异）。
- 延迟。
- 拒绝和错误。

能捕获：成本暴涨、长度回归、明显的拒绝变化、严重错误。不能捕获：用户会感知到的质量差异。影子是冒烟测试，不是质量测试。

### 金丝雀发布

带闸门的渐进流量迁移。典型进度：1% → 10% → 25% → 50% → 75% → 100%。每一步对以下 5 个指标进行闸门检查：

1. **延迟分位数** — P50、P95、P99。违规条件：canary 的 P99 > 基线的 1.5 倍。
2. **每次请求成本** — 加权美元。违规条件：高于基线 >20%。
3. **错误 / 拒绝率** — 包括 5xx 和显式拒绝。违规条件：是基线的 2 倍。
4. **输出长度分布** — 均值 + P99。违规条件：分布发生漂移。
5. **用户反馈率** — 点踩 / 工单提交率。违规条件：是基线的 1.5 倍。

### 非确定性是新的方差

相同的输入会产生不相同的输出。原因包括：

- GPU 浮点运算的非结合性（浮点归约顺序因批次不同而变化）。
- 批量大小差异（同一 prompt 在 128 批次 vs 16 批次里）。
- 采样（temperature > 0）。

测得：在相同评估集上，多次运行的准确率变化可高达 15%。在发布中“稳定”意味着指标在预期噪声范围内而非与基线完全相同。将闸门设置在噪声地板之上。

### 成本是变量

一个表现好 20% 的模型每次调用可能贵 3 倍。每次请求成本是五个闸门之一。发布一个“更好”但破坏单位经济性的模型是需要回滚的情形。

### 回滚是武器

- 策略标志（feature flag 系统）：在配置中翻转百分比；耗时秒级。
- 模型固定（注册表摘要）：固定的模型不会自动升级。
- 回滚 = 恢复标志 + 将摘要设回之前的模型。秒级完成，不是小时。

如果你的栈需要重新部署才能回滚，那么在发布前先修好它。

### 工具链

**Argo Rollouts** / **Flagger** — Kubernetes 的渐进交付控制器。可与 Istio/Linkerd 加权路由集成。

**Istio 加权路由** — 网格层面的流量拆分。

**KServe / Seldon Core** — 带内置金丝雀功能的模型服务。

**特性开关（Feature flags）** — LaunchDarkly、Flagsmith、Unleash。策略级别翻转，无需重新部署。

### 指标节奏

金丝雀闸门每 5–15 分钟检查一次，取决于流量规模。1% 流量且 10 req/min 在每个窗口可得 50–150 个样本 —— 足以用于延迟，但用户反馈仍然很嘈杂。10% 流量则约 10 倍样本。每一步应该暂停足够长以在该步累积足够样本。

### A/B 步骤是可选的

如果新模型行为明显不同（行为不同、成本曲线不同、语气不同），在金丝雀通过后在 50% 处进行 A/B 测试。若只是改进版本，在金丝雀闸门通过后可直接推进到 100%。

### 需要记住的数字

- 金丝雀进度：1% → 10% → 25% → 50% → 75% → 100%。
- 非确定性上限：在相同输入上运行间的方差可达 15%。
- 五个金丝雀指标：延迟、成本、错误/拒绝、输出长度、用户反馈。
- 成本闸门：超过基线 >20% 为违规。
- 回滚：秒级而非小时级。

## 使用方法

`code/main.py` 模拟带注入回归的金丝雀发布。报告金丝雀在哪个阶段停止以及哪个闸门触发。

## 发布交付物

本课产出 `outputs/skill-rollout-runbook.md`。给定候选模型、基线和容忍风险，设计影子→金丝雀→100% 的发布计划。

## 练习

1. 运行 `code/main.py`。注入 25% 的成本回归。金丝雀在哪个阶段停止？  
2. 你的新模型离线准确率提升 3% 但每次请求成本 +18%。是否发布？取决于策略 —— 写出两种路径。  
3. 设计一个端到端小于 60 秒的回滚。列出所需基础设施。  
4. 非确定性在你的评估上表现为 ±7%。设置金丝雀闸门以避免误报。你会使用哪些倍数？  
5. 影子模式在金丝雀之前捕获到 40% 的成本激增。写出会在影子中触发的告警规则。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Shadow mode | "duplicate to new" | 零影响的发送到候选模型以便记录 |
| Canary | "progressive traffic" | 带闸门的渐进性面向用户的发布 |
| Gates | "rollout checks" | 阻止推进的指标阈值 |
| Non-determinism | "LLM variance" | 不可约的运行间差异 |
| Policy flag | "flag flip rollback" | 配置级回滚，秒级而非小时级 |
| Model pin | "registry digest" | 指向模型版本的不可变引用 |
| Argo Rollouts | "K8s progressive" | Kubernetes 原生的金丝雀/回滚控制器 |
| KServe | "inference K8s" | 带金丝雀原语的模型服务 |
| Istio weighted | "mesh split" | 服务网格的流量拆分器 |

## 延伸阅读

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)  
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)  
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)  
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)  
- [Flagger docs](https://docs.flagger.app/)