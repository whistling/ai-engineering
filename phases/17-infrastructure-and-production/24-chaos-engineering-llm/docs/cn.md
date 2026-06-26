# 用于 LLM 生产的混沌工程

> 到 2026 年，针对 LLM 的混沌工程已成为独立学科。在生产环境运行实验前的先决条件：定义好的 SLI/SLO、trace+metric+log 可观察性、自动回滚、运行手册、值班。架构有四个平面：control（实验调度器）、target（服务、基础设施、数据存储）、safety（护栏 + 中止 + 流量过滤）、observability（指标 + 跟踪 + 日志），以及反馈回路（反馈到 SLO 调整）。护栏是强制性的：burn-rate 告警在每日错误预算消耗 > 2x 预期时暂停实验；抑制窗口 + trace-ID 关联用于去重告警噪声。节奏：每周小型金丝雀 + SLO 复审；每月游戏日 + 事故回顾；每季度跨团队弹性审计 + 依赖映射。LLM 特有的实验：内存过载、网络故障、提供方宕机、畸形提示词、KV 缓存驱逐风暴。工具：Harness Chaos Engineering（基于 LLM 的推荐、缩小爆炸半径、与 MCP 工具集成）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes 原生）。

**Type:** 学习
**Languages:** Python (stdlib，玩具混沌实验运行器)
**Prerequisites:** Phase 17 · 23 (AI 的 SRE), Phase 17 · 13 (可观察性)
**Time:** ~60 分钟

## 学习目标

- 列出五个混沌工程的先决条件（SLI/SLO、可观察性、回滚、运行手册、值班）并解释跳过任意一项会如何破坏该实践。
- 画出四个平面（control、target、safety、observability）以及流入 SLO 的反馈回路示意图。
- 枚举五个 LLM 特定实验（内存过载、网络故障、提供方故障、畸形提示词、KV 驱逐风暴）。
- 根据技术栈选择一个工具 — Harness、LitmusChaos、Chaos Mesh。

## 问题描述

传统技术栈的混沌测试已是成熟实践。LLM 技术栈增加了新的失败模式。一个 4K 令牌的提示词中含有一个中毒字符可能会使分词器停顿 12 秒。上游提供方返回 429；你的网关重试；重试放大并发后你的服务发生 OOM。突发负载下的 KV 缓存驱逐风暴导致重新预填充的级联，饱和计算资源。

这些问题都不会出现在单元测试中。混沌工程是让你在用户发现问题之前先行发现它们的办法。

## 概念

### 先决条件

在没有以下条件的情况下不要在生产环境运行混沌实验：

1. **SLI/SLO** — 定义好的服务级指标与目标。
2. **可观察性** — traces、metrics、logs，并接入仪表盘。
3. **自动回滚** — Phase 17 · 20 的策略标志回滚。
4. **运行手册** — 结构化的，应遵循 Phase 17 · 23。
5. **值班** — 有人可以响应。

缺少任何一项都会让混沌变成真正的事故。

### 四个平面 + 反馈

**Control 平面** — 实验调度器（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**Target 平面** — 服务、pod、节点、负载均衡、数据存储。

**Safety 平面** — 紧急开关、抑制窗口、爆炸半径限制、错误预算闸门。

**Observability 平面** — 常规指标 + trace-ID 关联，用于区分混沌引发的失败和自然失败。

**反馈回路** — 发现会回馈到 SLO 调整、运行手册更新、代码修复。

### 护栏是强制性的

- **Burn-rate 告警**：当每日错误预算消耗超过预期的 2x 时暂停实验。
- **抑制窗口**：在实验爆炸半径范围内静默非实验产生的告警。
- **Trace-ID 关联**：所有实验引发的错误带上标签，以便值班人员可以去重。

### 五个 LLM 特定实验

1. **内存过载** — 通过发送长上下文请求并提高并发来触发 KV 缓存抢占风暴。观察：服务是优雅地丢弃流量还是直接崩溃？
2. **网络故障** — 切断推理网关与提供方之间的连接。观察：回退是否在 SLA 内触发？（Phase 17 · 19）
3. **提供方故障模拟** — 对 OpenAI 返回 100% 的 429。观察：路由是否切换到 Anthropic？（Phase 17 · 16, 19）
4. **畸形提示词** — 注入会使分词器停滞的负载（例如深度嵌套的 Unicode、大的 UTF-8 码点）。观察：单个请求是否会锁住 worker？
5. **KV 驱逐风暴** — 通过饱和 vLLM 的块预算强制驱逐。观察：LMCache 能否恢复，还是服务降级？

### 节奏

- **每周** — 在预发布环境进行小型金丝雀实验，或在生产中以 5% 的流量演练。
- **每月** — 针对特定场景安排的游戏日；跨团队参与；事后回顾。
- **每季度** — 跨团队弹性审计；依赖映射更新。

### 工具链

- **Harness Chaos Engineering** — 商业产品；基于 LLM 的实验推荐；缩小爆炸半径；与 MCP 工具集成。
- **LitmusChaos** — CNCF 已毕业；基于 Kubernetes workflow。
- **Chaos Mesh** — CNCF 沙箱级；Kubernetes 原生 CRD 风格。
- **Gremlin** — 商业；广泛支持。
- **AWS FIS** / **Azure Chaos Studio** — 云厂商的托管混沌服务。

### 从小处开始

第一个实验：在稳定流量下杀掉一个 decode replica pod。观察重路由与恢复。如果这个看起来安全且有效，逐步升级到网络混沌。

第一个 LLM 特定实验：注入一个提供方的 429，持续 5 分钟。观察回退行为。大多数团队会发现他们的回退链并没有充分测试。

### 你应该记住的数字

- 四个平面：control、target、safety、observability。
- Burn-rate 暂停阈值：每日错误预算消耗 > 2x。
- 节奏：每周金丝雀、每月游戏日、每季度审计。
- 五个 LLM 实验：内存、网络、提供方、畸形提示词、KV 风暴。

## 使用示例

`code/main.py` 模拟三个带有安全平面闸门的混沌实验。报告哪些实验会触发 burn-rate 中止。

## 交付物

本课生成 `outputs/skill-chaos-plan.md`。根据技术栈和成熟度，挑选前三个实验和推荐工具。

## 练习

1. 运行 `code/main.py`。哪个实验触发了 burn-rate 闸门，为什么？
2. 为基于 vLLM 的 RAG 服务设计前五个混沌实验。包含成功判据。
3. 你的 burn-rate 告警暂停了实验。如何判断根本原因是混沌实验引起还是自然故障？
4. 论证混沌应该只在预发布环境运行还是也应该在生产运行。什么时候在生产运行是正确的选择？
5. 列出三种通用网络混沌无法复现的 LLM 特定失败模式。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SLI / SLO | "service targets" | 指标 + 目标；必备先决条件 |
| Blast radius | "scope" | 实验影响的服务 / 用户范围 |
| Burn-rate alert | "budget gate" | 当错误预算消耗率 > 2x 时触发 |
| Game day | "monthly drill" | 安排好的跨团队混沌演练 |
| LitmusChaos | "CNCF workflow" | 已毕业的 CNCF Kubernetes 混沌工具 |
| Chaos Mesh | "CNCF CRD" | CNCF 沙箱级的 Kubernetes 原生混沌 |
| Harness CE | "commercial AI-assisted" | Harness 提供的带 AI 推荐的混沌产品 |
| Malformed prompt | "tokenizer bomb" | 会使分词器停滞的输入 |
| KV eviction storm | "preemption cascade" | 大规模驱逐触发的重新预填充级联 |

## 延伸阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)