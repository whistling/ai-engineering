# FinOps for LLMs — 单位经济学与多租户归因

> 传统的 FinOps 在 LLM 支出面前失效。成本是基于令牌的交易，而不是资源在线时长。标签无法映射 —— API 调用是一次交易，不是一个资产。工程决策（提示词设计、上下文窗口、输出长度）即财务决策。2026 年的运营手册在第一天就要埋点三个归因维度：按用户（`user_id`）用于席位定价与扩展，按任务（`task_id` + `route`）用于产品面成本与优先级，按租户（`tenant_id`）用于单位经济学与续约。四个令牌层——提示、工具、记忆、响应——合并计费会掩盖开销。多租户产品的强制执行阶梯：按租户的速率限制（2-3 倍预期峰值，返回明确的 429 + `Retry-After`）；每日支出上限（合同上限的 1.5-3 倍触发速率收紧 + 告警）；支出 z-score > 4 时的终止开关（自动暂停 + 呼叫值班）。归因模式：打标签并聚合、遥测连接器（trace-ID → 计费；准确度最高）、抽样并外推、基于模型的分配、事件溯源、实时流式。单位指标：每个已解决查询的成本、每个生成产物的成本 —— 而不是 $/M 令牌。事后打标签总会漏掉边缘情况；在请求创建时就埋点。

**Type:** 学习  
**Languages:** Python (标准库，带终止开关的轻量成本归因模拟器)  
**Prerequisites:** Phase 17 · 13 (可观测性), Phase 17 · 14 (缓存)  
**Time:** ~60 分钟

## 学习目标

- 解释为什么传统 FinOps（标签 + 分层）在 LLM 支出上失效，并列出三个新的归因维度。
- 列举四个令牌层（提示、工具、记忆、响应）以及为何单一桶计费会掩盖成本。
- 为多租户产品设计一套强制执行阶梯（速率 → 支出上限 → 终止开关）。
- 选择一个单位指标（每已解决查询/每产物的成本），而不是 $/M 令牌。

## 问题

你的账单显示 $40,000。你不知道：
- 哪个租户花了这笔钱。
- 是哪个产品功能导致的。
- 是否有个别用户在滥用。
- 是提示膨胀、工具调用，还是记忆放大造成的。

云资源（EC2、S3）可以通过标签传播到账单明细，但 LLM API 调用不会自动打标签 —— 你必须在调用处盖上用户/任务/租户并传递下去。事后归因总会漏掉边缘情况。

## 概念

### 三个归因维度

**Per-user** (`user_id`)：是谁在产生成本。用于席位定价、扩展对话，识别高价值用户。

**Per-task** (`task_id` + `route`)：哪个产品界面在消耗成本。用于功能优先级、决定是否下线高成本功能。

**Per-tenant** (`tenant_id`)：哪个客户是有利可图的。用于单位经济学、续约定价、分层阈值。

在调用处同时埋点这三者，从第一天开始。事后处理总是更差。

### 四个令牌层

| Layer | Example | Typical % of total |
|-------|---------|---------------------|
| Prompt | system + user input | 40-60% |
| Tool | tool-call results fed back | 20-40% (agent workloads) |
| Memory | prior conversation / retrieved docs | 10-30% |
| Response | model output | 10-30% |

把这四项放在同一个桶里会让优化变成盲操。把它们在你的归因 schema 中拆开。

（注：这里使用的术语“提示/工具/记忆/响应”对应常见的工程层级，分别映射到提示词工程、工具调用、检索/历史上下文与模型输出的令牌消耗。）

### 强制执行阶梯

1. **速率限制（Rate limit）** 按租户。设置为预期峰值的 2-3 倍。返回 429 并带 `Retry-After`。租户会感到摩擦，但不会收到意外账单。

2. **每日支出上限（Daily spend cap）** 按租户。设置为合同上限的 1.5-3 倍。触发时：收紧速率限制并告警客户成功（CS）。

3. **终止开关（Kill switch）** 当支出 z-score 相对于租户基线 > 4 时触发。自动暂停该租户；呼叫值班并升级到运维 + CS。

### 归因模式

- **打标签并聚合（Tag-and-aggregate）**：在请求头盖元数据；后端聚合。简单但粗糙。
- **遥测连接器（Telemetry joiner）**：通过 trace ID 将跟踪与计费连接。准确度最高。成熟团队采用此法。
- **抽样 + 外推（Sampling + extrapolation）**：抽样 5-10%，乘以因子。对粗略支出有效；会漏掉尾部。
- **基于模型的分配（Model-based allocation）**：用回归等模型推断成本驱动因子。适用于无标签的遗留数据。
- **事件溯源（Event-sourced）**：将成本作为事件流（Kafka / Kinesis）记录。支持实时化。
- **实时流式（Real-time streaming）**：仪表盘亚秒级更新。

### 单位指标（Cost per X）

$/M 令牌是厂商话术。产品层指标应当是：

- 每个已解决的支持工单成本。
- 每篇生成文章的成本。
- 每个成功代理任务的成本。
- 每个用户会话分钟的成本。

将成本绑到产品结果上。否则优化就没有锚点。

### 成本归因追踪示例形状

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

在每次调用时发出此信息。存入数据湖。按维度聚合。Phase 17 · 13 的可观测性栈是放置这些数据的地方。

### 复合节流栈（compounded-savings stack）

栈：缓存 + 批处理 + 路由 + 网关。四项同时生效时：
- L2 缓存（Phase 17 · 14）：输入成本约降低 10 倍。
- 批处理（Phase 17 · 15）：节省约 50%。
- 路由到廉价模型（Phase 17 · 16）：成本再降低约 60%。
- 网关效率（Phase 17 · 19）：减少冗余与重试带来的损耗。

最优堆叠情况下：相对于天真的基线可降到约 ~5-10%。大多数团队启用了 2-3 个杠杆；很少有团队同时开启全部四个。

### 需要记住的数字

- 归因维度：按用户、按任务、按租户。
- 四个令牌层：提示、工具、记忆、响应。
- 终止开关阈值：支出 z-score > 4。
- 单位指标：每已解决查询的成本，而不是 $/M 令牌。
- 堆叠优化：理论上可降到基线的 ~5-10%。

## 使用说明

`code/main.py` 模拟了一个具有三层强制执行阶梯的多租户 LLM 服务。注入一个滥用租户并演示终止开关触发。

## 部署产出

本课产出 `outputs/skill-finops-plan.md`。基于产品与规模，设计归因 schema 与强制执行阶梯。

## 练习

1. 运行 `code/main.py`。终止开关在什么 z-score 触发？你如何选择阈值？
2. 设计一个按租户、按任务的成本仪表盘。你会优先构建哪 5 个视图？
3. 如果你最大的租户在单位经济学上是亏损的，提出三项按客户影响排序的干预措施。
4. 计算一个支持产品的每已解决工单成本：3M 令牌/工单，~800 工单/天，使用 GPT-5 缓存费率。
5. 论证事后打标签是否有可能奏效。在哪些场景下它是可以接受的？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Per-user attribution | "user-level cost" | `user_id` stamped on every call |
| Per-task attribution | "feature cost" | `task_id` + `route` identify product surface |
| Per-tenant attribution | "customer cost" | `tenant_id`; drives unit economics |
| Four token layers | "cost layers" | prompt + tool + memory + response |
| Rate limit | "429 guard" | Per-tenant ceiling enforced at gateway |
| Daily spend cap | "daily ceiling" | Tenant-scoped budget with alert |
| Kill switch | "auto-pause" | Spend z-score > 4 triggers auto-suspension |
| Cost per resolved | "product unit metric" | Cost tied to product outcome, not tokens |
| Telemetry joiner | "trace-to-billing" | Highest-accuracy attribution pattern |
| Stacked optimization | "cache+batch+route+gateway" | Compounding savings to ~5-10% baseline |

## 延伸阅读

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)