# 人类在环（Human-in-the-Loop）：Propose-Then-Commit（先提议后提交）

> 2026 年关于 HITL 的共识是明确的。它并不是“agent 提出请求，用户点击 Approve”。它是先提议后提交：将提议的操作持久化到可靠存储并带上幂等键；向审阅者展示意图、数据来源、涉及的权限、影响范围以及回滚计划；仅在正面确认后才提交；执行后再进行验证以确认副作用确实发生。LangGraph 的 `interrupt()` 加上 PostgreSQL 检查点、Microsoft Agent Framework 的 `RequestInfoEvent`、以及 Cloudflare 的 `waitForApproval()` 都实现了相同的形态。典型的失败模式是橡皮图章式批准：在未审阅的情况下点击了“Approve?”。文档中推荐的缓解措施是带有明确检查表的质询与响应（challenge-and-response）。

**Type:** 学习  
**Languages:** Python（stdlib，propose-then-commit 状态机，带幂等性）  
**Prerequisites:** Phase 15 · 12（持久化执行），Phase 15 · 14（触发线）  
**Time:** ~60 分钟

## 问题

一个 agent 发起一个操作。用户需要决定：批准还是不批准。如果决策是即时的，那很可能并非真正的审查。如果决策是结构化的，它会慢但更可信。工程问题在于如何让结构化审查成为最省力的路径。

2023 年左右的 HITL 模式是一个同步提示：“Agent 想要发送邮件给 X，正文为 Y —— 批准吗？”用户点击 Approve。大家感觉系统安全。但在实际中，这种界面会被大量橡皮图章式通过：用户快速批准，批准行为的预测能力很低，当 agent 出错时，审计日志显示一长串用户已经记不清的批准记录。

2026 年的模式——先提议后提交（propose-then-commit）——把 HITL 移到持久化基底，附加结构化元数据，并要求正面提交。每个托管 agent SDK 都会发布一个版本：LangGraph 的 `interrupt()`、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()`。API 名称可能不同；但模式相同。

## 概念

### propose-then-commit 状态机

1. 提议（Propose）。Agent 生成一个提议的操作。将其持久化到可靠存储（PostgreSQL、Redis、Durable Object 等）。包括：
   - intent（为什么 agent 要这么做）
   - data lineage（导致该提议的具体来源）
   - permissions touched（涉及哪些作用域 / 文件 / 端点）
   - blast radius（最坏情况影响范围）
   - rollback plan（如果提交后如何回滚）
   - idempotency key（每个提议唯一；重提交返回相同记录）
2. 展示（Surface）。审阅者可以看到包含全部元数据的提议。审阅者必须是人（而不是 agent 自我审查）。
3. 提交（Commit）。收到正面确认后，执行该操作。
4. 验证（Verify）。执行后读取并确认副作用确实发生。如果验证步骤失败，系统进入已知的异常状态并触发告警。

### 幂等键（idempotency key）

没有幂等键时，短暂故障后的重试可能导致已批准的操作被重复执行。具体示例：用户批准“从 A 转账 $100 给 B。”网络抖动导致工作流重试。用户只批准了一次，但转账执行了两次。幂等键将批准绑定到单一、唯一的副作用；第二次执行变为无操作。

这与 Stripe 和 AWS API 使用的幂等模式相同。Microsoft Agent Framework 文档也明确复用了该做法用于 agent 批准。

### 持久性：为什么批准可以超越进程生命周期

批准等待区是 agent 不拥有的一段状态。工作流被暂停（Lesson 12）。当批准到达时，工作流从精确的暂停点恢复。这就是为什么 LangGraph 将 `interrupt()` 与 PostgreSQL 检查点配对，而不是仅在内存中保存状态——两天后的批准仍能找到完整的工作流状态。

### 橡皮图章式批准与质询-响应（challenge-and-response）的缓解

默认的 HITL UI（“Approve”/“Reject” 按钮）会产生快速批准而没有真实审查。文件化的缓解策略：使用质询-响应检查表，要求在“批准”按钮可用前对特定问题给出肯定回答。具体形态：

- “您是否理解该操作会触及哪些资源？[ ]”
- “您是否已验证影响范围（blast radius）是可接受的？[ ]”
- “如果失败，您是否有回滚计划？[ ]”

这不是形式主义的官僚流程，而是一个强制函数。不能勾选这些项的审阅者要么请求澄清（升级），要么拒绝（安全默认）。Anthropic 的 agent 安全研究明确把基于检查表的 HITL 作为缓解橡皮图章式批准模式的方法之一。

### 什么算是有影响力的操作（consequential）

并非所有操作都需要先提议后提交。2026 年的指导：

- 有影响力的操作（始终 HITL）：不可逆写入、金融交易、外发通信、生产数据库更改、破坏性文件系统操作。
- 可逆的操作（有时 HITL）：本地文件编辑、暂存环境变更、具有明确回滚的可逆写入。
- 读取与检查（从不 HITL）：读取文件、列出资源、调用只读 API。

### 操作后的验证

“提交已运行”并不等于“副作用已发生”。网络分区和竞态条件可能导致工作流认为成功但后端并未持久化。验证步骤会在提交后重新读取目标资源以确认。这与数据库事务使用 `RETURNING` 子句或在 `PutObject` 后调用 AWS `GetObject` 的模式相同。

### 欧盟 AI 法案第 14 条

第 14 条要求对欧盟内高风险 AI 系统进行有效的人类监督。“有效”不是装饰性的。法规语言明确排除了橡皮图章式模式。Microsoft Agent Governance Toolkit 的合规文档指出，先提议后提交并配合质询-响应是能通过第 14 条审查的形态。

## 使用方法

`code/main.py` 在 stdlib Python 中实现了一个 propose-then-commit 状态机。持久化存储是一个 JSON 文件。幂等键是 (thread_id, action_signature) 的哈希。驱动程序模拟三种情况：干净的批准流程、短暂故障后的重试（不得重复执行）、以及橡皮图章默认与质询-响应流程的对比。

## 交付（Ship It）

`outputs/skill-hitl-design.md` 审查了一个针对 propose-then-commit 形态的建议 HITL 工作流，并标记了缺失的元数据、幂等性、验证或质询-响应层。

## 练习

1. 运行 `code/main.py`。确认对已批准提议的重试使用持久记录且不会重新执行。现在将幂等键改为包含时间戳，展示重试会导致重复执行。

2. 在提议记录中扩展 `rollback` 字段。模拟一个验证步骤失败的执行，展示回滚自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。识别该 API 包含的、玩具引擎缺失的一个元数据字段。将其添加并解释它防护了什么风险。

4. 为一个具体操作（例如“向公开的 Twitter 账号发帖”）设计一份质询-响应检查表。审阅者必须回答哪三个问题？为何选这三项？

5. 选择一个同步 “Approve?” 提示足够的案例（无需持久化存储）。解释原因，并指出你接受的风险类别。

## 关键术语

| Term | 人们怎么说 | 它实际的含义 |
|---|---|---|
| Propose-then-commit | “Two-phase approval” | 持久化的提议 + 正面提交 + 验证 |
| Idempotency key | “Retry-safe token” | 每个提议唯一；第二次执行为无操作 |
| Data lineage | “Where it came from” | 导致该提议的具体源内容 |
| Blast radius | “Worst case” | 操作出错时的影响范围 |
| Rubber-stamp | “Fast approval” | 在没有真实审查的情况下点击“Approve” |
| Challenge-and-response | “Forcing checklist” | 审阅者必须对特定问题作出肯定确认 |
| RequestInfoEvent | “MS Agent Framework primitive” | 带结构化元数据的持久化 HITL 请求 |
| `interrupt()` / `waitForApproval()` | “Framework primitives” | LangGraph / Cloudflare 上等价的框架原语 |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`、持久化批准。  
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 与 Durable Objects。  
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 HITL 作为长期风险缓解手段。  
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 针对高风险系统的人类监督法规基线。  
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 关于监督的宪制化框架。