# Action Budgets, Iteration Caps, and Cost Governors

> A mid-sized e-commerce agent's monthly LLM cost jumped from $1,200 to $4,800 after its team enabled the "order-tracking" skill. That is not a pricing bug. That is an agent that found a new loop and kept spending inside it. Microsoft's Agent Governance Toolkit (April 2, 2026) codifies the defense against this class: per-request `max_tokens`, per-task token and dollar budgets, per-day/month caps, iteration caps, tiered model routing, prompt caching, context windowing, HITL checkpoints on expensive actions, kill switches on budget breach. Anthropic's Claude Code Agent SDK ships the same primitives under different names. Financial velocity limits — e.g. cut access on >$50 in 10 minutes — catch loops faster than monthly caps.

**Type:** 学习  
**Languages:** Python（stdlib，分层成本治理模拟器）  
**Prerequisites:** Phase 15 · 10（权限模式），Phase 15 · 12（持久执行）  
**Time:** ~60 分钟

## The Problem

Autonomous agents spend real money on every turn. A chatbot's bad output is a bad reply; an agent's bad loop is a bill. The industry-documented term for the failure mode is "Denial of Wallet" — the agent keeps reasoning, keeps tool-calling, keeps billing, and nothing stops it because nothing was designed to.

修复并不是一个数字的问题，而是一组在不同时间尺度和粒度上叠加的限制：按请求、按任务、按小时、按日、按月。合理设计的限额栈能在几分钟内抓住失控循环，在数小时内发现慢性泄漏，在一天内拦截错误发布。对于长期运行且自主的 agent，同样的限额栈可以保证总体预算受控。

这是工程学的教训：数学本身很简单，失败的是执行纪律。下面列出的限额都在 Microsoft Agent Governance Toolkit 或 Anthropic Claude Code Agent SDK 文档中有所命名或描述。

## The Concept

### The cost-governor stack

1. **`max_tokens` per request.** 简单明了。防止单次调用产出无限制的完成长度。
2. **Per-task token budget.** 整个任务运行中不要超过 N 个 token。到达上限时强制停止。
3. **Per-task dollar budget.** 与 token 预算等价，但以货币计。Claude Code 中为 `max_budget_usd`。
4. **Per-tool call cap.** 每种工具最多调用 N 次，例如 `WebFetch`、`shell_exec` 等。
5. **Iteration cap (`max_turns`).** 限制 agent 循环迭代次数；防止无限推理循环。
6. **Per-minute / per-hour / per-day / per-month cap.** 滚动窗口限制；在不同时间尺度捕获泄漏。
7. **Financial velocity limit.** 例如：“10 分钟内消费超过 $50 则切断访问。”比月度上限更快捕获循环性燃烧。
8. **Tiered model routing.** 默认使用小模型；只有在分类器判断任务确实需要时才升级到大模型。
9. **Prompt caching.** 系统提示与稳定上下文存于提供方缓存；重发的 token 成本近乎为零。
10. **Context windowing.** 通过压缩/摘要保持活动上下文低于阈值；直接降低 token 成本。
11. **HITL checkpoints on expensive actions.** 在已知昂贵的操作（长时间工具调用、大文件下载、昂贵的模型升级）之前，要求人工确认。
12. **Kill switch on budget breach.** 任何限额触发时中止会话。记录触发原因；重新启用需要独立路径。

### Why the stack, not one cap

单一的月度上限只会在钱包快没了才抓到失控 agent。单一的每次请求上限无法在会话层面捕获问题。不同的失效模式需要不同时间尺度的防护：

- **Runaway loop**（agent 被卡在 5 秒重试中）：由 velocity limit 捕获。
- **Slow leak**（agent 每个任务执行量约为预期的 2 倍）：由日度上限捕获。
- **Bad release**（新版本使用 5 倍 token）：由周/月上限捕获。
- **Legitimate surge**（真实的需求激增，而非 bug）：由小时/日上限结合清晰的日志识别。

### Claude Code's budget surface

Claude Code Agent SDK 对外暴露（公开文档）：

- `max_turns` — 迭代上限。
- `max_budget_usd` — 美元上限；触发时会话中止。
- `allowed_tools` / `disallowed_tools` — 工具允许列表与拒绝列表。
- 在工具使用前的 Hook 点用于自定义成本记账。

将其与权限模式阶梯（第 10 课）结合，一个没有 `max_budget_usd` 的 `autoMode` 会话即为未受治理的自治。Anthropic 明确将 Auto Mode 框定为需要预算控制；分类器与成本控制是正交的。

### EU AI Act, OWASP Agentic Top 10

Microsoft 的 Agent Governance Toolkit 涵盖了 OWASP Agentic Top 10 和 EU AI Act 第 14 条（人工监督）要求。在欧盟生产环境中，日志记录与上限执行不是可选项。

### The observed $1,200 → $4,800 case

Microsoft 文档中的真实案例：一个电商 agent 在新增工具后月度成本翻了三倍。该工具允许 agent 在每次会话中轮询订单状态。没有循环检测、没有按工具上限、没有周比周增长警报。修复方法是对该工具添加按次上限并引入每日增长告警。这是一个通用模板：每个新工具表面都是一个潜在的循环；每个新工具都需要自身的上限和告警。

## Use It

`code/main.py` 模拟在有与没有分层成本治理栈情况下的 agent 运行。模拟 agent 在若干轮后会陷入轮询循环；分层治理栈能在 velocity 窗口内捕获该循环，而单一的月度上限要到几天后才会触发。

## Ship It

`outputs/skill-agent-budget-audit.md` 审核拟部署 agent 的成本治理栈并标记缺失的层级。

## Exercises

1. 运行 `code/main.py`。确认在轮询循环轨迹上，velocity limit 在 iteration cap 之前触发。现在禁用 velocity limit，测量在 iteration cap 捕获之前 agent “花费”了多少。

2. 为浏览器 agent（第 11 课）设计一个按工具的上限集合。哪个工具需要最严格的上限？哪个工具可以不设限且无风险？

3. 阅读 Microsoft Agent Governance Toolkit 文档。列出工具包中命名的每一种上限类型。将每一种映射到一个失效模式（runaway loop、slow leak、bad release、surge）。

4. 为一个现实任务（例如“在仓库中对 50 个 issue 做分类”）定价一个通宵无人看守的运行。将 `max_budget_usd` 设为你点估值的 2 倍。说明为何要设为 2 倍。

5. Claude Code 的 `max_budget_usd` 对会话聚合成本生效。设计一个你会在外部强制执行的补充 velocity limit。什么会触发切断？重新启用流程是什么样的？

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Denial of Wallet | "Runaway bill" | Agent loop generating spend with no cap to stop it — 代理持续消费且无人阻止的循环账单 |
| `max_tokens` | "Per-request cap" | 单次完成长度的上限 |
| `max_turns` | "Iteration cap" | 会话中 agent 循环迭代次数的上限 |
| `max_budget_usd` | "Dollar kill switch" | 会话成本上限；触发则中止会话 |
| Velocity limit | "Rate cap" | 在短窗口内的消费速率上限（例如 $50 / 10 分钟） |
| Tiered routing | "Small model first" | 先用廉价模型作为默认；仅在分类器判断必要时升级 |
| Prompt caching | "Cached system prompt" | 提示缓存：提供方侧缓存将重发提示的 token 成本降至接近零 |
| HITL checkpoint | "Human approval gate" | 人类在环检查点：在昂贵操作前要求人工确认 |

## Further Reading

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd`、工具允许列表等。  
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 成本治理检查点。  
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 提供方侧的成本控制。  
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) — 缓存机制。  
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长期 agent 的成本曲线研究。