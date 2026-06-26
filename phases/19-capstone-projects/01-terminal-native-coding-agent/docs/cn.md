# Capstone 01 — 终端原生编码代理

> 到 2026 年，编码代理的形态已经稳定。一个 TUI 绑定、一个有状态的计划、一个沙箱化的工具表面、一个基于前沿模型的计划—执行—观察—恢复循环。Claude Code、Cursor 3 和 OpenCode 从 50 米外看起来都差不多。本 Capstone 要求你端到端构建一个 —— 从 CLI 输入到拉取请求输出 —— 并在 SWE-bench Pro 上将其与 mini-swe-agent 和 Live-SWE-agent 进行比较。你将学到难点不在于模型调用，而在于工具循环、沙箱以及 50 轮运行的成本上限。

**Type:** 综合项目  
**Languages:** TypeScript / Bun（harness），Python（评估脚本）  
**Prerequisites:** Phase 11（LLM engineering）、Phase 13（tools and protocols）、Phase 14（agents）、Phase 15（autonomous systems）、Phase 17（infrastructure）  
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18  
**Time:** 35 小时

## 问题

到 2026 年，编码代理已成为占主导地位的 AI 应用类别。Claude Code（Anthropic）、Cursor 3（含 Composer 2 和 Agent Tabs）、Amp（Sourcegraph）、OpenCode（112k stars）、Factory Droids 与 Google Jules 都在发布同构架的变体：终端绑定、受权限控制的工具表面、一个沙箱，以及围绕前沿模型构建的计划—执行—观察循环。前沿模型的范围很窄 —— Live-SWE-agent 使用 Opus 4.5 在 SWE-bench Verified 上达到了 79.2% —— 但工程工艺很广。大多数失败模式并不是模型犯错，而是工具循环不稳定、上下文污染、代币费用暴走和破坏性文件系统操作。

你无法从外部推断这些代理的行为。你必须亲自构建一个，观察当 ripgrep 返回 8MB 匹配时循环在第 47 轮崩溃，然后重建截断层。这就是本 Capstone 的意义。

## 概念

harness 有四个表面。Plan 维护一个 TodoWrite 风格的状态对象，模型在每轮重写它。Act 派发工具调用（read、edit、run、search、git）。Observe 捕获 stdout / stderr / 退出码，进行截断，并将摘要反馈回去。Recover 在不耗尽上下文窗口或无限循环的情况下处理工具错误。2026 年的形态增加了一项：hooks。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact` —— 这些都是可配置的扩展点，操作人员可以在此注入策略、遥测和护栏。

沙箱使用 E2B 或 Daytona。每个任务在一个新的 devcontainer 中运行，挂载一个可读写的 git worktree。harness 永远不接触主机文件系统。worktree 在成功或失败后被拆除。成本控制在三层执行：每轮代币上限、每会话美元预算和硬性轮数上限（通常 50）。可观测性层使用 OpenTelemetry spans，遵循 GenAI 语义约定，发送到自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- Harness 运行时：Bun 1.2 + Ink 5（终端内的 React）
- 模型接入：通过 OpenRouter 统一 API 使用 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最难的任务）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱：E2B 沙箱（JS SDK）或 Daytona devcontainers
- 代码搜索：以子进程形式调用 ripgrep；为 17 种语言预编译 tree-sitter 解析器
- 隔离：每个任务执行 `git worktree add`，在成功/失败后清理
- 评估框架：SWE-bench Pro（verified 子集）+ Terminal-Bench 2.0 + 你自己的 30 题留出集
- 可观测性：OpenTelemetry SDK，使用 `gen_ai.*` semconv → 自托管 Langfuse
- PR 发布：具有细粒度权限的 GitHub App，作用域仅限目标仓库

## 构建指南

1. TUI 与命令循环。用 Ink 搭建 Bun 项目。接受 `agent run <repo> "<task>"`。打印分割视图：计划窗格（上部）、工具调用流（中部）、代币预算（底部）。实现在 Ctrl-C 时触发 `SessionEnd` hook 再退出的取消行为。

2. 计划状态。定义一个类型化的 TodoWrite 模式（包含 pending / in_progress / done 条目和备注）。模型在每轮以工具调用形式重写完整状态 —— 不允许它做增量变更。将计划持久化到 `.agent/state.json`，以便崩溃后恢复。

3. 工具表面。定义六个工具：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（status / diff / commit / push）。通过 MCP StreamableHTTP 暴露，使 harness 与传输方式无关。每个工具返回截断输出（每次调用上限 4k tokens）。

4. 沙箱封装。每个任务启动一个 E2B 沙箱。执行 `git worktree add -b agent/$TASK_ID` 创建一个新分支。所有工具调用都在沙箱内执行。主机文件系统不可达。

5. Hooks。实现所有八种 2026 年的 hook 类型。至少连接四个用户自定义 hook：（a）`PreToolUse` 破坏性命令护栏，阻止在 worktree 之外执行 `rm -rf`；（b）`PostToolUse` 代币记账；（c）`SessionStart` 预算初始化；（d）`Stop` 写出最终追踪包。

6. 评估循环。克隆 SWE-bench Pro 的 30 个 issue 子集（Python）。针对每个问题运行你的 harness。将结果与 mini-swe-agent 在 pass@1、每题轮数和每题美元成本上比较。将结果写入 `eval/results.jsonl`。

7. 成本控制。硬性截止：50 轮、200k 上下文、每题 $5。`PreCompact` hook 在达到 150k 时将早期轮次总结为 prior-state block，以释放新观察的空间同时不丢失计划。

8. PR 发布。成功时的最后一步是 `git push` + 调用 GitHub API 打开一个 PR，PR 正文包含计划和 diff 摘要。

## 使用示例

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

（注：上例为终端交互示范，命令与工具输出保持原样）

## 交付

交付物位于 `outputs/skill-terminal-coding-agent.md`。给定一个仓库路径和任务描述，它在沙箱中运行完整的计划—执行—观察循环，并返回一个 PR URL 加上追踪包。该 Capstone 的评分标准：

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs baseline | 在 30 道匹配的 Python 题上，你的 harness 与 mini-swe-agent 的对比 |
| 20 | Architecture clarity | Plan/act/observe 的分离、hook 面、工具 schema —— 与 Live-SWE-agent 的布局对照评审 |
| 20 | Safety | 沙箱逃逸测试、权限提示、破坏性命令护栏通过红队测试 |
| 20 | Observability | 追踪完整性（100% 的工具调用有 span）、每轮代币记账 |
| 15 | Developer UX | 冷启动 < 2s、崩溃恢复能恢复计划、Ctrl-C 在中途工具时能干净取消 |
| **100** | | |

## 练习

1. 将基础模型从 Claude Sonnet 4.7 换成在 vLLM 上部署的 Qwen3-Coder-30B。比较 pass@1 和 每题美元成本。报告开放模型在哪些场景下表现欠佳。

2. 添加一个 `reviewer` 子代理，在 PR 发布前读取 diff 并可以请求修订循环。测量是否会导致假阳性审查使 SWE-bench 的通过率低于单代理基线（提示：通常会）。

3. 对沙箱进行压力测试：写一个尝试 `curl` 外部 URL 的任务，和一个尝试写入 worktree 外部的任务。确认两者都被 `PreToolUse` hook 阻止，并记录尝试行为。

4. 用一个更小的模型（Haiku 4.5）实现 `PreCompact` 摘要。测量在 3 倍压缩下计划保真度损失多少。

5. 将 MCP StreamableHTTP 传输替换为 stdio。基准测试冷启动和每次调用延迟。为仅限本地使用选择一个赢家。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Harness | “The agent loop” | 围绕模型的代码，负责派发工具、维护计划状态并强制执行预算 |
| Hook | “Agent event listener” | 在 harness 的八个生命周期事件之一上运行的用户自定义脚本 |
| Worktree | “Git sandbox” | 一个独立路径下的关联 git 签出；可丢弃且不影响主克隆 |
| TodoWrite | “Plan state” | 一个类型化的待办/进行中/完成列表，模型在每轮重写整个内容 |
| StreamableHTTP | “MCP transport” | 2026 年的 MCP 修订：带双向流的长连接 HTTP；替代 SSE |
| Token ceiling | “Context budget” | 对输入+输出代币的每轮或每会话上限；触发压缩或终止 |
| pass@1 | “Single-attempt pass rate” | 在不重试或不窥视测试集的情况下第一次运行即解决的 SWE-bench 题目比例 |

（表中术语保留为英文原词以便对应实现与文档）

## 延伸阅读

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的参考 harness  
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs 与 Composer 2 的产品说明  
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — 用于与 SWE-bench harness 比较的最小基线  
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%  
- [OpenCode](https://opencode.ai) — 开源 harness，112k stars  
- [SWE-bench Pro leaderboard](https://www.swebench.com) — 本 Capstone 目标的评估榜单  
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据等说明  
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用与代币使用的 span 模式