# Capstone 16 — GitHub Issue-to-PR Autonomous Agent

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 都在 2026 年收敛出相同的产品形态：给 issue 打标签，得到一个 PR。在云沙箱中运行一个 agent，验证测试通过，并提交一个准备好审阅的 PR 并附上理由。难点在于自动重现仓库的构建环境、防止凭据泄露、执行每仓库预算并确保 agent 无法执行 force-push。本毕业设计实现自托管版本，并在成本与通过率上与托管替代品进行比较。

**Type:** 毕业设计  
**Languages:** Python（agent）、TypeScript（GitHub App）、YAML（Actions）  
**Prerequisites:** Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（agents）、Phase 15（autonomous）、Phase 17（infrastructure）  
**Phases exercised:** P11 · P13 · P14 · P15 · P17  
**Time:** 30 小时

## 问题

异步云端编码 agent 与交互式编码 agent（capstone 01）属于不同的产品范畴。用户体验是 GitHub 标签。你给一个 issue 打上 `@agent fix this`，一个 worker 在云沙箱中启动，克隆仓库，运行测试，编辑文件，验证，并打开一个包含 agent 理由的 PR。没有交互式循环，也没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都走向了这一方向。

工程挑战是明确的：环境重现（agent 必须从头构建仓库，而非使用已有的开发镜像缓存）、不稳定的测试（必须重跑或隔离）、凭据范围限制（使用最小化的细粒度 GitHub App 权限）、每仓库预算执行以及禁止 force-push 策略。该毕业设计测量通过率、成本和安全性，并与托管替代品进行比较。

## 概念

触发器是一个 GitHub webhook（issue 标签或 PR 评论）。调度器将工作入队到 ECS Fargate 或 Lambda。worker 将仓库拉入 Daytona 或 E2B 沙箱，并基于仓库推断出通用 Dockerfile（语言、框架）。agent 运行 mini-swe-agent 或 SWE-agent v2 循环，使用 Claude Opus 4.7 或 GPT-5.4-Codex。迭代流程：读取代码、提出修复、应用补丁、运行测试。

验证是门控步骤。必须在沙箱中通过完整 CI 才能打开 PR。计算覆盖率差异；如果下降超出阈值，仍会打开 PR 但打上 `needs-review` 标签。agent 将理由作为 PR 描述发布，并在 PR 中创建一个可由审阅者 @agent 以便后续交互的线程。

安全性通过两个不同的 GitHub 面向进行范围限制：App 提供短期的安装令牌，具有 `workflows: read` 和有限的仓库 contents/PR 权限；分支保护（而非 App 权限）强制执行“禁止直接写入 `main`”和“禁止 force-push”——App 永远不会被加入绕过白名单。对 `.github/workflows` 的路径作用只在 worker 端通过 allow-list 的方式对拟议 diff 强制检查，因为 GitHub App 权限无法做到路径级别的粒度限制。调度器在每个仓库上强制日预算上限（例如每仓库每日最多 5 个 PR，每 PR 20 美元）。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

（以上架构图为流程概览：从 GitHub Label/Comment 到 Lambda 调度，再到 Fargate/Runner 启动沙箱、运行 agent 循环、在沙箱内验证 CI 与覆盖率差异，最后通过 GitHub App 打开 PR。）

## 技术栈

- 触发：具有细粒度令牌的 GitHub App；Webhook 接收器由 Lambda 或 Fly.io 承载  
- Worker：ECS Fargate 任务（或 GitHub Actions 自托管 runner）  
- 沙箱：每任务的 Daytona devcontainer 或 E2B 沙箱  
- Agent 循环：mini-swe-agent 基线或 SWE-agent v2，使用 Claude Opus 4.7 / GPT-5.4-Codex  
- 检索：tree-sitter repo-map + ripgrep  
- 验证：在沙箱内运行完整 CI + 覆盖率差异门控  
- 可观测性：Langfuse，PR body 中链接每个 PR 的 trace 存档  
- 预算：按仓库的日美元上限；每仓库每日最大 PR 数

## 构建步骤

1. **GitHub App。** 使用细粒度安装令牌：issues read+write、pull_requests write、contents read+write、workflows read。通过分支保护（唯一能做到的面向）强制 “禁止直接推送到 `main`” 和 “禁止 force-push”；App 不在绕过列表中。因为 GitHub App 权限无法按路径粒度控制，worker 需要在拟议 diff 上执行 allow-list 检查以强制 “禁止对 `.github/workflows` 的写入”。

2. **Webhook 接收器。** Lambda 函数接收 issue 标签 / PR 评论 webhook。按标签 `@agent fix this` 过滤并将任务入队到 SQS。

3. **调度器。** 从 SQS 弹出任务。执行每仓库每日预算检查。启动一个 ECS Fargate 任务，传入仓库 URL、issue 正文和一个新的 Daytona 沙箱实例。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（pip/venv、pnpm、go mod、cargo）。如果仓库缺少 Dockerfile，则动态生成一个 Dockerfile。

5. **Agent 循环。** 使用 mini-swe-agent 或 SWE-agent v2（例如 Claude Opus 4.7）。工具链包括：ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。硬限制：$20 成本上限、30 分钟墙钟时间、30 次 agent 回合上限。

6. **验证。** 循环结束后，在沙箱中运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率差异。若 CI 红（失败）：中止，不打开 PR。若覆盖率下降超过 2%：仍打开 PR，但附加 `needs-review` 标签。

7. **PR 发布。** 推送 agent 分支。通过 GitHub API 打开 PR，内容包含：标题、理由、diff 摘要、trace URL、成本与回合数。

8. **凭据卫生。** Worker 使用短期的 GitHub App 安装令牌运行。日志在归档前进行敏感信息清洗以去除秘密。

9. **评估。** 使用 30 个有不同难度的内部种子 issue。测量通过率、PR 质量（diff 大小、样式、覆盖率）、成本、延迟。并与 Cursor Background Agents 与 AWS Remote SWE Agents 在相同问题集上进行对比。

## 使用示例

```
# 在 github.com 上
  - 用户将 issue #842 打上 `@agent fix this` 标签
  - 14 分钟后出现 PR #1903
  - body:
    > 修复了 widget.dedupe() 中因比较器为 null 导致的 NPE。
    > 添加了回归测试 widget_test.go::TestDedupeNullComparator。
    > 覆盖率差异：+0.12%
    > 回合数：7  成本：$1.80  Trace: langfuse:...
    > 标签：needs-review
```

## 交付物

`outputs/skill-issue-to-pr.md` 是交付物。一个 GitHub App + 异步云 worker，将带标签的 issue 转换为具有成本边界和受限凭据的可审阅 PR。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | 通过率（30 个 issue） | 端到端成功（CI 绿灯 + 覆盖率合格） |
| 20 | PR 质量 | diff 大小、覆盖率差异、样式合规性 |
| 20 | 每次解决问题的成本与延迟 | 每个 PR 的美元成本与墙钟时间 |
| 20 | 安全性 | 作用域令牌、每仓库预算、禁止 force-push、凭据卫生 |
| 15 | 操作员 UX | 理由注释、重试能力、@-mention 后续交互 |
| **100** | | |

## 练习

1. 添加一个 “修复不稳定测试” 模式：当标签为 `@agent stabilize-flake TestX` 时，在沙箱中运行该测试 50 次并提出一个能够稳定它的最小改动。

2. 在三个共享问题上与 Cursor Background Agents 比较成本。报告在哪些场景下哪种工具具有优势。

3. 实现一个预算仪表板：按仓库每日成本、按用户成本。对异常情况发出告警。

4. 构建一个 “dry-run” 模式：打开一个草稿 PR（draft PR），但不运行 CI，让审阅者以更低成本审查计划。

5. 添加保留策略：未在 7 天内合并且没有活动的 PR 分支自动删除。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GitHub App | "Scoped bot identity" | 具有细粒度权限 + 短期安装令牌的 App |
| Async cloud agent | "Background agent" | 在云沙箱中运行的非交互式 worker，而非终端式交互 |
| Environment inference | "Dockerfile synthesis" | 检测语言与包管理器，若缺失则生成 Dockerfile |
| Verification | "CI-in-sandbox" | 在 worker 中运行完整测试套件后再打开 PR |
| Coverage delta | "Coverage preservation" | 从基线到 agent 分支的测试覆盖率百分比变化 |
| Per-repo budget | "Daily ceiling" | 在调度器处执行的美元与 PR 数量上限 |
| Rationale | "PR body explanation" | agent 对更改内容与原因的总结；必须写入 PR 正文 |

## 参考阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 典型的异步云 agent 参考实现  
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考  
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案  
- [OpenAI Codex (cloud)](https://openai.com/codex) — 托管竞品  
- [Google Jules](https://jules.google) — Google 的托管版本  
- [Factory Droids](https://www.factory.ai) — 另一商业参考  
- [GitHub App documentation](https://docs.github.com/en/apps) — 作用域化 bot 身份的文档  
- [Daytona cloud sandboxes](https://daytona.io) — 沙箱参考