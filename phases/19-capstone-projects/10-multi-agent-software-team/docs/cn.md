# Capstone 10 — 多代理软件工程团队

> SWE-AF 的 factory 架构、MetaGPT 的基于角色的提示词、AutoGen 0.4 的类型化 actor 图、Cognition 的 Devin，以及 Factory 的 Droids 在 2026 年趋于一致的形态：一位架构师做计划，N 位编码者在并行 worktree 中工作，评审者做闸门，测试者做验证。并行 worktree 把墙钟时间转化为吞吐量。共享状态与交接协议成为故障表面。这个 capstone 的任务是搭建该团队，在 SWE-bench Pro 上评估，并报告哪些交接点会失败以及失败频率。

**Type:** Capstone  
**Languages:** Python / TypeScript (agents), Shell (worktree scripts)  
**Prerequisites:** Phase 11 (LLM 工程), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 17 (infrastructure)  
**Phases exercised:** P11 · P13 · P14 · P15 · P16 · P17  
**Time:** 40 小时

## 问题

单代理的编码流水线在大任务上遇到天花板。原因并非单个代理能力弱，而是一个 200k 令牌的上下文无法同时容纳架构计划、四个并行代码切片、评审注释与测试输出。多代理工厂将问题拆分：架构师拥有总体计划，编码者在并行 worktree 中各自实现子任务，评审者做闸门，测试者做验证。SWE-AF 的“factory”架构、MetaGPT 的角色分工、AutoGen 的类型化 actor 图 —— 三种表述描述的其实是同一结构。

失败面在于交接。架构师设计了编码者无法实现的方案；编码者产生冲突的 diff；评审者批准了幻觉性修复；测试者与仍在写作的编码者发生竞态。你需要构建该团队，在 50 个 SWE-bench Pro 问题上运行，跟踪每一次交接，并发布事后分析报告。

## 概念

角色是类型化的代理。架构师（Claude Opus 4.7）读取 issue，写出计划，并把计划拆成带有明确接口的子任务。编码者（Claude Sonnet 4.7，N 个并行实例，每个在一个 `git worktree` + Daytona 沙盒中）独立实现子任务。评审者（GPT-5.4）读取合并后的 diff，决定批准或请求具体修改。测试者（Gemini 2.5 Pro）在隔离环境运行测试套件并报告通过/失败及相关产物。

通信通过共享任务看板（基于文件或 Redis）。每个角色消费其被允许处理的任务。交接采用 A2A 协议的类型化消息。协调要点包括：合并冲突解决（由协调器角色或自动三路合并完成）、共享状态同步（在编码者开始后计划被冻结；重新规划作为独立事件）、以及评审闸门（评审者不得批准自己提出的改动或自己所作的改动）。

令牌放大（token amplification）是隐藏成本。每个角色边界都会增加摘要提示和交接上下文。一次 40 回合的单代理运行在四个角色之间可能变成 160 次交互。评估标准特别衡量令牌效率相对于单代理基线，因为问题不是“多代理能否工作”，而是“按花费计算是否更胜一筹”。

## 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- Orchestration: LangGraph，带共享状态 + 每个代理的子图  
- Messaging: A2A protocol（Google 2025），用于类型化的代理间消息  
- Models: Opus 4.7（架构师），Sonnet 4.7（编码者），GPT-5.4（评审者），Gemini 2.5 Pro（测试者）  
- Worktree 隔离：每个编码者使用 `git worktree add` + Daytona 沙盒  
- Merge coordinator：自定义三路合并 + LLM 辅助冲突解决  
- Eval：SWE-bench Pro（50 个 issue）、SWE-AF 场景、HumanEval++ 单元测试  
- Observability：Langfuse，带角色标记的 span 和每代理令牌计费  
- Deployment：K8s，每个角色作为单独的 Deployment + 针对 backlog 的 HPA

## 构建步骤

1. Task board。基于文件的 JSONL，包含类型化消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。代理根据标签订阅。  
2. Architect。读取 GitHub issue，使用 Opus 4.7 和一个要求明确子任务接口（被修改的文件、公开函数、测试影响）的计划模板。发出一个带子任务 DAG 的 `plan_request`。  
3. Coders。N 个并行 worker，每个从看板认领一个子任务。每个 worker 创建一个新的 `git worktree add` 分支加上 Daytona 沙盒。实现子任务并提交。发出包含补丁和测试差异的 `diff_ready`。  
4. Merge coordinator。在所有编码者完成后，将 N 个分支三路合并到 staging 分支。仅在文件级重叠时触发 LLM 辅助的冲突解决。  
5. Reviewer。GPT-5.4 读取合并后的 diff。不得批准它自己撰写的 diff。发出 `approved`（无操作）或带回相关编码者的 `review_feedback`（具体变更请求）。  
6. Tester。Gemini 2.5 Pro 在干净的沙盒运行测试套件。捕获产物。发出 `test_passed` 或 `test_failed`（含堆栈追踪）。失败的测试循环回出问题的子任务编码者。  
7. Handoff accounting。每条跨角色边界的消息在 Langfuse 中创建一个 span，记录负载大小和所用模型。计算每个子任务的令牌放大（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。  
8. Eval。在 50 个 SWE-bench Pro issue 上运行。将 pass@1 和每解决问题的美元成本与单代理基线（一个 Sonnet 4.7 在单个 worktree）进行比较。  
9. Post-mortem。对于每个失败的 issue，识别破坏交接的点（计划过于模糊、合并冲突、评审误批准、测试抖动等）。生成交接失败直方图。

## 使用示例

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] 计划：4 个子任务（parser、cache、api、migration）
[board]     已分派到 4 个编码者的并行 worktree
[coder-A]   子任务 parser  -> 42 行，本地测试通过
[coder-B]   子任务 cache   -> 88 行，本地测试通过
[coder-C]   子任务 api     -> 31 行，本地测试通过
[coder-D]   子任务 migration -> 19 行，本地测试通过
[merge]     三路合并：0 个冲突
[reviewer]  对 cache 提出评论（线程池大小）；已路由到 coder-B
[coder-B]   修订：92 行；提交
[reviewer]  批准
[tester]    412 个测试全部通过
[pr]        已打开 #3382   4 位编码者，1 次修订，$4.90，18 分钟
```

## 交付物

`outputs/skill-multi-agent-team.md` 是交付成果。给定一个 issue URL 和并行度，团队会产生一个可合并的 PR，并提供按角色的令牌计费明细。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 在匹配的 50 个 issue 子集上的 pass@1 |
| 20 | Parallel speedup | 墙钟时间对比单代理基线 |
| 20 | Review quality | 在注入错误探针上的误批准率 |
| 20 | Token efficiency | 每解决问题的总令牌数对比单代理 |
| 15 | Coordination engineering | 合并冲突解决方案、交接失败直方图 |
| **100** | | |

## 练习

1. 在运行过程中向一个 diff 注入一个明显的错误（在主体之前多加一个 `return None`）。测量评审者的误批准率。调整评审者提示词，直到误批准率低于 5%。  
2. 缩减为两个编码者（架构师 + 编码者 + 评审者 + 测试者，编码者顺序执行两个子任务）。对比墙钟时间和通过率。  
3. 将合并协调器替换为单写者约束（子任务修改互不重叠的文件集）。度量架构师在规划上的额外负担。  
4. 将评审者从 GPT-5.4 换成 Claude Opus 4.7。测量误批准率与令牌成本差异。  
5. 增加第五个角色：文档撰写者（Haiku 4.5）。评审通过后生成变更日志。衡量文档质量是否值得额外的令牌开销。

## 术语表

| Term | 常说法 | 实际含义 |
|------|--------|---------|
| Parallel worktree | “隔离分支” | 通过 `git worktree add` 为每个编码者生成一个独立的工作树 |
| Task board | “共享消息总线” | 基于文件或 Redis 的类型化消息存储，代理订阅其感兴趣的条目 |
| Handoff | “角色边界” | 任何从一个角色上下文传到另一个角色的消息 |
| Token amplification | “多代理开销” | 跨所有角色的总令牌数 / 单代理完成同一任务的令牌数 |
| A2A protocol | “Agent-to-agent” | Google 的 2025 年代理间类型化消息规范 |
| Merge coordinator | “整合者” | 负责三路合并并调停冲突的组件 |
| False approval | “评审幻觉” | 评审者批准了含已知错误的 diff |

## 延伸阅读

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 2026 年参考的多代理工厂  
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 基于角色的多代理框架  
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft 的类型化 actor 框架  
- [Cognition AI (Devin)](https://cognition.ai) — 参考产品  
- [Factory Droids](https://www.factory.ai) — 另一个参考产品  
- [Google A2A protocol](https://developers.google.com/agent-to-agent) — 代理间消息规范  
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — 隔离子系统文档  
- [SWE-bench Pro](https://www.swebench.com) — 评估目标