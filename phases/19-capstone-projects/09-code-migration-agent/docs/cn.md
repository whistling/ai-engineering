# Capstone 09 — Code Migration Agent (Repo-Level Language / Runtime Upgrade)

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 迁移器在 2026 年定下了标杆。Moderne 的 OpenRewrite 在大规模场景下做确定性 AST 重写。Grit 以 codemod 风格的 DSL 针对同一问题。生产模式通常将二者结合：一个用于安全重写的确定性基座，加上用于模糊情况的 agent 层、每个分支的沙箱构建，以及在打开 PR 之前变绿的测试工具链。capstone 的目标是迁移 50 个真实仓库并发布通过率与失败分类学。

**Type:** Capstone  
**Languages:** Python (agent), Java / Python (targets), TypeScript (dashboard)  
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)  
**Phases exercised:** P5 · P7 · P11 · P13 · P14 · P15 · P17  
**Time:** 30 小时

## 问题

大规模代码迁移是 2026 年编码 agent 最清晰的生产化应用之一。衡量标准明确（迁移后测试套件是否通过？）、回报显著（将 Java 8 整 fleet 升级是需要大量人力的项目），且基准公开（MigrationBench 的 50 仓库子集）。Moderne 的 OpenRewrite 负责确定性部分。agent 层负责 OpenRewrite recipes 无法覆盖的所有情况：模糊的重写、构建系统漂移、长尾语法、传递依赖破坏。

你将构建一个 agent，其输入为 Java 8 仓库（或 Python 2 仓库），输出为一个 CI 变绿的迁移分支。你需要衡量通过率、测试覆盖率保留、每仓库成本，并构建失败分类学。与仅使用确定性工具的 baseline 并列对比可以揭示 agent 实际带来的价值点。

## 概念

流水线包含两层。确定性基座（Java 使用 OpenRewrite，Python 使用 libcst）负责大部分机械化、可审核且安全的重写：imports、方法签名、空值安全编辑、try-with-resources、弃用 API 替换。它速度快并生成可审计的 diff。agent 层（使用 OpenAI Agents SDK 或 LangGraph，运行在 Claude Opus 4.7 和 GPT-5.4-Codex 上）处理 recipes 无法解决的情形：构建文件升级（Maven/Gradle/pyproject）、传递依赖冲突、测试抖动、定制注解。

每个仓库在 Daytona 沙箱中创建分支，预装目标运行时。agent 以迭代方式工作：运行构建、分类失败、应用修复、重试。硬性上限：每仓库 30 分钟、每仓库 $8、最多 20 次 agent 回合。如果所有测试通过且覆盖率无下降，分支将打开 PR。否则，该仓库会以证据附带的方式归入某个失败类。

失败分类学是交付物。在 50 个仓库中，什么出问题最多？传递依赖？定制注解？构建工具版本？与迁移无关的测试抖动？每个类别给出计数和示例 diff。未来的 recipe 作者可以针对前三大类进行优化。

## 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- 确定性基座：OpenRewrite（Java）或 libcst（Python）
- Agent：OpenAI Agents SDK 或 LangGraph，运行于 Claude Opus 4.7 + GPT-5.4-Codex
- 沙箱：Daytona per-branch devcontainers，预装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准：Amazon MigrationBench 50 仓库子集（Java 8 -> 17）、Google App Engine Py2-to-Py3 仓库
- 测试工具链：并行运行器，使用 Jacoco（Java）或 coverage.py（Python）收集覆盖率
- 可观测性：Langfuse + 每个仓库带有每个 diff chunk 的 trace bundle
- 仪表盘：失败分类仪表盘，按类计数并展示示例 diff

## 构建步骤

1. **Recipe pass。** 先运行 OpenRewrite（Java）或 libcst（Python）recipes。捕获 70–80% 的机械化迁移并提交为 "recipe" 提交。

2. **Build trial。** 在 Daytona 沙箱中安装目标运行时并运行构建。如果变绿，跳过 agent 阶段直接跑测试；如果失败，交由 agent 处理。

3. **Agent loop。** 使用 LangGraph 和工具集：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。Agent 对失败进行分类（依赖、语法、测试、构建工具）并应用针对性修复。重试构建。

4. **预算上限。** 每仓库 30 分钟实时时间、$8 成本、20 次 agent 回合。任一限制触发将中止并将仓库归入 "budget_exhausted"，同时附带当前 diff。

5. **测试与覆盖门控。** 构建变绿后运行测试套件。将覆盖率与基线仓库比较。如果覆盖率下降超过 2%，归类为 "coverage_regression"。

6. **打开 PR。** 成功后推送分支，打开 PR，附上 diff 以及哪些 recipes 应用和哪些提交由 agent 编写的摘要。

7. **失败分类学。** 对每个失败仓库打标签：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建仪表盘。

8. **50 仓库运行。** 在 MigrationBench 子集上执行。报告按类的通过率、每仓库成本、覆盖率保留，以及与仅确定性工具的 baseline 对比。

## 使用示例

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付物

`outputs/skill-migration-agent.md` 是最终交付件。对于给定仓库，它先执行确定性 recipes，随后进入 agent 循环以产生一个变绿的迁移分支，或者将仓库按失败分类记录归档。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | MigrationBench pass rate | 50-repo subset pass@1 |
| 20 | Test-coverage preservation | Mean coverage delta vs base |
| 20 | Cost per migrated repo | $/repo on passing runs |
| 20 | Agent / deterministic-tool integration | Fraction of fixes that OpenRewrite handled vs agent authored |
| 15 | Failure analysis write-up | Taxonomy completeness with exemplars |
| **100** | | |

## 练习

1. 仅使用 OpenRewrite（无 agent）运行迁移流水线。将通过率与完整流水线对比。识别 agent 真正带来差异的案例。

2. 实现一个 "lint-clean" 检查：迁移后运行样式 linter（Java 用 spotless，Python 用 ruff）。如果出现新的 lint 错误则失败 PR。统计覆盖率保留但样式回归的比率。

3. 添加一个 "最小化 diff" 优化器：在 agent 分支通过测试后进行二次清理以去除不必要更改。报告 diff 大小的减少量。

4. 扩展到第三种迁移：Node 18 到 Node 22。复用沙箱包装；将 recipe 层替换为自定义 codemod。

5. 测量 time-to-first-green-build (TTFGB) 作为 UX 指标。目标：p50 < 10 分钟。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Deterministic substrate | "Recipe engine" | OpenRewrite / libcst：具有安全保证的声明式 AST 重写 |
| Codemod | "Code-modifying program" | 一条机械化修改源代码的重写规则 |
| Build drift | "Tool version skew" | Maven / Gradle / uv 在主要版本间的细微行为差异 |
| Failure class | "Taxonomy bucket" | 仓库未迁移成功的标注原因：依赖、语法、测试、构建工具、预算等 |
| Coverage delta | "Coverage preservation" | 迁移分支相对于基线的测试覆盖率百分比变化 |
| Agent turn | "Tool-call round" | agent 循环中的一次 plan -> act -> observe 周期 |
| Budget exhaustion | "Hit the ceiling" | 仓库耗尽 30 分钟 / $8 / 20 回合限制仍未通过 |

注意：文中术语已使用标准中文 AI 工程术语翻译（例如：提示词工程、RAG、嵌入、微调、上下文窗口、少样本、思维链、护栏、函数调用、投机性解码、位置嵌入、自注意力、指令微调、分布式训练 等）以保证与行业惯例的一致性。

## 参考阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 年的权威基准  
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — 确定性基座参考  
- [OpenRewrite documentation](https://docs.openrewrite.org) — recipe 编写文档  
- [Grit.io](https://www.grit.io) — 另一种 codemod DSL  
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考  
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — 另一迁移基准  
- [libcst](https://github.com/Instagram/LibCST) — Python 的确定性基座  
- [Daytona sandboxes](https://daytona.io) — 每分支沙箱参考