# Skills and Agent SDKs — Anthropic Skills, AGENTS.md, OpenAI Apps SDK

> MCP 说明 “有什么工具可用”。Skills 说明 “如何完成一项任务”。到 2026 年，这三层堆栈同时存在。Anthropic 的 Agent Skills（开放标准，2025 年 12 月）以 SKILL.md 形式发布，支持渐进式披露。OpenAI 的 Apps SDK 是在 MCP 之上加上小部件元数据。AGENTS.md（到现在已在 60,000+ 仓库中）位于仓库根目录，作为项目级别的 agent 上下文。本课说明每一层覆盖的内容，并构建一个最小的 SKILL.md + AGENTS.md 包，可以在不同 agent 间迁移。

**Type:** 学习  
**Languages:** Python (stdlib, SKILL.md 解析器和加载器)  
**Prerequisites:** Phase 13 · 07（MCP 服务器）  
**Time:** ~45 分钟

## 学习目标

- 区分三层：AGENTS.md（项目上下文）、SKILL.md（可复用的做法）、MCP（工具）。
- 编写带有 YAML frontmatter 和渐进式披露的 SKILL.md。
- 将技能以文件系统方式加载到 agent 运行时中。
- 将一个技能与 MCP 服务器和 AGENTS.md 组合，使一个包在 Claude Code、Cursor 和 Codex 中通用。

## 问题场景

一位工程师把撰写发布说明的工作流提炼为多步骤提示：“读取最新合并的 PR。按领域分组。为每组总结。按照团队风格写变更日志条目。把草稿发到 Slack。”他们把它放在团队的 Notion 文档里。

现在他们想在 Claude Code、Cursor 和 Codex CLI 中使用这个工作流。每个 agent 加载指令的方式不同：Claude Code 的斜线命令、Cursor 的规则、Codex 的 `.codex.md`。工程师因此复制了三份工作流并维护三份拷贝。

AGENTS.md 和 SKILL.md 联合解决了这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的 agent 在会话开始时读取它。“这个项目怎么运作？有哪些约定？哪些命令运行测试？”
- **SKILL.md** 是可移植的捆绑：YAML frontmatter（name，description）+ markdown 正文 + 可选资源。支持技能的 agent 可按名称按需加载它们。
- **MCP**（Phase 13 · 06-14）处理技能需要调用的工具。

三层，其中一个可移植工件。

## 概念

### AGENTS.md (agents.md)

于 2025 年末推出，至 2026 年 4 月被 60,000+ 仓库采用。仓库根目录下的单个文件。格式示例：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agents 会在会话开始时读取此文件，并用它来为该项目校准它们的行为。到 2026 年，所有主流的代码类 agent 都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed 等。

### SKILL.md 格式

Anthropic 的 Agent Skills（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: 为最新合并的 PR 按照本项目风格撰写变更日志条目。
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter 声明了技能的标识。正文是在技能加载时展示给模型的提示。

### 渐进式披露

技能可以引用子资源，agent 只有在需要时才会去获取它们。示例结构：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 中写着“参见 style-guide.md 获取风格规则”。agent 只在技能实际运行时拉取 style-guide.md。这避免了在提示中膨胀大量模型可能无需的细节。

### 文件系统发现

Agent 运行时会扫描已知目录下的 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

按文件夹名和 frontmatter 中的 `name` 进行加载。Claude Code、Anthropic Claude Agent SDK、SkillKit（跨 agent）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话开始时加载技能，并将它们作为可调用的“agents”暴露在运行时内。当用户调用技能时，agent 的循环会将请求分派到相应技能。

### OpenAI Apps SDK

于 2025 年 10 月推出；直接构建在 MCP 之上。统一了 OpenAI 早期的 Connectors 和 Custom GPT Actions 到单一的开发表面。一个 Apps SDK 应用包含：

- 一个 MCP 服务器（工具、资源、提示）。
- 以及用于 ChatGPT UI 的小部件元数据。
- 可选的 MCP Apps `ui://` 资源用于交互界面。

相同的协议，更丰富的用户体验。

### 通过 SkillKit 实现跨 agent 可移植性

像 SkillKit 这样的工具以及其他跨 agent 分发层，会把单一的 SKILL.md 翻译成 32+ 个 AI agent 的原生格式（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）。一个事实来源，多方消费。

### 三层堆栈

| Layer | File | Loaded when | Purpose |
|-------|------|-------------|---------|
| AGENTS.md | repo root | session start | 项目级约定 |
| SKILL.md | skills directory | skill invoked | 可复用的工作流 |
| MCP server | external process | tools needed | 可调用的动作 |

三者可以组合：agent 在会话开始读取 AGENTS.md，用户调用技能，技能的指令包含 MCP 工具调用，agent 通过 MCP 客户端进行调度。

## 使用示例

`code/main.py` 提供了一个基于 stdlib 的 SKILL.md 解析器和加载器。它会在 `./skills/` 下发现技能，解析 YAML frontmatter 与 markdown 正文，并生成以技能名为键的字典。随后它模拟一个 agent 循环，通过名称调用 `release-notes-writer`。

可关注点：

- 使用最小化的 stdlib 解析器解析 YAML frontmatter（无 `pyyaml` 依赖）。
- 技能正文逐字保存；agent 在调用时将其预置到 system prompt 中。
- 通过 `read_subresource` 函数演示渐进式披露：按需拉取技能引用的文件。

## 发布打包

本课产出 `outputs/skill-agent-bundle.md`。给定一个工作流，技能会生成组合的 SKILL.md + AGENTS.md + MCP-server-blueprint 包，能在多种 agent 间通用。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个技能，确认加载器能发现它。

2. 为本课程仓库编写一个 AGENTS.md。包含测试命令、风格约定，以及 Phase 13 的心智模型。

3. 将你团队内部文档中的一个多步骤工作流移植为 SKILL.md。验证它能在 Claude Code 中加载。

4. 手动把该技能翻译为 Cursor 和 Codex 的原生规则格式。统计格式之间的差异 —— 这就是 SkillKit 自动化的翻译面。

5. 阅读 Anthropic Agent Skills 的博客文章。找出本课加载器未覆盖的 Claude Agent SDK 的一个功能。（提示：agent 子调用。）

## 术语表

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SKILL.md | "The skill file" | YAML frontmatter 加上 markdown 正文，由 agent 运行时加载 |
| AGENTS.md | "Repo-root agent context" | 仓库级的约定文件，会在会话开始时读取 |
| Progressive disclosure | "Lazy-load sub-resources" | 技能正文引用的子文件仅在需要时拉取 |
| Frontmatter | "YAML block at top" | 包含 metadata（name, description）的 `---` 分隔块 |
| Claude Agent SDK | "Anthropic's skill runtime" | `@anthropic-ai/claude-agent-sdk`，负责加载技能并进行路由 |
| OpenAI Apps SDK | "MCP + widget meta" | 基于 MCP 的 OpenAI 开发表面，附带 ChatGPT UI 钩子 |
| Skill discovery | "Filesystem scan" | 在已知目录中查找 SKILL.md，并按 name 建立索引 |
| Cross-agent portability | "One skill many agents" | 通过 SkillKit 风格的工具将一个 SKILL.md 翻译成 32+ 个 agent 的格式 |
| Agent Skill | "Portable know-how" | 可复用的任务模板，独立于 MCP 的工具概念 |
| Apps SDK | "MCP plus ChatGPT UI" | 将 Connectors 和 Custom GPTs 在 MCP 上统一化 |

## 延伸阅读

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发平台
- [agents.md](https://agents.md/) — AGENTS.md 格式与采纳列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例