# Capstone — 构建完整的工具生态系统

> Phase 13 教会了每一块拼图。本 Capstone 将它们接线成一个生产就绪的系统：带有工具 + 资源 + 提示词 + 任务 + UI 的 MCP 服务器，边缘处的 OAuth 2.1，RBAC 网关，多服务器客户端，A2A 子代理调用，发送到采集器的 OTel 跟踪，在 CI 中的工具投毒检测，以及一套 AGENTS.md + SKILL.md。到最后你能为每一个架构选择做答辩。

**Type:** 构建  
**Languages:** Python (stdlib, 端到端生态系统整合)  
**Prerequisites:** Phase 13 · 01 through 21  
**Time:** ~120 分钟

## 学习目标

- 组合一个暴露工具、资源、提示词和带有 `ui://` 应用的任务的 MCP 服务器。
- 在服务器前端放置一个执行 RBAC 和固定哈希验证的 OAuth 2.1 网关。
- 编写一个多服务器客户端，使用带有 GenAI 属性的 OTel 进行端到端追踪。
- 将部分工作负载委派给 A2A 子代理；验证不透明性（opacity）得以保留。
- 用 AGENTS.md + SKILL.md 打包整个堆栈，使其他代理能够驱动它。

## 问题说明

交付“研究并报告”系统：

- 用户请求：“总结 2026 年在 arXiv 上被引用最多的三篇关于代理协议的论文。”
- 系统：通过 MCP 搜索 arXiv；将论文摘要任务委托给通过 A2A 的专门写手代理；聚合结果；将交互式报告渲染为 MCP Apps 的 `ui://` 资源；将每一步都记录到 OTel。

Phase 13 的所有原语都会出现。这不是玩具 —— 到 2026 年，Anthropic（Claude Research 产品）、OpenAI（带 Apps SDK 的 GPTs）和第三方交付的生产研究助手系统都具有完全相同的形态。

## 概念

### 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (纯)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (长任务)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A 调用: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### 跟踪层级

```
agent.invoke_agent
 ├── llm.chat (启动)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── 任务状态转换（内部不透明）
 ├── mcp.call -> tools/call generate_report (任务增强)
 │    └── tasks/status 轮询
 │    └── tasks/result（完成，返回 ui:// 资源）
 └── llm.chat (最终合成)
```

一个 trace id。每个 span 都包含正确的 `gen_ai.*` 属性。

### 安全态势

- OAuth 2.1 + PKCE，使用资源指示器将受众（audience）钉定到网关。
- 网关持有上游凭据；用户永远看不到它们。
- RBAC：`alice` 拥有 `research:read`、`research:write`，可以调用所有工具。`bob` 只有 `research:read`，不能调用 `generate_report`。
- 固定描述清单：剔除了任何工具哈希发生变化的服务器。
- “二重规则”审计（Rule of Two）：没有工具同时结合不受信任的输入、敏感数据和具有重大后果的动作。

### 呈现

最终的 `generate_report` 任务返回内容块以及一个 `ui://report/current` 资源。客户端宿主（如 Claude Desktop 等）在沙箱 iframe 中渲染交互式仪表盘。仪表盘包含已排序的论文列表、引用计数，以及一个按钮，当用户点击某篇论文时会调用 `host.callTool('summarize_paper', {arxiv_id})`。

### 打包

整个项目按以下结构交付：

```
research-system/
  AGENTS.md                     # 项目约定
  skills/
    run-research/
      SKILL.md                  # 顶层工作流
  servers/
    research-mcp/               # MCP 服务器
      pyproject.toml
      src/
  agents/
    writer/                     # A2A 写手代理
  gateway/
    config.yaml                 # RBAC + 固定清单（pinned manifest）
```

用户用 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 的用户都可以通过调用 `run-research` skill 来驱动该系统。

### 每个 Phase 13 课程的贡献

| Lesson | 本 Capstone 使用到的内容 |
|--------|------------------------|
| 01-05 | 工具接口、提供者可移植性、并行调用、schema、lint |
| 06-10 | MCP 原语、服务器、客户端、传输、资源 + 提示词 |
| 11-14 | 采样、roots + 引导（elicitation）、异步任务、`ui://` 应用 |
| 15-17 | 工具投毒、OAuth 2.1、网关 + 注册表 |
| 18 | A2A 子代理委派 |
| 19 | OTel GenAI 跟踪 |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 使用说明

`code/main.py` 将前几课的模式拼接成一个可运行的演示。全部使用 stdlib，全部进程内实现，便于端到端阅读。它运行研究与报告场景的完整流程：与网关握手、模拟 OAuth 2.1、合并 tools/list、把 `generate_report` 当作任务执行、对 A2A 写手的调用、返回 `ui://` 资源、并发出 OTel spans。

需要关注的点：

- 每一跳共用一个 trace id。
- 网关策略会阻止第二个用户进行写操作。
- 任务生命周期走向 working → completed，并返回文本和 `ui://` 内容。
- A2A 调用的内部状态对协调器保持不透明。
- AGENTS.md 和 SKILL.md 是其他代理复现该工作流所需的唯一文件。

## 交付物

本课程会生成 `outputs/skill-ecosystem-blueprint.md`。针对一个产品需求（研究、摘要、自动化），该 skill 会产出完整架构：哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测、以及如何打包。

## 练习

1. 运行 `code/main.py`。注意单一 trace id 以及 spans 的嵌套关系。统计演示触及了多少个 Phase 13 的原语。

2. 扩展示例：添加第二个后端 MCP 服务器（例如 `bibliography`），并确认网关将它的工具合并到相同的命名空间中。

3. 将假 A2A 写手代理替换为在子进程中运行的真实代理。使用第 19 课的测试工具（harness）。

4. 在路由网关中添加 PII 删减步骤（位于协调器和 LLM 之间）。确认用户查询中的电子邮件地址被清洗掉。

5. 为将维护该系统的同事写一份 AGENTS.md。阅读时间应少于五分钟，并给出驱动该 Capstone 在 Cursor 或 Codex 中运行所需的一切信息。

## 关键术语

| 术语 | 人们如何说 | 实际含义 |
|------|-----------|----------|
| Capstone | “Phase-13 整合演示” | 使用每个原语的端到端系统 |
| Research and report | “该场景” | 搜索、摘要、呈现 模式 |
| Ecosystem | “所有组件一起” | 服务器 + 客户端 + 网关 + 子代理 + 遥测 + 打包 |
| Trace hierarchy | “单一 trace id” | 每一跳的 span 共享该 trace；通过 span id 建立父子关系 |
| Gateway-issued token | “传递式认证” | 客户端只看到网关的 token；网关持有上游凭据 |
| Merged namespace | “平展的工具列表” | 网关处的多服务器合并，冲突时加前缀 |
| Opacity boundary | “A2A 调用隐藏内部” | 子代理的推理对协调器不可见 |
| Three-layer stack | “AGENTS.md + SKILL.md + MCP” | 项目上下文 + 工作流 + 工具 |
| Defense-in-depth | “多层安全” | 固定哈希、OAuth、RBAC、二重规则、审计日志 |
| Spec compliance matrix | “我们交付的规范要求” | 将交付物映射到 2025-11-25 要求的检查表 |

## 延伸阅读

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 汇总参考  
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议的发展方向  
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考  
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范的跟踪约定  
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — 生产代理运行时模式