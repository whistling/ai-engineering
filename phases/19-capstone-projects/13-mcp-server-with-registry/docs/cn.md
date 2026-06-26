# Capstone 13 — MCP Server with Registry and Governance

> Model Context Protocol 在 2026 年不再是未来概念，而成为默认的工具使用规范。Anthropic、OpenAI、Google 以及所有主要 IDE 都附带 MCP 客户端。Pinterest 发布了其内部的 MCP 服务器生态。AAIF Registry 在 `.well-known` 中将能力元数据形式化。AWS ECS 发布了参考的无状态部署。Block 的 goose-agent 将相同协议嵌入到托管助理中。2026 年的生产形态是：StreamableHTTP 传输、OAuth 2.1 作用域、OPA 策略门控，以及一个让平台团队发现、验证并启用服务器的注册表。将其端到端构建出来。

**Type:** 实战项目  
**Languages:** Python（服务器，使用 FastMCP）或 TypeScript（@modelcontextprotocol/sdk），Go（注册表服务）  
**Prerequisites:** Phase 11（LLM 工程）、Phase 13（工具与 MCP）、Phase 14（Agent）、Phase 17（基础设施）、Phase 18（安全）  
**Phases exercised:** P11 · P13 · P14 · P17 · P18  
**Time:** 25 小时

## 问题

MCP 已成为工具使用的通用语言。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 以及所有托管 Agent 现在都消费 MCP 服务器。生产环境的挑战不是编写服务器（FastMCP 使这很容易），而是在企业要求下进行大规模部署：每租户的 OAuth 作用域、对破坏性工具的 OPA 策略、StreamableHTTP 的无状态水平扩展、用于发现的注册表、以及每次工具调用的审计日志。Pinterest 的内部 MCP 生态和 AAIF Registry 规范定义了 2026 年的标准。

你将构建一个暴露 10 个内部工具的 MCP 服务器（Postgres 只读、S3 列表、Jira、Linear、Datadog 等），一个供平台发现的注册表 UI，以及用于破坏性工具的人为审批门控。负载测试要演示 StreamableHTTP 的横向扩展。审计链满足企业安全审查要求。

## 概念

MCP 2026 修订版规定将 StreamableHTTP 作为默认传输。与早期的 stdio 与 SSE 形态不同，StreamableHTTP 默认是无状态的：单个 HTTP 端点接受 JSON-RPC 请求、流式返回响应，并支持用于通知的长连接。无状态意味着可以在负载均衡器后面进行水平扩展。

授权采用 OAuth 2.1，按工具划分作用域。Token 会携带类似 `jira:read`、`s3:list`、`postgres:query:readonly` 的作用域。MCP 服务器在调用工具时检查作用域，而不仅仅是在会话开始时检查。对于高风险工具，服务器会拒绝在最近 N 分钟内未通过 `approved:by:human` 提升的任何调用——该提升通过 Slack 审核卡发起。

注册表是一个独立服务。每个 MCP 服务器都公开一个 `.well-known/mcp-capabilities` 文档，包含其工具清单、传输 URL、认证需求。注册表会轮询、验证并建立索引。平台团队使用注册表 UI 查看可用工具、所需作用域以及归属团队。

## 架构

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## 技术栈

- 服务器框架：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）  
- 传输：StreamableHTTP over HTTPS（无状态）  
- 认证：OAuth 2.1，使用 SPIFFE / SPIRE 的工作负载身份（workload identity）  
- 策略：每个工具的 OPA / Rego 规则；每次请求调用决策服务  
- 注册表：自托管，消费 `.well-known/mcp-capabilities` 清单  
- 人为审批：用于破坏性工具的 Slack 交互消息  
- 部署：AWS ECS Fargate 或 Fly.io，每租户一个服务器或共享服务器配合租户范围  
- 审计：每租户的结构化 JSONL，记录每次调用的血缘信息

## 构建

1. 工具表面（Tool surface）。暴露 10 个内部工具：Postgres 只读查询、S3 列表对象、Jira 搜索/获取、Linear 搜索/获取、Datadog 指标查询、PagerDuty 值班查询、GitHub 只读、Notion 搜索、Slack 搜索、Salesforce 只读。每个工具都有类型化的 schema 和一个作用域标签。

2. FastMCP 服务器。挂载这些工具。配置 StreamableHTTP 传输。添加用于 OAuth token introspection 和作用域强制的中间件。

3. OPA 策略。为每个工具编写 Rego 策略：哪些作用域允许调用、适用的 PII 脱敏规则、有效载荷大小上限。每次工具调用时调用决策服务。

4. 注册表服务。一个独立的 Go 或 TS 服务，轮询已注册服务器的 `.well-known/mcp-capabilities`，使用 JSON Schema 验证，并提供列表 / 搜索 / 验证 / 启用-禁用 的 UI。

5. 能力清单（Capability manifest）。每个服务器公开 `.well-known/mcp-capabilities`，包含：工具列表、认证要求、传输 URL、归属团队、SLO。

6. 破坏性工具分离。会修改状态的工具（如 Jira create、Linear create、Postgres 写入）部署在第二个 MCP 服务器上，采用更严格的认证流程：token 必须带有在 15 分钟内通过 Slack 卡片提升的 `approved:by:human` 作用域。

7. 审计日志。每租户的追加式 JSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前通过 Presidio 进行 PII 脱敏。

8. 负载测试。对 StreamableHTTP 进行 100 个并发客户端测试。通过新增第二个副本演示横向扩展；展示负载均衡器在无需会话粘滞（session stickiness）的情况下重新分配流量。

9. 合规性测试。针对两个服务器运行官方 MCP 合规套件。通过所有必需部分。

## 使用示例

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 交付

`outputs/skill-mcp-server.md` 描述了交付物。生产级的 MCP 服务器 + 注册表 + 审计层，适用于内部工具，具备 OAuth 2.1 作用域和 OPA 门控。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Spec conformance | StreamableHTTP + capability manifest 通过 MCP 合规性测试 |
| 20 | Security | 作用域强制、每个工具的 OPA 覆盖、密钥与凭据管理良好 |
| 20 | Observability | 每次工具调用的审计日志并进行 PII 脱敏 |
| 20 | Scale | 100 客户端负载测试的横向扩展演示 |
| 15 | Registry UX | 发现 / 验证 / 启用-禁用 的工作流 |
| **100** | | |

## 练习

1. 添加一个新工具（Confluence 搜索）。在不修改核心服务器的情况下，通过注册表验证流发布它。

2. 编写一个 OPA 策略，当 Postgres 查询结果包含名为 `email`、`ssn` 或 `phone` 的列时进行脱敏。在探测查询中演练该策略。

3. 在本地对比 StreamableHTTP 与 stdio 的延迟基准。报告每次调用的 p50 / p95。

4. 实现按租户配额：每租户每个工具每分钟最多 N 次调用。通过第二条 OPA 规则强制执行。

5. 从 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 运行 MCP 合规套件并修复所有失败项。

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| StreamableHTTP | "2026 MCP transport" | 无状态的 HTTP + 流式；用于替代网络服务器的 SSE + stdio |
| Capability manifest | "Well-known doc" | `.well-known/mcp-capabilities`，包含工具清单、认证、传输 URL |
| OPA / Rego | "Policy engine" | 用于根据外部规则授权工具调用的 Open Policy Agent |
| Scope elevation | "Approved-by-human" | 通过 Slack 审批授予的短期作用域，用于破坏性工具 |
| Registry | "Tool discovery" | 从能力清单中索引 MCP 服务器的服务 |
| Workload identity | "SPIFFE / SPIRE" | 用于 OAuth token 签发的服务级加密身份（工作负载身份） |
| Conformance suite | "Spec tests" | 针对 StreamableHTTP 与工具清单正确性的官方 MCP 测试套件 |

## 延伸阅读

- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据、注册表  
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) — 2026 年的注册表规范  
- [AWS ECS reference deployment](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — 参考的生产部署  
- [Pinterest internal MCP ecosystem](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — 参考的内部部署  
- [Block `goose` MCP usage](https://block.github.io/goose/) — 参考的 Agent 消费模式  
- [FastMCP](https://github.com/jlowin/fastmcp) — Python 服务器框架  
- [Open Policy Agent](https://www.openpolicyagent.org/) — 策略引擎参考  
- [SPIFFE / SPIRE](https://spiffe.io) — 工作负载身份参考