# MCP Gateways and Registries — Enterprise Control Planes

> 企业不能允许每个开发者随意安装任意 MCP 服务器。网关集中化处理认证、RBAC、审计、速率限制、缓存和工具投毒检测，然后将合并后的工具表面作为单一 MCP 端点暴露出去。官方 MCP 注册表（Anthropic + GitHub + PulseMCP + Microsoft，命名空间验证）是规范的上游。此课程说明网关的定位，演示一个最小实现，并概览 2026 年的厂商生态。

**Type:** 学习  
**Languages:** Python（标准库，最小网关）  
**Prerequisites:** Phase 13 · 15（tool poisoning），Phase 13 · 16（OAuth 2.1）  
**Time:** ~45 分钟

## 学习目标

- 说明 MCP 网关的位置（在 MCP 客户端和多个后端 MCP 服务器之间）。
- 实现五项网关职责：认证、RBAC、审计、速率限制、策略。
- 在网关层强制执行已固定工具哈希清单（pinned-tool-hash manifest）。
- 区分官方 MCP 注册表与元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题背景

某家财富 500 强企业有 30 个批准的 MCP 服务器、5000 名开发者、合规与审计需求，以及希望集中化策略的安全团队。允许每个开发者在其 IDE 中安装任意服务器并不可行。

网关模式：

1. 网关作为单一可流式（Streamable）的 HTTP 端点运行，开发者连接到该端点。
2. 网关持有每个后端 MCP 服务器的凭证。
3. 每个开发者请求通过网关自身的 OAuth 进行认证和作用域约束。
4. 网关将调用路由到后端服务器，并应用策略。
5. 所有调用都记录以便审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway — 在 2025–2026 年均发布了网关或网关功能。

同时，官方 MCP 注册表作为规范上游启动：经策划、命名空间验证、采用反向 DNS 命名的服务器，网关可以从中拉取。元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）则汇聚了来自多个来源的服务器。

## 概念

### 五项网关职责

1. **Auth.** 使用 OAuth 2.1 标识开发者；映射到用户角色。  
2. **RBAC.** 基于用户的策略：允许访问哪些服务器、哪些工具、哪些作用域。  
3. **Audit.** 每次调用都记录谁、做了什么、何时、结果如何。  
4. **Rate limit.** 对每个用户 / 每个工具 / 每个服务器设置上限以防滥用。  
5. **Policy.** 拒绝被投毒的描述，强制执行 Rule of Two，脱敏个人信息（PII）。

### 网关作为单一端点

对开发者而言，网关看起来像一个 MCP 服务器。内部它会路由到 N 个后端。会话 ID（Phase 13 · 09）在边界处被改写。

### 凭证保管

开发者永远看不到后端令牌。网关持有这些令牌（或代理到负责的身份提供方）。一个在网关上拥有 `notes:read` 权限的开发者可以通过网关使用网关自身的后端凭证间接访问 notes MCP 服务器——但仅在将该传递访问与策略绑定的情况下才允许。

### 网关层的工具哈希固定

网关维护一份已批准工具描述的清单（SHA256 哈希）。在发现阶段，它会获取每个后端的 `tools/list`，将哈希与清单比对，并移除任何描述发生变更的工具。这是 Phase 13 · 15 中的“拔地毯（rug-pull）”防御在中心化处的应用。

### 以代码化形式管理策略

高级网关使用 OPA/Rego、Kyverno 或 Styra 来表达策略。像“用户 `alice` 仅能在 org `acme` 的仓库上调用 `github.open_pr`”这样的规则以声明式编码。简单网关使用手写的 Python，两者都是有效的形态。

### 会话感知路由

当用户会话包含多个服务器的混合时，网关会进行复用：开发者的单个 MCP 会话包含 N 个后端会话，每个服务器一个。任何后端的通知都会通过网关路由回开发者会话。

### 命名空间合并

网关会合并来自所有后端的工具命名空间，冲突时通常添加前缀。例如 `github.open_pr`、`notes.search`。这使路由明确无歧义。

### 注册表

- **官方 MCP 注册表** (`registry.modelcontextprotocol.io`)。由 Anthropic、GitHub、PulseMCP、Microsoft 联合管理启动。命名空间验证（反向 DNS：`io.github.user/server`）。预先筛选以保证基本质量。  
- **Glama。** 以搜索为中心的元注册表，聚合了多个来源。  
- **MCPMarket。** 偏向商业目录，包含厂商列表。  
- **MCP.so。** 社区目录；开放提交。  
- **Smithery。** 类包管理器的安装流程。  
- **LobeHub。** 在其 LobeChat 应用中集成的注册表 UI。

企业网关默认从官方注册表拉取，允许管理员从元注册表添加经策划的条目，并拒绝任何未被固定的条目。

### 反向 DNS 命名

官方注册表要求公共服务器采用反向 DNS 名称：`io.github.alice/notes`。命名空间可防止抢注并使信任委派更清晰。

### 厂商调研，2026 年 4 月

| Vendor | Strength |
|--------|----------|
| Cloudflare MCP Portals | 边缘托管；集成 OAuth；提供免费层 |
| Kong AI Gateway | 原生 K8s 支持；细粒度策略；将日志导出到 OpenTelemetry |
| IBM ContextForge | 企业身份与访问管理；合规性；审计导出 |
| TrueFoundry | 偏向 DevOps；以指标为先 |
| MintMCP | 面向开发者平台 |
| Envoy AI Gateway | 开源；可自定义过滤器 |

Phase 17（生产基础设施）将更深入地探讨网关运维。

## 使用示例

`code/main.py` 提供了一个约 150 行的最小网关：通过伪造的 Bearer 令牌对用户进行认证，维护每用户的 RBAC 策略，将请求路由到两个后端 MCP 服务器，将每次调用写入审计日志，强制速率限制，并拒绝任何其描述哈希与已固定清单不符的后端工具。

值得关注的点：

- `RBAC` 字典以 `user_id` 为键，值为允许的 `server_tool` 条目。  
- `AUDIT_LOG` 是一个追加式（append-only）的事件列表。  
- 速率限制对每个用户使用令牌桶（token bucket）。  
- 固定清单是一个 `server::tool -> hash` 的字典。

## 交付物

本课产出 `outputs/skill-gateway-bootstrap.md`。给定企业 MCP 计划（用户、后端、合规），该技能输出一个网关配置规范。

## 练习

1. 运行 `code/main.py`。以一个被允许的用户发起调用；然后以一个被拒绝的用户发起；再做一次超过速率限制的突发请求。验证这三种流程。  
2. 添加一条策略，在返回给客户端之前从结果中脱敏 PII。使用一个简单的正则替换 SSN 形状的字符串；注意该方法的缺口（电子邮件、电话号码）。  
3. 扩展审计日志以发出 OpenTelemetry GenAI spans。Phase 13 · 20 覆盖了精确属性。  
4. 为一个由 50 名开发者和五个后端（notes、github、postgres、jira、slack）组成的团队设计 RBAC 策略。谁获得只读？谁获得写权限？  
5. 从头到尾阅读 Cloudflare 的企业 MCP 文章。找出 Cloudflare 提供的、该 stdlib 网关尚未实现的一项功能。

## 关键术语

| 术语 | 常说法 | 实际含义 |
|------|--------|----------|
| Gateway | “MCP 代理” | 在客户端和后端之间的集中化服务器（网关） |
| Credential vaulting | “后端令牌保留在服务端” | 开发者永远看不到上游令牌 |
| Session-aware routing | “多后端会话” | 网关为每个开发者会话复用 N 个后端会话 |
| Tool-hash pinning | “已批准清单” | 每个已批准工具描述的 SHA256；在中心化处阻止拔地毯 |
| RBAC | “基于用户的策略” | 针对工具和服务器的基于角色的访问控制 |
| Policy-as-code | “声明式规则” | 在网关强制执行的 OPA/Rego、Kyverno、Styra 策略 |
| Audit log | “谁、什么、何时” | 用于合规的追加式事件日志 |
| Rate limit | “每用户令牌桶” | 每分钟上限以防滥用 |
| Official MCP Registry | “规范上游” | `registry.modelcontextprotocol.io`，命名空间已验证 |
| Reverse-DNS naming | “注册表命名空间” | `io.github.user/server` 命名约定 |

## 延伸阅读

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — 规范上游，命名空间已验证  
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — 使用 OAuth 与策略的网关模式  
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — 开源参考网关  
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 功能对比文章  
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — IBM 的企业网关