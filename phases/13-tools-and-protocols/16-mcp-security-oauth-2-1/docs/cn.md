# MCP Security II — OAuth 2.1, Resource Indicators, Incremental Scopes

> 远程 MCP 服务器需要授权（authorization），不仅仅是认证（authentication）。2025-11-25 规范对齐到 OAuth 2.1 + PKCE + resource indicators (RFC 8707) + protected-resource metadata (RFC 9728)。SEP-835 添加了基于 403 WWW-Authenticate 的增量 scope 同意与权限升级（step-up authorization）。本课将以状态机实现该权限升级流程，从而可以查看每一步跳转。

**Type:** 构建
**Languages:** Python（stdlib，OAuth 状态机模拟器）
**Prerequisites:** Phase 13 · 09（传输），Phase 13 · 15（安全 I）
**Time:** ~75 分钟

## 学习目标

- 区分资源服务器和授权服务器的职责。
- 理解带 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和 protected-resource metadata（RFC 9728）防止 confused-deputy 攻击。
- 实现权限升级：服务器返回 403 并在 WWW-Authenticate 中要求更高的 scope；客户端重新提示用户同意并重试。

## 问题背景

早期 MCP（2025 年之前）部署的远程服务器使用临时的 API key，甚至没有任何鉴权。2025-11-25 的规范用一个完整的 OAuth 2.1 配置来弥补这一缺陷。

三个现实需求：

- **普通远程服务器。** 用户安装一个能访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。OAuth 2.1 + PKCE 是合适的方案。
- **权限升级（Scope escalation）。** 一个笔记服务器最初授予 `notes:read`，之后为了某个操作需要 `notes:write`。不必重走整个流程，step-up（SEP-835）请求附加的 scope。
- **防止 confused deputy。** 客户端持有面向 Server A 的令牌。恶意的 Server A 尝试将该令牌用于 Server B。resource indicators（RFC 8707）将令牌固定到预定的受众，从而阻止这种滥用。

OAuth 2.1 本身并不是新的。新的在于 MCP 的配置剖面：指定了必需的流程（仅授权码 + PKCE；默认禁用 implicit、client credentials），每次令牌请求都强制要求 resource indicators，并发布 protected-resource metadata 以便客户端知道应当请求哪里。

## 概念

### 角色

- **Client（客户端）。** MCP 客户端（如 Claude Desktop、Cursor 等）。
- **Resource server（资源服务器）。** MCP 服务器（笔记、GitHub、Postgres 等）。
- **Authorization server（授权服务器）。** 签发令牌的实体。可能与资源服务器为同一服务，也可能是独立的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 的配置中，资源服务器和授权服务器可以位于同一主机，但应当通过不同的 URL 加以区分。

### 授权码 + PKCE

流程：

1. 客户端生成 `code_verifier`（随机）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端向 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...` 发起 POST。
5. 授权服务器校验 verifier 的哈希是否与存储的 challenge 匹配，并签发访问令牌。
6. 客户端使用该令牌：在每次对资源服务器的请求中添加 `Authorization: Bearer ...`。

PKCE 防止授权码被拦截后被盗用。resource indicators 防止该令牌在其他地方被使用。

### Protected-resource metadata（RFC 9728）

资源服务器发布一个 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端通过资源服务器发现授权服务器。这减少了配置——客户端只需要知道资源的 URL。

### Resource indicators（RFC 8707）

在令牌请求中的 `resource` 参数将令牌固定到预期的受众。下发的令牌包含 `aud: "https://notes.example.com"`。另一个 MCP 服务器若收到该令牌应校验 `aud` 并拒绝不匹配的令牌。

### Scope 模型

Scopes 是以空格分隔的字符串。常见的 MCP 约定：

- `notes:read`, `notes:write`, `notes:delete`
- `admin:*` 用于管理权限（谨慎使用）
- `profile:read` 用于身份信息

范围选择应遵循最小权限原则：仅请求现在需要的权限，需要更多权限时再进行 step-up。

### 权限升级（Step-up authorization，SEP-835）

用户授予了 `notes:read`。后来他们要求代理删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端检测到 `insufficient_scope` 错误，提示用户同意附加的 scope，执行一个小型的 OAuth 流以获取新权限，然后用新令牌重试请求。

### 令牌受众校验

每次请求：服务器检查 `token.aud == self.resource_url`。不匹配则返回 401。这阻止了跨服务器的令牌重用。

### 短生命周期令牌与旋转

访问令牌应为短时有效（默认 1 小时）。刷新令牌在每次刷新时轮换。客户端在后台处理静默刷新。

### 禁止令牌透传

采样服务器（Phase 13 · 11）不得将客户端令牌透传给其他服务。采样请求是边界。

### 防止 confused deputy

令牌绑定到 `aud`。客户端绑定到 `client_id`。每个请求都要针对两者进行验证。规范明确禁止了在 pre-MCP 远程工具生态中常见的“传递令牌”模式。

### 客户端 ID 发现

每个 MCP 客户端在一个固定的 URL 发布其元数据文档。授权服务器可以抓取该客户端元数据来发现 redirect URI 和联系信息。这消除了手工的客户端注册。

### 网关与 OAuth

Phase 13 · 17 展示了企业网关如何处理 OAuth：网关持有上游服务器的凭证，客户端得到的令牌由网关签发，上游令牌永远不会离开网关。这改变了信任模型——用户只需一次与网关认证；网关处理对 N 个服务器的授权。

## 使用方法

`code/main.py` 以状态机模拟完整的 OAuth 2.1 权限升级流程。它实现了：

- PKCE code-verifier / challenge 生成。
- 带 resource indicator 的授权码流程。
- Protected-resource metadata 端点。
- 带受众校验的令牌验证。
- 在 `insufficient_scope` 情况下的权限升级。

本课没有运行 HTTP 服务器；状态机在内存中运行，便于追踪每一步跳转。Phase 13 · 17 的网关课程把它接到实际传输层。

## 交付物

本课产出 `outputs/skill-oauth-scope-planner.md`。给定一个带工具的远程 MCP 服务器，该 skill 设计 scope 集合、固定规则（pinning rules）和权限升级策略。

## 练习

1. 运行 `code/main.py`。跟踪两个 scope 的权限升级流程。注意在权限升级时哪些跳转会重复。

2. 添加刷新令牌轮换：每次刷新都发放一个新的刷新令牌并使旧令牌失效。模拟被盗的刷新令牌在轮换后被使用并确认其失败。

3. 使用 stdlib 的 http.server 实现 protected-resource metadata 端点作为真实的 HTTP 响应。镜像第 09 课中的 /mcp 端点。

4. 为一个 GitHub MCP 服务器设计 scope 层级：读取仓库（read repo）、创建 PR（write PR）、批准 PR（approve PR）、合并 PR（merge PR）、管理员（admin）。在每个层级之间使用 step-up。

5. 阅读 RFC 8707 和 RFC 9728。识别在 9728 中 MCP 与 RFC 示例不同使用的一个字段。（提示：与 `scopes_supported` 有关。）

## 术语表

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OAuth 2.1 | "Modern OAuth" | 整合后的 RFC，强制使用 PKCE 并禁止 implicit 流 |
| PKCE | "Proof-of-possession" | code verifier + challenge，用于防止授权码拦截 |
| Resource indicator | "Token audience" | RFC 8707 中的 `resource` 参数，将令牌绑定到单一服务器 |
| Protected-resource metadata | "Discovery doc" | RFC 9728 的 `.well-known/oauth-protected-resource` |
| Step-up authorization | "Incremental consent" | SEP-835：按需增加 scope 的流程 |
| `insufficient_scope` | "403 with WWW-Authenticate" | 服务器提示重新同意以获取更大 scope |
| Confused deputy | "Token reuse across services" | 受信任持有者不当地转发令牌的攻击 |
| Short-lived token | "Access token TTL" | 短时有效的 Bearer；使用刷新令牌续期 |
| Scope hierarchy | "Least privilege stack" | 分级的最小权限集合，层级间使用 step-up |
| Client ID metadata | "Client discovery doc" | 客户端在某个 URL 发布的 OAuth 元数据文档 |

## 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — MCP OAuth 配置规范（权威）
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 更改的讲解
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定的 RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档 RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实践性的权限升级流程演练