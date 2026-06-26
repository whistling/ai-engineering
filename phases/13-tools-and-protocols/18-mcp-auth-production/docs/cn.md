# MCP Auth in Production — Enrollment, JWKS Refresh, Audience-Pinned Tokens

> 第16课在内存中搭起了 OAuth 2.1 状态机。到 2026 年，你交付给真实组织的每个 MCP 服务器都运行在生产级别的认证后面：可扩展到无界客户端群体的客户端注册（优先使用 Client ID Metadata Documents，向后兼容地回退到动态客户端注册），授权服务器元数据发现（RFC 8414 *或* OpenID Connect Discovery），不会在凌晨 3 点打断令牌校验的 JWKS 缓存刷新，以及拒绝跨资源重放的受众绑定令牌。此课用三个角色建模完整表面——授权服务器、资源服务器（MCP 服务器）和客户端——以便你能追踪从发现到已验证工具调用的每一步。

> **Spec note (2025-11-25):** 2025 年 11 月 25 日的 MCP 授权规范将动态客户端注册从 `SHOULD` 降级为 `MAY`，并将 **Client ID Metadata Documents (CIMD)** 设为推荐的默认入网机制。本课按规范的优先顺序教授两者，并在演示代码中保留 DCR 以便完整演示，因为它在单进程中是自包含的。

**Type:** 构建  
**Languages:** Python（标准库）  
**Prerequisites:** 阶段 13 · 16（OAuth 2.1 状态机），阶段 13 · 17（网关）  
**Time:** ~90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证契约。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端在无需管理员干预的情况下完成入网。
- 将 JWKS 键作为缓存按计划刷新，使签名验证在密钥轮换期间不中断。
- 使用 RFC 8707 资源指示器将令牌钉到单个 MCP 资源，并拒绝混淆代理（confused-deputy）重用。
- 清晰分离三种角色——授权服务器、资源服务器、客户端——使每个角色只执行属于它的检查。
- 阅读 IdP 能力矩阵并在 IdP 不满足 MCP 的认证配置时拒绝部署。

## 问题描述

第16课的模拟器在内存中运行 OAuth 2.1。生产环境有三处操作缺口，内存-only 模拟器看不到。

第一个缺口是入网。真实组织运行数百个 MCP 服务器和数千个 MCP 客户端。运维不会把每个 Cursor 用户都人工注册为 OAuth 客户端。2025-11-25 规范给客户端一个优先顺序来解决此问题：如果已有预注册的 `client_id` 则使用它，否则使用 **Client ID Metadata Document**（客户端用其控制的 HTTPS URL 标识自己，授权服务器 *拉取* 元数据），否则回退到 **RFC 7591 动态客户端注册**（客户端向 `/register` 发起 `POST`，即时获得 `client_id`），否则提示用户。CIMD 是推荐默认，因为它完全消除了每台服务器的注册，同时保留基于 DNS 的信任模型；DCR 为向后兼容保留。两者都从授权服务器的元数据中发现入口点：CIMD 用 `client_id_metadata_document_supported`，DCR 用 `registration_endpoint`。

第二个缺口是密钥轮换。JWT 验证依赖于授权服务器发布的签名密钥（JSON Web Key Set，JWKS）。授权服务器按计划轮换这些密钥（通常按小时，有时在事件响应下更快）。如果 MCP 服务器只在启动时获取一次 JWKS，则在轮换窗口之后所有请求都会失败直到重启。生产环境将 JWKS 作为缓存值，并运行刷新任务在旧密钥过期前覆盖缓存，同时在缓存未命中时回退到即时抓取，以处理签名于比缓存更新更晚的密钥的令牌到达的情况。

第三个缺口是受众绑定。第16课引入了 RFC 8707 的资源指示器。在生产中，该指示器成为每次请求上的严格声明检查。MCP 服务器在每次调用时将 `token.aud` 与其自己的规范资源 URL 比较，若不匹配则以 HTTP 401 拒绝。这是防止上游 MCP 服务器（或持有某服务器令牌的恶意客户端）在同一信任网格中对另一台服务器重放该令牌的唯一协议层防线。

本课将每个缺口映射到具体表面的一部分。元数据文档是一个 HTTP 端点。JWKS 缓存刷新是定时任务加键值缓存。JWT 验证是在资源服务器调度任何工具之前运行的例程。保持三种角色分离并让每个角色仅执行其应有的检查：授权服务器签发并轮换密钥，资源服务器缓存并验证，客户端发现并入网。

## 概念

### RFC 8414 — OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

客户端在给定 MCP 资源 URL 时会按链式方式进行发现：RFC 9728 的 `oauth-protected-resource`（资源服务器的文档）标明了 issuer，然后 RFC 8414（本 RFC）列出每个端点。客户端永远不应硬编码授权 URL。

在将某 IdP 视为可信以供 MCP 使用之前，你要验证的契约：

- `code_challenge_methods_supported` 必须包含 `S256`（按 RFC 7636 的 PKCE）。规范明确：如果该字段**缺失**，说明授权服务器不支持 PKCE，客户端**必须**拒绝继续。
- `grant_types_supported` 必须包含 `authorization_code`，并拒绝 `password` 和 `implicit`。
- 至少要公布一种入网路径：`client_id_metadata_document_supported: true`（CIMD，首选）**或** `registration_endpoint`（RFC 7591 DCR，回退）。任一满足契约；不再强制要求 DCR。
- `response_types_supported` 必须恰好为 `["code"]`，符合 OAuth 2.1。

如果缺少 `S256`，MCP 服务器拒绝针对该 IdP 的部署——PKCE 没有降级模式。如果*两者都没有*公布入网路径且你没有预注册的 `client_id`，你也无法入网；这是部署清单的问题，不是代码的问题。

### RFC 9728（回顾）— 受保护资源元数据

第16课已覆盖 RFC 9728。生产环境的差别：该文档是客户端查找由*本* MCP 服务器信任的授权服务器的唯一来源。单个 MCP 服务器可能接受来自多个 IdP 的令牌（例如一个用于员工，一个用于合作伙伴）。RFC 9728 声明了该集合；RFC 8414 记录了每个 IdP 支持的内容。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents（推荐默认）

CIMD 将注册从*推送*改为*拉取*。客户端不再请求授权服务器开出 `client_id`，而是使用其控制的 HTTPS URL 作为 `client_id`。该 URL 解析出一个 JSON 元数据文档；授权服务器在 OAuth 流程中按需获取它。信任以 DNS 为根基：如果服务器运营者信任 `app.example.com`，就信任由 `https://app.example.com/client.json` 提供的客户端。无注册往返、无 `client_id` 命名空间耗尽，也无每服务器需保持同步的服务器端状态。

客户端托管的元数据文档示例：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的 `client_id` 值**必须**等于其被提供的 URL（授权服务器会校验，不匹配会被拒绝）。授权服务器通过在其 RFC 8414 元数据中设置 `client_id_metadata_document_supported: true` 来声明对 CIMD 的支持。

规范在安全方面有两点直白说明：

- **SSRF。** 授权服务器会获取一个攻击者提供的 URL。它必须防护服务器端请求伪造（不要获取到内部/管理员端点）。
- **localhost 冒充。** 仅靠 CIMD 无法阻止本地攻击者声称某合法客户端的元数据 URL 并绑定任意 `localhost` 重定向。授权服务器**必须**在同意页面上清晰显示重定向 URI 的主机名，并且**应该**对仅 `localhost` 的重定向发出警告。

因为 CIMD 无需服务器端状态，所以不像 DCR 那样需要建立注册器。客户端端是只读的：从静态 HTTPS 端点提供你的元数据文档，让授权服务器去拉取它。

### RFC 7591 — 动态客户端注册（回退 / 向后兼容）

DCR 现在是一个 `MAY`，为向后兼容 2025-11-25 之前的部署和尚未支持 CIMD 的 IdP 而保留。没有它（也没有 CIMD 或预注册），每个 MCP 客户端（Cursor、Claude Desktop、自定义 agent）都需要与 IdP 管理员进行带外交换。有了 DCR，客户端可以发起 POST：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器响应包含 `client_id` 和用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

对于运行在用户设备上的 MCP 客户端，`token_endpoint_auth_method: none` 是合适的默认。它们只得到一个 `client_id` —— 没有可被窃取的 `client_secret`。PKCE 提供了公共客户端所需的持有证明。

生产环境的三个陷阱：

- 注册端点必须按源 IP 进行速率限制。否则，敌对方可脚本化生成数百万个假注册并耗尽 `client_id` 命名空间。在注册器处理请求前执行速率限制检查。
- 一些企业 IdP 要求 `software_statement`（一个为客户端担保的签名 JWT）。本课的模拟跳过了它；生产环境要加上验证步骤，拒绝来自非 localhost 重定向 URI 的未签名注册。
- `registration_access_token` 必须以哈希形式存储，而不是明文。该令牌被窃取意味着攻击者可以重写客户端的重定向 URI。

### RFC 8707（回顾）— 资源指示器

第16课已建立其形状。生产规则：每次令牌请求都包含 `resource=<canonical-mcp-url>`，MCP 服务器在每次调用时验证 `token.aud` 是否匹配其自己的资源 URL。规范资源 URI 是服务器的*最具体*标识符：使用小写方案和主机，不含片段，惯例上不带尾部斜杠。路径组件**不会**被规则剥离——当需要标识单独的 MCP 服务器时，规范包含路径。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是有效的规范 URI。为每台服务器选一个并把 `aud` 精确钉到它。（本课的模拟为简洁起见使用裸主机形式的受众，如 `https://notes.example.com`；在同一源下托管多个 MCP 服务器的部署通过路径加以区分。）

### RFC 7636（回顾）— PKCE

PKCE 在 OAuth 2.1 中是强制的。本课的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器会拒绝任何没有 verifier 的令牌请求，或 verifier 与存储的 challenge 不匹配的请求（即哈希不一致）。

### MCP 规范 2025-11-25 的认证配置文件

MCP 规范（2025-11-25）对 MCP 服务器的授权层必须做的工作有精确要求：

- 实现 RFC 9728 的受保护资源元数据，并通过 401 的 `WWW-Authenticate: Bearer resource_metadata="..."` 头部**或** well-known URI `/.well-known/oauth-protected-resource` 提供其位置（SEP-985 使头部为可选并提供 well-known 回退）。元数据的 `authorization_servers` 字段**必须**至少列出一个服务器。
- 在**每次**请求中仅接受通过 `Authorization: Bearer ...` 传递的令牌——绝不在查询字符串中传递，也不要只在会话开始时验证一次。
- 每次请求验证 `aud`、`iss`、`exp` 和必需的作用域。服务器**必须**验证令牌是否专门为其签发（受众）；缺失或不匹配的 `aud` 应被拒绝，绝不作为通配符处理。
- 在 401/403 时返回 `WWW-Authenticate: Bearer`，携带 `error=...`、`resource_metadata="<PRM-URL>"` 参数（该 URL 指向元数据文档，*不是*裸资源）以及在 `insufficient_scope`（403）时返回 `scope="..."`。注意：该参数名为 `resource_metadata`，它是一个发现指针——挑战中没有 `resource` 参数。
- 授权服务器发现接受 **任意一项**：RFC 8414 OAuth 元数据 **或** OpenID Connect Discovery 1.0；客户端必须按优先顺序尝试这两个 well-known 后缀。
- 防止 **mix-up 攻击** 是客户端（不是服务器）的责任：客户端在重定向前记录期望的 `issuer`，并在兑换 code 之前验证授权响应中的 `iss` 参数（RFC 9207）。仅靠 PKCE 无法阻止 mix-up，因为客户端会把 `code_verifier` 交给它被引导到的任意 token 端点。

OAuth 2.1 草案是基质；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 构成表面；MCP 规范是其配置文件。

### IdP 能力矩阵

并非每个 IdP 都支持完整的 MCP 配置文件。下表记录的是截至 2025-11-25 规范时的事实能力声明。这是一个*部署门禁*，而不是建议。

CIMD 在 2025-11-25 规范中引入，底层 OAuth 草案直至 2025 年 10 月才被采纳，因此厂商支持仍在到位——将下面的 “CIMD” 当做“目前状态，请在你的租户中验证”，而不是永久声明。

| IdP 类别 | AS metadata (8414/OIDC) | CIMD | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | 说明 |
|---|---:|---:|---:|---:|---:|---|
| 自托管（Keycloak） | yes | emerging | yes | yes (since 24.x) | yes | 本课中 MCP 配置文件的参考 IdP；端到端完整 DCR 路径，CIMD 跟踪新规范。 |
| 企业 SSO（Microsoft Entra ID） | yes | emerging | yes (premium tiers) | yes | yes | DCR 在不同租户层级的可用性不同；在目标租户中部署前请验证。 |
| 企业 SSO（Okta） | yes | emerging | yes (Okta CIC / Auth0) | yes | yes | DCR 在 Auth0（现为 Okta CIC）可用；经典 Okta 组织需要管理员预注册。 |
| 社交登录 IdP（通用） | varies | no | rarely | rarely | yes | 大多数社交 IdP 将客户端视为静态合作伙伴；没有自助入网。仅用作身份源，需在之上部署支持 MCP 的授权服务器。 |
| 自建/自制 | depends | depends | depends | depends | depends | 如果你自己交付 IdP，务必实现完整配置文件并优先支持 CIMD。跳过 PKCE 或受众绑定会破坏 MCP 的认证契约。 |

部署清单的拒绝规则：如果所选 IdP 在 `code_challenge_methods_supported` 中不包含 `S256`，MCP 服务器拒绝启动——PKCE 没有降级模式。入网是一个较软的门：你需要*一种*可用路径（预注册的 `client_id`、`client_id_metadata_document_supported: true`，或 `registration_endpoint`）。DCR 的缺失本身不再是拒绝触发器，因为 CIMD 或预注册可以覆盖它。

### JWKS 刷新模式（AS 轮换，资源服务器刷新）

将两个动词分开很重要，因为混淆它们会导致真实生产缺陷：

- **Rotate（轮换）** 是*授权服务器*所做的：铸造一个新的签名密钥，在 JWKS 中发布，并在稍后退役旧密钥。资源服务器不参与此事且不能执行它——它不持有 IdP 的私钥。
- **Refresh（刷新）** 是*资源服务器*所做的：重新 `GET` 发布的 JWKS 并写入其缓存。这是资源服务器唯一应做的 JWKS 操作。

生产故障模式是缓存陈旧。用定时刷新任务加键值缓存来解决它。资源服务器运行一个任务（cron、定时器或运行时提供的任何机制），在固定间隔内获取 `<issuer>/.well-known/jwks.json` 并覆盖 `cache[issuer] = {keys, fetched_at}`。验证器从该缓存读取。若 token 的 `kid` 在缓存中缺失，应触发**一次**同步刷新作为回退，然后再重新检查。这同时处理两种情况：定时刷新，以及新密钥在下次计划刷新前签发并导致的钥匙重叠窗口。

回退路径**必须**是重新抓取，绝不能是轮换。如果将缓存未命中路径接到“轮换并铸造”上会有两个问题： (1) 铸造新密钥会产生一个新的 `kid`，但仍不会匹配令牌的 `kid`，因此查找仍会失败；(2) 攻击者用随机 `kid` 喷洒令牌会迫使进行无限的密钥创建——自我造成的 DoS。重新抓取是幂等的，因此一个无效的 `kid` 最多浪费一次请求。

缓存结构示例：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

两个密钥同时存在是稳态。授权服务器通过在退役前引入下一个密钥（例如 `k_2026_04`）来轮换密钥，这样用旧密钥签发的令牌在其过期前仍然有效。缓存保存它们的并集；验证器按 `kid` 选择具体密钥。

### 验证例程

MCP 服务器在分发任何工具前运行验证。`code/main.py` 使用的调用形如：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS 缓存解析签名密钥（在未命中时刷新一次），验证签名，然后检查 `iss` 是否在允许列表中、`aud` 是否匹配本服务器的规范资源、`exp` 以及所需的作用域——在首次失败时返回 `WWW-Authenticate` 挑战。将其保持为资源服务器上的单一例程意味着每个入口点（每次工具调用、每种传输）都经过相同的检查；不存在到达工具而未先验证的路径。

### 受众重放示例（访问令牌权限限制）

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）都向同一个授权服务器注册。服务器 A 被攻破。攻击者取到某用户的 notes 令牌并在服务器 B 上重放该令牌。

服务器 B 的验证器流程：

1. 解码 JWT，按 `kid` 获取 JWKS，验证签名。
2. 在其受保护资源元数据的 `authorization_servers` 中检查 `iss`。（通过——同一 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——令牌的 `aud` 为 `https://notes.example.com`。）
4. 返回 401，带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

受众声明是协议层面上防御该攻击的唯一手段。为性能跳过它是最常见的生产错误；验证器必须在每次请求上运行，而不是只在会话开始时运行。规范把这称为**访问令牌权限限制**：MCP 服务器**必须**拒绝任何未在受众中指名它的令牌。

> **命名说明。** 规范把术语 *confused deputy* 保留给一个相关但不同的问题：MCP 服务器作为 OAuth **代理** 调用第三方 API，使用静态 client id 并在未经逐客户端用户同意的情况下转发令牌。受众绑定可以修复上文的重放；混淆代理问题的修复是逐客户端同意**并且**绝不将入站令牌传递给上游 API（MCP 服务器**必须**为上游获取其自己的独立令牌）。

### Mix-up 攻击（客户端侧防御，服务器无法提供）

客户端在其生命周期中会与许多授权服务器通信。恶意授权服务器可以试图使客户端在授权码被一个诚实的 AS 签发后，把该码兑换到攻击者的 token 端点。受众绑定对此无济于事——攻击发生在尚无令牌之前。防御措施在客户端（RFC 9207）：

1. 在重定向之前，客户端记录从已验证 AS 元数据中获得的期望 `issuer`。
2. 在授权响应时，客户端将返回的 `iss` 参数与记录的 issuer 做比较（简单的字符串比较，不做规范化），然后才去兑换 code。
3. 若不匹配（或当 AS 宣告支持 `authorization_response_iss_parameter_supported` 但 `iss` 缺失）→ 拒绝，并且不要展示 `error` 字段。

仅靠 PKCE 无法阻止 mix-up，因为客户端会把 `code_verifier` 交给任何被引导到的 token 端点。这就是为什么规范要求在每次请求中将 issuer 与 PKCE verifier 和 `state` 一并记录。

### 故障模式

- **陈旧的 JWKS。** 在 AS 轮换密钥后验证器拒绝有效令牌。修复方法是上文的 cron 刷新 + 缓存未命中再抓取模式。切勿在无刷新作业的情况下缓存 JWKS。
- **以轮换作为回退。** 把缓存未命中路径接到“轮换并铸造”上是一个真实的 bug：它永远不会生成缺失的 `kid`，并且把攻击者控制的 `kid` 值变成密钥创建 DoS。回退必须是幂等的 `refresh-jwks`。
- **缺失 `aud` 声明。** 有些 IdP 默认在令牌请求中不包含 `resource` 时省略 `aud`。验证器必须拒绝缺失 `aud` 的令牌，不应将其视为通配符。
- **缺少 `iss` 检查导致的 mix-up。** 未在客户端上验证 RFC 9207 的授权响应 `iss` 参数与重定向前记录的 issuer 的客户端，可被引导去在攻击者的 token 端点兑换诚实 AS 的 code。这是客户端侧的失败；资源服务器无法弥补。
- **作用域升级竞态。** 对同一用户的两个并发提升流（step-up）都可能成功并产生两个作用域不同的访问令牌。验证器必须使用请求中提供的令牌，而不是查找“用户当前的作用域”——那会产生 TOCTOU 窗口。
- **注册令牌被窃取。** 泄露的 `registration_access_token` 允许攻击者改写重定向 URI。对存储的数据进行哈希；在每次更新时要求客户端提供明文；在可疑时旋转它。
- **未对 `iss` 钉定。** 接受任何 `iss` 的验证器允许攻击者架设自己的授权服务器、为目标受众注册客户端并签发令牌。受保护资源元数据的 `authorization_servers` 列表就是允许列表；必须强制执行。

## 使用示例

`code/main.py` 使用标准库 Python 和三个角色 —— `AuthorizationServer`、`ResourceServer` 与 `Client` —— 演示完整生产流。流程：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点并检查其入网选项（CIMD 用 `client_id_metadata_document_supported`，DCR 用 `registration_endpoint`）以及是否支持 `S256` PKCE。
3. 演示按 DCR 回退路径走：客户端向 `/register`（RFC 7591）发 POST，并收到 `client_id`。（CIMD 客户端将以其 own HTTPS `client_id` URL 出场并跳过此步骤。）
4. MCP 客户端使用带 PKCE 的授权码流程（RFC 7636）并携带 `resource` 指示器（RFC 8707）。
5. MCP 客户端以 `Authorization: Bearer ...` 调用 MCP 服务器上的某个工具。
6. MCP 服务器运行 `validate`，从 JWKS 缓存解析签名密钥。
7. IdP 轮换密钥；定时刷新重新拉取 JWKS 到缓存。
8. 下一次调用在刷新后的密钥上验证通过而无需重启，并且在重叠窗口期旧令牌仍然验证通过。
9. 在另一 MCP 资源上进行的受众重放尝试将收到 401，带有 `audience mismatch` 与 `resource_metadata` 指针。

这里的 JWT 使用 HS256 和共享密钥（因此本课仅需标准库即可运行）。生产环境使用 RS256 或 EdDSA 并采用 JWKS 模式；验证逻辑本质上相同。因为 IdP 与资源服务器都运行在同一进程中，`refresh_jwks` 直接读取授权服务器的密钥列表；在真实网络中它是对 `jwks_uri` 的 HTTP `GET`。

## 交付

本课生成 `outputs/skill-mcp-auth.md`。给定一个 MCP 服务器配置和 IdP 能力集，技能会输出需要立起的认证表面 —— 受保护资源元数据、要使用的入网路径（CIMD、预注册或 DCR 回退）、JWKS 刷新计划、作用域映射，以及当 IdP 不支持完整 RFC 配置文件时要应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。追踪流程。注意 IdP 在步骤 6 中轮换密钥、计划的 `refresh_jwks` 重新拉取已发布集合，以及旧令牌（重叠窗口）和新令牌在无重启的情况下都能验证通过。

2. 将一个新的 IdP 添加到受保护资源元数据的 `authorization_servers` 列表。用新 IdP 签发一个令牌并确认验证器接受它。用未列出的 IdP 签发令牌并确认验证器以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝。

3. 在 `register_client` 中添加一个在注册器接受请求前运行的速率限制检查。使用按源 IP 存放在小字典中的令牌桶。

4. 阅读 RFC 7591 并识别本课 `/register` 处理程序未验证的两个字段。加入验证逻辑。（提示：`software_statement` 和 `redirect_uris` 的 URI 方案。）

5. 添加一个 Client ID Metadata Document 路径。提供一个其 `client_id` 等于其自身 URL 的 `client.json`，并让授权服务器去获取并验证它（若 `client_id` ≠ URL 则拒绝）。确认 CIMD 客户端能在不调用 `register_client` 的情况下入网。

6. 证明 DoS 修复。向验证器发送一个带随机 `kid` 的令牌并确认 `refresh_jwks` 最多运行一次且授权服务器的密钥数量不增长。然后故意把回退重接到“轮换并铸造”，观察每个伪令牌导致密钥数量增加——随后恢复回退为重新抓取。

7. 在客户端实现 RFC 9207 的 `iss` 检查：在授权请求前记录期望的 issuer，然后拒绝返回 `iss` 与之不匹配的授权响应。

## 术语解释

| 术语 | 大家说的 | 实际含义 |
|------|---------|---------|
| ASM | "OAuth metadata document" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "Client metadata URL" | Client ID Metadata Document — 用作 `client_id` 的 HTTPS URL；AS 拉取该 JSON。自 2025-11-25 起为推荐默认 |
| DCR | "Self-service client registration" | RFC 7591 `POST /register` 流程；在 2025-11-25 被降为 `MAY` 回退 |
| JWKS | "Public keys for JWT validation" | JSON Web Key Set，从 `jwks_uri` 抓取，按 `kid` 索引 |
| Rotate vs refresh | "Updating the keys" | *Rotate* = 授权服务器铸造/退役签名密钥；*refresh* = 资源服务器重新获取已发布集合。资源服务器仅能刷新 |
| Resource indicator | "Audience parameter" | RFC 8707 的 `resource` 参数，用于把令牌钉到单个服务器 |
| `aud` claim | "Audience" | 验证器将其与规范资源 URL 比较的 JWT 声明 |
| Audience replay | "Token replay" | 为服务器 A 签发的令牌被提交给服务器 B；通过受众验证防御（规范：访问令牌权限限制） |
| Confused deputy | "Proxy token misuse" | 一个使用静态 client ID 的 MCP 代理转发令牌而未做逐客户端同意；与受众重放不同 |
| Mix-up attack | "Wrong token endpoint" | 客户端被引导在攻击者端点兑换诚实 AS 的 code；客户端通过 RFC 9207 的 `iss` 防御 |
| `iss` allow-list | "Trusted authorization servers" | 受保护资源元数据的 `authorization_servers` 中列出的集合 |
| `resource_metadata` | "Where to find the PRM doc" | 在 401/403 的 `WWW-Authenticate` 中指向 RFC 9728 元数据 URL 的参数 |
| Public client | "Native or browser client" | 无 `client_secret` 的 OAuth 客户端；PKCE 提供补偿机制 |
| `WWW-Authenticate` | "401/403 response header" | 携带 `Bearer error=...` 指令以驱动客户端恢复的头部 |

## 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 本课实现的 MCP 认证配置文件
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 2025-11-25 的变更（CIMD、XAA、DCR 降级）
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — 关于优先采用 CIMD 而非 DCR 的理由
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD 草案
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（回退路径）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端的持有证明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众钉定
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — 防止 mix-up 攻击的 `iss` 参数
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 统一的 OAuth 基质