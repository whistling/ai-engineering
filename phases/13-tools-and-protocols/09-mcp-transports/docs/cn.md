# MCP Transports — stdio vs Streamable HTTP vs SSE Migration

> stdio 只在本地可用，其他地方不可用。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输已被弃用，并将于 2026 年中期移除。选错传输会导致迁移成本；选对则可获得可在远程托管的 MCP 服务器，具备会话连续性和防止 DNS 重绑定的保护。

**Type:** 学习  
**Languages:** Python（stdlib，Streamable HTTP 端点骨架）  
**Prerequisites:** Phase 13 · 07, 08（MCP 服务器和客户端）  
**Time:** ~45 分钟

## 学习目标

- 根据部署形态（本地 vs 远程、单进程 vs 集群）在 stdio 和 Streamable HTTP 之间做出选择。
- 实现 Streamable HTTP 单端点模式：用 POST 发送请求，用 GET 建立会话流。
- 强制执行 `Origin` 验证和会话 id 语义，以抵御 DNS 重绑定攻击。
- 在 2026 年中期移除截止前，将遗留的 HTTP+SSE 服务器迁移到 Streamable HTTP。

## 问题概述

第一个 MCP 远程传输（2024-11）是 HTTP+SSE：两个端点，一个用于客户端的 POST，另一个是服务器到客户端的 Server-Sent-Events 通道。它能工作，但也很笨重：每个会话需要两个端点，某些 CDN 前的缓存会出错，并且强依赖于长连接的 SSE，而一些 WAF 会主动中断这些连接。

2025-03-26 规范用 Streamable HTTP 取代了它：一个端点，POST 用于客户端请求，GET 用于建立会话流，两者共享 `Mcp-Session-Id` 头。从那以后构建或迁移的每个服务器都使用 Streamable HTTP。旧的 SSE 模式正在弃用——Atlassian Rovo 已在 2026 年 6 月 30 日移除；Keboola 在 2026 年 4 月 1 日移除；大多数剩余的企业服务器将在 2026 年底之前移除。

而 stdio 对本地服务器仍然很重要。Claude Desktop、VS Code 以及所有 IDE 形态的客户端都通过 stdio 启动服务器。正确的心智模型是：stdio 用于“此机器”，Streamable HTTP 用于“通过网络”。两者不应混用。

## 概念

### stdio

- 子进程传输。客户端派生服务器，通过 stdin/stdout 通信。
- 每行一个 JSON 对象。以换行分隔。
- 无会话 id；进程身份即会话。
- 无需认证（子进程继承了父进程的信任边界）。
- 切勿用于远程服务器——那时你需要 SSH 或 socat 隧道，既然如此就直接使用 Streamable HTTP。

### Streamable HTTP

单一端点 `/mcp`（或任何路径）。支持三种 HTTP 方法：

- **POST /mcp.** 客户端发送 JSON-RPC 消息。服务器要么以单个 JSON 响应回复，要么返回一个包含一个或多个响应的 SSE 流（适用于批量响应和与该请求相关的通知）。
- **GET /mcp.** 客户端打开一个长连接的 SSE 通道。服务器用它来进行服务器到客户端的请求（采样、通知、触发询问等）。
- **DELETE /mcp.** 客户端显式终止会话。

会话由服务器在首次响应时设置并由客户端在随后的每次请求中回显的 `Mcp-Session-Id` 头识别。会话 id 必须是加密随机的（128+ 位）；为安全起见，拒绝客户端自选 id。

### 单端点 vs 双端点

旧规范的双端点模式在 2026 年仍可调用——规范将其标记为“向后兼容”。但所有新服务器都应使用单端点。官方 SDK 发出的都是单端点；只有在要与未迁移的远端通信时才使用遗留模式。

### `Origin` 验证与 DNS 重绑定

浏览器今天不是 MCP 客户端，但攻击者可以制作网页，诱使浏览器向 `localhost:1234/mcp` 发起 POST——这可能就是用户本地 MCP 服务器监听的地址。如果服务器不检查 `Origin`，浏览器的同源策略也无法保护它，因为 `Origin: http://evil.com` 在跨域请求中也是合法的。

2025-11-25 规范要求服务器拒绝 `Origin` 不在允许列表中的请求。该允许列表通常包含 MCP 客户端主机（例如 `https://claude.ai`、`vscode-webview://*`）以及用于本地 UI 的 localhost 变体。

### 会话 id 生命周期

1. 客户端在首次请求时不带 `Mcp-Session-Id`。
2. 服务器分配一个随机 id，并在响应头中设置 `Mcp-Session-Id`。
3. 客户端在随后所有请求以及用于流的 `GET /mcp` 中回显该头。
4. 会话可被服务器吊销；客户端在后续请求中看到 404 并必须重新初始化。
5. 客户端可以显式 DELETE 会话以实现干净关闭。

### 保活与重连

SSE 连接会中断。客户端通过使用相同的 `Mcp-Session-Id` 重新 GET 来重建连接。服务器必须排队在断开期间丢失的事件（在合理窗口内）并通过客户端回显的 `last-event-id` 头重放这些事件。

Phase 13 · 13 涵盖 Tasks，可让长时运行的工作在整个会话重连期间存活。

### 向后兼容探测

想同时支持新旧服务器的客户端可以：

1. POST 到 `/mcp`。
2. 如果响应是带 JSON 或 SSE 的 `200 OK`，则这是 Streamable HTTP。
3. 如果响应是 `200 OK` 且 `Content-Type: text/event-stream` 同时带有指向二级端点的 `Location` 头，则这是遗留的 HTTP+SSE；按照 `Location` 继续。

### Cloudflare、ngrok 与托管

2026 年的生产级远程 MCP 服务器运行在 Cloudflare Workers（使用其 MCP Agents SDK）、Vercel Functions，或容器化的 Node/Python 上。关键是：你的托管必须支持 SSE GET 的长连接。Vercel 免费层将连接限制为 10 秒，不适用。Cloudflare Workers 支持无限期的流。

### 网关合成

当你用网关（Phase 13 · 17）在多个 MCP 服务器前端进行组合时，网关是一个单一的 Streamable HTTP 端点，它会重写会话 id 并对上游进行复用。工具在网关层合并；客户端看到的是单一逻辑服务器。

### 传输故障模式

- stdio SIGPIPE。子进程在写入中死亡会触发 SIGPIPE；服务器应当干净地退出。客户端应检测 EOF 并将会话标记为已死。
- HTTP 502 / 504。Cloudflare、nginx 及其他代理在上游失败时会返回这些。Streamable HTTP 客户端应在短暂退避后重试一次。
- SSE 连接断开。TCP RST、代理超时或客户端网络变化都可能关闭流。客户端用 `Mcp-Session-Id` 和可选的 `last-event-id` 重连以恢复。
- 会话吊销。服务器使会话 id 失效；客户端在下次请求中看到 404。客户端必须重新握手。
- 时钟偏差。客户端与服务器在资源 TTL 计算上出现分歧。客户端应把服务器时间戳视为权威。

### 何时绕过 Streamable HTTP

一些企业在自己的网络内部将 MCP 服务器部署在 gRPC 或消息队列后面。这不是规范内的标准做法——MCP 规范并未正式定义这些。网关可以向 MCP 客户端暴露 Streamable HTTP 表面，同时在内部使用 gRPC。保持对外表面符合规范；网关负责翻译。

## 使用方法

`code/main.py` 使用 `http.server`（标准库）实现了一个最小的 Streamable HTTP 端点。它处理 `/mcp` 上的 POST、GET 和 DELETE，在首次响应时设置 `Mcp-Session-Id`，验证 `Origin`，并拒绝来自不在允许列表中的 Origin 的请求。处理器重用第 07 课笔记服务器的分发逻辑。

值得注意的点：

- POST 处理器读取 JSON-RPC 请求体，分发处理，并写入一个 JSON 响应（单响应变体；SSE 变体在结构上类似）。
- `Origin` 检查会拒绝默认的探针 `http://evil.example`，但接受 `http://localhost`。
- 会话 id 为随机 128 位十六进制字符串；服务器在内存中保存每个会话的状态。

## 交付物

本课产出 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（遗留）MCP 服务器，该技能会产出一个迁移方案到 Streamable HTTP，包含会话 id 连续性、Origin 检查与向后兼容的探测支持。

## 练习

1. 运行 `code/main.py`。用 `curl` POST 一次 `initialize` 并观察响应头中的 `Mcp-Session-Id`。再 POST 一次并回显该头以验证会话连续性。

2. 添加一个 GET 处理器以打开 SSE 流。每 5 秒发送一次 `notifications/progress` 事件。使用相同的会话 id 重新 GET 以确认服务器接受重连。

3. 实现 `last-event-id` 重放逻辑。在重连时重放自该 id 之后生成的任何事件。

4. 扩展 `Origin` 验证以支持通配模式（`https://*.example.com`），并确认它接受 `https://app.example.com`，但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册表中选择一个遗留的 HTTP+SSE 服务器（有若干），并草拟迁移方案：端点处理、会话 id 生成和头语义需要发生哪些变化。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| stdio transport | "Local child process" | JSON-RPC 通过 stdin/stdout，换行分隔 |
| Streamable HTTP | "The remote transport" | 单端点 POST + GET + 可选 SSE，2025-03-26 规范 |
| HTTP+SSE | "Legacy" | 两端点模型，将在 2026 年中期移除 |
| `Mcp-Session-Id` | "Session header" | 由服务器分配并在随后的每次请求中回显的随机 id |
| `Origin` allowlist | "DNS-rebinding defense" | 拒绝其 Origin 未被批准的请求 |
| Single endpoint | "One URL" | `/mcp` 处理所有会话操作的 POST / GET / DELETE |
| `last-event-id` | "SSE replay" | 用于在丢失流后无缝恢复并不漏事件的头 |
| Backwards-compat probe | "Old vs new detection" | 客户端通过响应形态自动选择传输的探测 |
| Long-lived HTTP | "SSE streaming" | 服务器在一条 TCP 连接上推送分钟或小时级的事件 |
| Session revocation | "Force re-init" | 服务器使会话 id 失效；客户端必须重新握手 |

## 延伸阅读

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 与 Streamable HTTP 的权威参考  
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的修订版  
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — 在 Workers 上托管 Streamable HTTP 的模式  
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 不同部署形态的比较  
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体迁移截止日期示例