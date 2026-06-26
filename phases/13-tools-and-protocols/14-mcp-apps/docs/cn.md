# MCP Apps — 通过 `ui://` 提供交互式 UI 资源

> 纯文本工具输出限制了代理能展示的内容。MCP Apps（SEP-1724，官方发布时间：2026-01-26）允许工具返回在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中内联呈现的沙盒化交互式 HTML。仪表板、表单、地图、3D 场景，都可通过同一扩展实现。本课程讲解 `ui://` 资源方案、`text/html;profile=mcp-app` MIME、iframe 沙盒 postMessage 协议，以及允许服务器渲染 HTML 所带来的安全面。

**Type:** 构建  
**Languages:** Python（stdlib、UI 资源发射器）、HTML（示例应用）  
**Prerequisites:** Phase 13 · 07 (MCP server)、Phase 13 · 10 (resources)  
**Time:** ~75 分钟

## 学习目标

- 在工具调用中返回一个 `ui://` 资源并设置正确的 MIME 与元数据。
- 使用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联的 UI。
- 实现 iframe 沙盒的 postMessage JSON-RPC，用于 UI 到宿主的通信。
- 应用防御性 CSP 和 permissions-policy 默认值，以抵御由 UI 发起的攻击。

## 问题背景

一个 2025 年的 `visualize_timeline` 工具可能会返回“Here are 14 notes organized chronologically: ...” 这样的文本段落。但用户实际想要的是交互式时间线。在 MCP Apps 出现之前，可选项是：客户端特定的 widget API（Claude artifacts、OpenAI Custom GPT HTML），或者根本没有 UI。

MCP Apps（SEP-1724，2026-01-26 发布）标准化了这份合约。工具结果包含一个 URI 为 `ui://...` 的 `resource`，其 MIME 为 `text/html;profile=mcp-app`。宿主在沙盒化 iframe 中渲染它，应用受限的 CSP，并且除非明确授权，否则没有网络访问。iframe 内的 UI 通过一个精简的 postMessage JSON-RPC 方言与宿主通信。

每个兼容的客户端（Claude Desktop、ChatGPT、Goose、VS Code）都以相同方式渲染相同的 `ui://` 资源。一个服务器、一个 HTML 包，通用 UI。

## 概念说明

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

然后宿主对 `ui://notes/timeline` URI 调用 `resources/read`，返回：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙盒

宿主在沙盒化的 `<iframe>` 中渲染 HTML，属性包括：

- `sandbox="allow-scripts allow-same-origin"`（或按服务器声明更严格的设置）
- 通过响应头应用服务器声明的 CSP。
- 不会使用宿主域的 cookies 或 localStorage。
- 网络访问受限于 CSP 中的 `connectSrc`。

### postMessage 协议

iframe 使用 `window.postMessage` 与宿主通信。一个精简的 JSON-RPC 2.0 方言：

务必将 `targetOrigin` 固定为对端的精确 origin，并在接收方通过 allowlist 校验 `event.origin` 后才处理负载。切勿在该通道两端使用 `"*"` —— 消息体中携带着工具调用和资源读取。

```js
// iframe -> 宿主（将 targetOrigin 固定为宿主的确切 origin）
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// 宿主 -> iframe（将 targetOrigin 固定为 iframe 的确切 origin）
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// 双方的接收器
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // 现在可以安全地处理 event.data
});
```

UI 可以调用的宿主端可用方法：

- `host.callTool(name, arguments)` — 调用服务器上的工具。
- `host.readResource(uri)` — 读取 MCP 资源。
- `host.getPrompt(name, arguments)` — 获取提示词模板。
- `host.close()` — 关闭/取消 UI。

每次调用仍然遵循 MCP 协议并继承服务器的权限。

### 权限

`_meta.ui.permissions` 列表请求额外能力：

- `camera` — 访问用户摄像头（用于扫描文档等 UI）。
- `microphone` — 语音输入。
- `geolocation` — 位置。
- `network:*` — 比单独 `connectSrc` 更宽泛的网络访问。

每一项权限都会在 UI 渲染前以提示的形式展示给用户。

### 安全风险

iframe 中的 HTML 仍然是 HTML，会带来新的攻击面：

- **通过 UI 的提示词注入（Prompt-injection）。** 恶意服务器 UI 可以展示看起来像系统消息的文本来欺骗用户。宿主渲染时应明显区分服务器 UI 与宿主 UI。
- **通过 `connectSrc` 的外泄。** 如果 CSP 允许 `connect-src: *`，UI 可以将数据发送到任意地点。默认应保持严格。
- **点击劫持（Clickjacking）。** UI 覆盖宿主 UI。宿主必须防止 z-index 操控并强制执行不透明度规则。
- **窃取焦点（Steal focus）。** UI 获取键盘焦点并拦截下一条消息。宿主必须拦截此类行为。

Phase 13 · 15 在 MCP 安全部分深入覆盖这些问题；本课只是做入门介绍。

### `ui/initialize` 握手

iframe 加载后，会通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主返回能力集合和会话令牌。UI 在随后的每次宿主调用中使用该会话令牌。

### AppRenderer / AppFrame SDK 基础

ext-apps SDK 提供两个便利原语：

- `AppRenderer`（服务器端）— 将 React / Vue / Solid 组件封装为 `ui://` 资源，并输出正确的 MIME 与元数据。
- `AppFrame`（客户端）— 接收资源、挂载 iframe，并调停 postMessage。

你可以使用这些 SDK，或者手工实现 HTML 与 JSON-RPC。

### 生态状态

MCP Apps 于 2026-01-26 发布。截止 2026 年 4 月的客户端支持情况：

- **Claude Desktop。** 自 2026 年 1 月起全面支持。
- **ChatGPT。** 通过 Apps SDK 全面支持（基于相同的 MCP Apps 协议）。
- **Cursor。** 测试版；通过设置启用。
- **VS Code。** 仅限 Insider 构建。
- **Goose。** 全面支持。
- **Zed、Windsurf。** 已列入路线图。

生产环境中的服务器示例：仪表板、地图可视化、数据表、图表构建器、沙盒 IDE 预览。

## 使用方法

`code/main.py` 将 notes 服务器扩展为一个 `visualize_timeline` 工具，该工具返回 `ui://notes/timeline` 资源，并为该 URI 提供 `resources/read` 处理器，返回一个小型但完整的包含 SVG 时间线的 HTML 包。HTML 使用 stdlib 模板 —— 无需构建系统。postMessage 在 JS 注释中示意，因为 stdlib 无法驱动浏览器。

重点查看：

- 工具响应中的 `_meta.ui`，包含 resourceUri、CSP、permissions。
- HTML 在无网络访问情况下呈现；所有数据均已内联。
- JS 使用 `window.parent.postMessage` 调用 `host.callTool`（在此 stdlib 演示中为有文档说明但不可运行）。

## 发布产物

本课生成 `outputs/skill-mcp-apps-spec.md`。针对一个会受益于交互式 UI 的工具，该技能会产出完整的 MCP Apps 合约：`ui://` URI、CSP、permissions、postMessage 入口点以及安全检查清单。

## 练习

1. 运行 `code/main.py` 并检查输出的 HTML。直接在浏览器中打开该 HTML；验证 SVG 能否渲染。然后草拟 UI 调用 `host.callTool("notes_update", ...)` 的 postMessage 合约。

2. 收紧 CSP：移除 `'unsafe-inline'` 并使用基于 nonce 的脚本策略。HTML 生成代码需要做哪些改动？

3. 添加第二个 UI 资源 `ui://notes/editor`，提供一个内联编辑笔记的表单。用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器可能在哪里注入内容？iframe 沙盒能防护哪些攻击，不能防护哪些攻击？

5. 阅读 SEP-1724 规范，找出该玩具实现未使用的 MCP Apps SDK 能力之一。（提示：组件级状态同步）

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| MCP Apps | “Interactive UI resources” | SEP-1724 扩展，已于 2026-01-26 发布 |
| `ui://` | “App URI scheme” | 用于 UI 包的资源方案 |
| `text/html;profile=mcp-app` | “The MIME” | MCP App HTML 的 Content-Type |
| Iframe sandbox | “Render container” | 使用 CSP 与权限对 UI 进行浏览器沙盒化 |
| postMessage JSON-RPC | “UI-to-host wire” | 基于 postMessage 的精简 JSON-RPC 用于 UI 与宿主调用 |
| `_meta.ui` | “Tool-UI binding” | 将工具结果与 UI 资源关联的元数据 |
| CSP | “Content-Security-Policy” | 声明脚本、网络、样式等允许的来源 |
| AppRenderer | “Server SDK primitive” | 将框架组件转换为 `ui://` 资源的服务器端 SDK |
| AppFrame | “Client SDK primitive” | 挂载 iframe 并调停 postMessage 的客户端辅助库 |
| `ui/initialize` | “Handshake” | UI 向宿主发起的首个 postMessage 握手 |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现与 SDK  
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 规范文档  
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — 高级文档  
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月发布文章  
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 风格的 SDK 参考