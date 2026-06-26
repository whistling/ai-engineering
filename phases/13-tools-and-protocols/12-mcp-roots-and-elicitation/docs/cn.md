# Roots and Elicitation — 范围限定与中途用户输入

> 硬编码路径在用户打开不同项目时立刻失效。预填的工具参数在用户欠缺说明时会失效。Roots 将服务器的操作范围限定为用户可控的一组 URI；elicitation 在工具调用中途暂停，向用户通过表单或 URL 询问结构化输入。两个客户端原语，对常见 MCP 失效模式的两个修复。SEP-1036（URL 模式 elicitation，2025-11-25）在 2026 年上半年仍为实验性 —— 在依赖它之前请检查 SDK 版本。

**Type:** 构建  
**Languages:** Python (stdlib, roots + elicitation demo)  
**Prerequisites:** Phase 13 · 07（MCP 服务器）  
**Time:** ~45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。  
- 将服务器的文件操作限制在声明的 root 集合内的 URI 范围内。  
- 使用 `elicitation/create` 在工具调用中途询问用户确认或结构化输入。  
- 在表单模式与 URL 模式的 elicitation 之间进行选择（后者为实验性；存在漂移风险）。

## 问题

在生产中，笔记类 MCP 服务器遇到的两个具体失败场景。

**断裂的路径假设。** 服务器针对 `~/notes` 编写。某位用户在另一台机器上把笔记放在 `~/Documents/Notes`，于是工具调用静默失败（找不到文件），更糟的是可能写到了错误的位置。

**用户知道但未提供的参数。** 用户请求“删除旧的 TPS 报告笔记”。模型调用了 `notes_delete(title: "TPS report")`，但有三个匹配的笔记，分别来自 2023、2024 和 2025 年。工具无法猜测。直接返回“模糊不清”的失败很恼人；对所有三条同时执行则可能造成灾难。

Roots 解决第一个问题：客户端在 `initialize` 时声明服务器可以访问的 URI 集合。Elicitation 解决第二个问题：服务器在工具调用中暂停，发送 `elicitation/create` 去让用户选择其中一个。

## 概念

### Roots

客户端在 `initialize` 时声明 root 列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

服务器随后可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须将 roots 视为边界：任何位于声明的 root 集合之外的文件读写操作都应被拒绝。客户端不会强制执行这一点（服务器仍是用户信任的代码），但符合规范的服务器会遵守。

当用户添加或移除 root 时，客户端会发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新其边界。

### 为什么 roots 是客户端原语

Roots 由客户端声明，因为它们代表用户的同意模型。用户对 Claude Desktop 说“给这个笔记服务器访问这两个目录的权限”。服务器不能扩大该范围。

### Elicitation：默认的表单模式

`elicitation/create` 接受一个表单 schema 和一段自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染表单、收集用户答案并返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的动作：`accept`（用户填写并提交）、`decline`（用户关闭表单）、`cancel`（用户取消整个工具调用）。

表单 schema 为扁平结构——v1 不支持嵌套对象。SDK 通常会拒绝任何复杂于单层的结构。

### Elicitation：URL 模式（SEP-1036，实验性）

2025-11-25 新增。服务器无需发送 schema，而是提供一个 URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开该 URL，等待完成，当用户返回时再返回结果。适用于 OAuth 流程、支付授权和需要浏览器交互且表单不足以处理的文档签名场景。

漂移风险提示：SEP-1036 的响应结构仍在演进；有些 SDK 返回回调 URL，有些返回完成令牌（completion token）。在生产环境使用 URL 模式前请阅读所用 SDK 的发行说明。

### 何时使用 elicitation

- 在破坏性操作前要求用户确认（破坏性提示 + elicitation）。  
- 消歧（在 N 个候选项中选择一个）。  
- 首次运行设置（API 密钥、目录、偏好设置）。  
- 类似 OAuth 的流程（使用 URL 模式）。

### 何时不应使用 elicitation

- 为工具的必需参数填值，但模型本可以用自然语言提问的场景。应使用普通的重新提示，而不是弹出 elicitation 对话。  
- 高频调用。Elicitation 会中断会话；不要在循环内部触发它。  
- 服务器本可以事后验证的内容。应先验证、返回错误，让模型通过文本向用户询问。

### 人机闭环桥接

Elicitation 与采样（sampling）结合可以实现 MCP 的“人机闭环”模型。服务器的 agent 循环可以在等待用户输入（elicitation）或等待模型推理（sampling）之间暂停。第 13 阶段 · 11 覆盖了采样；本课讲授 elicitation。将二者结合可以实现完整的中途控制。

## 使用示例

`code/main.py` 将笔记服务器扩展为：

- 对 `roots/list` 的响应，并在收到 root 列表变化通知后重新查询。  
- 一个在多条匹配时使用 `elicitation/create` 进行消歧的 `notes_delete` 工具。  
- 一个使用 URL 模式 elicitation 打开首运行配置页面（模拟）的 `notes_setup` 工具。  
- 一个边界检查，拒绝对声明 roots 之外的 URI 进行操作。

演示运行三种场景：正常路径（匹配一条）、消歧（匹配三条，触发 elicitation）、超出 root 的写入（被拒绝）。

## 发布产物

本课会生成 `outputs/skill-elicitation-form-designer.md`。针对可能需要用户确认或消歧的工具，skill 需要设计 elicitation 表单 schema 和消息模板。

## 练习

1. 运行 `code/main.py`。触发消歧路径；确认模拟的用户答案被正确路由回工具。  

2. 添加一个新工具 `notes_archive`，它每次都要求 elicitation 确认（破坏性提示）。检查 UX：与模型通过文本重新询问相比体验如何？  

3. 为首次运行的 OAuth 流程实现 URL 模式 elicitation。注意漂移风险并添加 SDK 版本保护。  

4. 扩展 `roots/list` 的处理：当收到通知时，服务器应原子地重新读取并重新扫描可能已超出范围的打开文件句柄。  

5. 阅读 GitHub 上的 SEP-1036 讨论串。找出一个仍未解决的问题，该问题会影响服务器如何处理 URL 模式的回调。

## 关键术语

| 术语 | 人们如何说 | 实际含义 |
|------|-----------|---------|
| Root | "同意边界" | 客户端允许服务器访问的 URI |
| `roots/list` | "服务器请求作用域" | 客户端返回当前的 root 集合 |
| `notifications/roots/list_changed` | "用户改变了作用域" | 客户端通知 root 集合已变化 |
| Elicitation | "在调用中间询问用户" | 服务器发起的结构化用户输入请求 |
| `elicitation/create` | "该方法" | 用于 elicitation 请求的 JSON-RPC 方法 |
| 表单模式 | "基于 schema 的表单" | 在客户端 UI 中渲染的扁平 JSON Schema 表单 |
| URL 模式 | "浏览器重定向" | SEP-1036 实验性；打开 URL 并等待完成 |
| `accept` / `decline` / `cancel` | "用户响应结果" | 服务器需要处理的三个分支 |
| 消歧 | "选择一个" | 当工具有 N 个候选项时常见的 elicitation 用例 |
| 扁平表单 | "仅顶层属性" | elicitation schema 不能嵌套 |

## 延伸阅读

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — 官方 roots 参考  
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 官方 elicitation 参考  
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 的新增功能讲解  
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 模式 elicitation 提案（实验性，存在漂移风险）  
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX 演示与讲解