# 工具接口 — 为什么代理需要结构化的 I/O

> 语言模型生成标记（tokens）。程序执行动作。两者之间的差距就是工具接口：一种让模型请求动作并由宿主执行的约定。每一种 2026 年的实现栈——OpenAI、Anthropic、Gemini 的函数调用；MCP 的 `tools/call`；A2A 的任务分片——都是对同一四步循环的不同编码。本课为该循环命名，并展示运行它的最小机制。

**Type:** 学习  
**Languages:** Python（标准库，不使用 LLM）  
**Prerequisites:** Phase 11（LLM completion APIs）  
**Time:** ~45 分钟

## 学习目标

- 解释为什么只能生成文本的 LLM 本身无法对真实世界采取行动。
- 画出四步工具调用循环（描述 → 决定 → 执行 → 观察），并指出每一步由谁负责。
- 将工具描述写成三部分：名称、JSON Schema 输入、以及确定性的执行器函数。
- 区分纯工具与有副作用工具，并说明这种划分对安全性的意义。

## 问题

LLM 对下一个标记发出一个概率分布。这就是整个输出表面。如果你问聊天模型“班加罗尔现在的天气如何”，它可以写出一段看起来合理的句子，但它无法真正调用天气 API。这句话可能偶然是对的，也可能是三天前的过时信息。

弥合这一差距的就是工具接口。宿主程序——你的代理运行时、Claude Desktop、ChatGPT、Cursor 或自定义脚本——向模型公布一组可调用的工具。模型在决定需要采取动作时，会发出一个结构化的负载，指明工具和其参数。宿主解析该负载，真正运行工具，并将结果反馈回去。循环继续，直到模型决定不再需要调用工具。

这一约定的第一个版本于 2023 年 6 月随 OpenAI 的 “functions” 参数发布。Anthropic 在 Claude 2.1 中引入了 `tool_use` 块。几个月后，Gemini 添加了 `functionDeclarations`。现在每个提供方都暴露相同的形态：请求方接收一个 JSON-Schema 类型化的工具列表，模型输出一个 JSON 负载的工具调用。模型上下文协议（Model Context Protocol，2024 年 11 月）将该约定泛化为由单一工具注册表服务所有模型。A2A（2026 年 4 月，v1.0）在代理到代理委派中分层使用了相同的原语。

四步循环是这些机制下面的不变式。Phase 13 的其余内容都是对此的不停阐述。

## 概念

### 第一步：描述

宿主用三个字段声明每个工具。

- **Name（名称）。** 一个稳定、机器可读的标识符。应为 `get_weather`，而不是“天气东西”。
- **Description（描述）。** 一段自然语言的简短说明。例如：“当用户询问某一具体城市的当前天气时使用。不要用于历史数据。”
- **Input schema（输入模式）。** 一个描述工具参数的 JSON Schema 对象（draft 2020-12）。

模型接收这些列表。现代提供方会使用特定模板将这些声明序列化到 system prompt 中，因此作为调用方你只需处理结构化形式。

### 第二步：决定

在收到用户消息和可用工具后，模型会选择下面三种行为之一。

1. **直接以文本回答。** 不调用工具。
2. **调用一个或多个工具。** 发出结构化的调用对象。在 `parallel_tool_calls: true`（OpenAI 和 Gemini 默认为 true，Anthropic 为可选）时，模型可以在一个回合中发出多个调用。
3. **拒绝。** 严格模式的结构化输出可以产生一个类型化的 `refusal` 块而不是调用。

工具调用负载有三个稳定字段：调用的 `id`、工具的 `name`，以及 JSON `arguments` 对象。id 的存在是为了让宿主将后续结果与特定调用相关联，这在并行调用可能乱序返回时很重要。

### 第三步：执行

宿主接收调用后，对参数按照声明的 schema 进行验证，并运行执行器。参数无效意味着模型为某个字段产生了幻觉或使用了错误的类型——这是弱模型的常见失败模式。生产环境的宿主在参数无效时会做三件事之一：快速失败并将错误呈现给模型、用受限解析器修复 JSON，或在 prompt 中包含验证错误后重试调用模型。

执行器本身就是普通代码。Python、TypeScript、一个 shell 命令、一个数据库查询。它会产生一个结果，通常是字符串，但也可以是任意 JSON 值或结构化内容块（在模型上下文协议中可以是文本、图像或资源引用）。结果必须可序列化。

### 第四步：观察

宿主将工具结果追加到会话中（作为具有匹配 `id` 的 `tool` 角色消息）并重新调用模型。模型现在在上下文中具有工具输出，可以产生最终回答或请求更多调用。该过程持续，直到模型不再发出调用或宿主达到迭代次数的安全限制为止。

### 信任分割

工具分为两类，这对安全性很重要。

- **纯（Pure）。** 只读、确定性、无副作用。例如 `get_weather`、`search_docs`、`get_current_time`。可以安全地进行投机性调用（speculative calls）。
- **有后果的（Consequential）。** 会改变状态、花费资金、触及用户数据。例如 `send_email`、`delete_file`、`execute_trade`。必须进行门控。

Meta 在 2026 年提出的代理安全 “二规则（Rule of Two）” 表示单个回合最多可以组合两项：不可信输入、敏感数据、有后果的动作。工具接口就是你执行该规则的地方——通过拒绝调用、要求用户确认或提升权限范围来强制执行。参见 Phase 13 · 15 的完整安全章节和 Phase 14 · 09 的代理级权限策略。

### 循环存在的位置

| 上下文 | 谁描述 | 谁决定 | 谁执行 |
|--------|--------|--------|--------|
| 单回合函数调用（OpenAI/Anthropic/Gemini） | 应用开发者 | LLM | 应用开发者 |
| 模型上下文协议（MCP） | MCP 服务器 | 通过 MCP 客户端的 LLM | MCP 服务器 |
| A2A | Agent Card 发布者 | 调用方代理 | 被调用代理 |
| Web 浏览器（函数调用代理） | 浏览器扩展 / WebMCP | LLM | 浏览器运行时 |

无论何处，都是相同的四步。列名会变；结构不会。

### 为什么不直接让模型输出 JSON？

“让模型以 JSON 回复”是函数调用出现之前的模式。在前沿模型上该模式的失败率大约为 5% 到 15%，在小型模型上更高。失败模式包括缺失大括号、尾随逗号、虚构字段以及错误类型。之后你需要做 JSON 修复、重试或受限解码器。

原生函数调用在三个方面更好。首先，提供方端到端对模型按精确调用形态进行训练，因此在严格模式下有效 JSON 的比率会提升到 98% 到 99%。其次，调用负载位于其自己的协议槽位，而不是自由文本中——因此工具调用不会泄露到用户可见回复中。第三，提供方通过受限解码（OpenAI 的严格模式、Anthropic 的 `tool_use`、Gemini 的 `responseSchema`）强制执行 schema 合规性。输出被保证通过验证。

Phase 13 · 02 并排比较了三家提供方的 API。Phase 13 · 04 深入讨论结构化输出。

### 安全断路器

当模型不再发出调用或宿主达到最大回合数时，循环终止。生产环境宿主通常将该值设为 5 到 20 回合。超过这个范围，几乎可以肯定进入了模型无法退出的循环。Claude Code 默认 20 回合；OpenAI Assistants 默认 10；Cursor 的代理模式默认 25。

另一种情形——无限循环——每 6 个月就会在事后分析中出现一次，例如“代理在夜间花费了 400 美元调用 API”。切勿在没有上限的情况下发布。

Phase 14 · 12 深入讨论错误恢复与自愈；Phase 17 涵盖生产速率限制。

### Phase 13 的后续发展

- 课程 02 到 05 将润色提供方级别的工具调用表面。
- 课程 06 到 14 将把循环泛化到模型上下文协议（MCP）。
- 课程 15 到 18 将保护循环免受恶意服务器、对抗性用户和未认证的远程授权表面的攻击。
- 课程 19 到 22 将把该模式扩展到代理间协作、可观测性、路由与打包。
- 课程 23 会发布一个使用所有原语的完整生态系统。

剩下的每一课都是对这一四步循环的展开。将其作为不变式铭记。

## 使用方法

`code/main.py` 在没有 LLM 的情况下运行四步循环。一个假的 “decider” 函数通过对用户消息做模式匹配来模拟模型；执行器、schema 验证器和观察步骤的挂载是真实的。运行它以查看具有可打印中间状态的完整请求/响应编排，然后在后续课程中将伪决策器替换为任何真实提供方。

关注点：

- 工具注册表为每个工具保存三个字段：name、description、schema，以及执行器引用。
- 验证器是一个最小的 JSON Schema 子集（types、required、enum、min/max），仅用标准库实现。Phase 13 · 04 会提供更完整的实现。
- 循环将迭代次数上限设为五。生产代理恰好需要这种断路器。

## 交付成果

本课会生成 `outputs/skill-tool-interface-reviewer.md`。给定一个草案工具定义（name + description + schema + executor outline），该 skill 会审计其循环适配性：名称是否机器稳定、描述是否为完整的使用说明、schema 是否正确使用 JSON Schema 2020-12，以及纯/有后果的分类是否明确。

## 练习

1. 在 `code/main.py` 中添加第四个工具，名为 `get_stock_price(ticker)`。将其描述写为 "Use when the user asks for a current stock price by ticker. Do not use for historical prices or market summaries." 运行测试挂架并确认伪决策器会把包含股票代码的查询路由到该新工具。

2. 破坏 schema 验证器。传入一个 `arguments` 对象缺少必需字段的调用，并确认宿主在执行前拒绝该调用。然后传入一个包含额外未知字段的调用。决定：宿主应该拒绝还是忽略？用安全论证来证明你的选择。

3. 将挂架中的每个工具分类为纯工具或有后果工具。向需要的注册条目添加 `consequential: true` 标志，并更改循环，使其在选择有后果工具时打印一行 “would confirm with user”。这就是每个生产宿主所需的确认门的形式。

4. 在纸上画出四步循环，并用上文的提供方列表为你最喜欢的客户端（Claude Desktop、Cursor、ChatGPT 或自定义栈）填写表格。与 Phase 13 · 06 中的 MCP 特定变体交叉参考。

5. 通读 OpenAI 的函数调用指南。从中找出一个出现在请求中但在此处四步循环所描述模型中未出现的字段。解释它增加了什么内容以及为什么它是方便的而非必需的。

## 术语表

| 术语 | 人们说的 | 实际含义 |
|------|--------|----------|
| Tool | “模型可以调用的东西” | 由 name + JSON-Schema 类型化输入 + 执行器函数 三元组组成 |
| Function calling / 函数调用 | “原生工具使用” | 提供方级别的 API 支持，允许输出结构化的工具调用而非自由文本 |
| Tool call | “模型请求执行的动作” | 模型发出的包含 `id`、`name`、`arguments` 的 JSON 负载 |
| Tool result | “工具返回的内容” | 执行器的输出，封装为具有匹配 id 的 `tool` 角色消息 |
| Parallel tool calls | “一次多个调用” | 在一个模型回合中发出的多个调用对象，彼此独立，可按 id 排序 |
| Strict mode / 严格模式 | “保证输出为 JSON” | 强约束解码，强制模型输出通过声明的 schema 验证 |
| Pure tool / 纯工具 | “只读工具” | 无副作用；可安全重试 |
| Consequential tool / 有后果工具 | “执行动作的工具” | 会改变外部状态；需要门控、审计或用户确认 |
| Four-step loop / 四步循环 | “工具调用周期” | describe → decide → execute → observe |
| Host | “代理运行时” | 持有工具注册表、调用模型并运行执行器的程序 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — OpenAI 风格工具声明和调用形态的权威参考  
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Claude 的 `tool_use` / `tool_result` 块格式  
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 的 `functionDeclarations` 与并行调用语义  
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 工具接口的提供方无关泛化规范  
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 每个现代工具 API 所使用的 schema 方言