# 并行工具调用与工具流式传输

> 将三个独立的天气查询串行执行需要三次往返。并行运行它们，总时间就会收敛到最慢的单次调用。现在每个前端提供者在一个回合内可以发出多个工具调用。收益显著；管道实现有细节。本课覆盖两部分：并行扇出和流式参数重组，重点是 id 关联陷阱。

**Type:** Build  
**Languages:** Python（stdlib、线程池 + 流式处理框架）  
**Prerequisites:** Phase 13 · 02（函数调用深度解析）  
**Time:** ~75 分钟

## 学习目标

- 解释为什么存在 `parallel_tool_calls: true` 以及何时应关闭它。  
- 在并行扇出时，将流式参数片段与正确的 tool-call id 关联。  
- 在不提前解析的情况下，将部分 `arguments` 字符串重组成完整 JSON。  
- 运行一个三城市天气基准，展示顺序与并行延迟差异。

## 问题描述

如果不使用并行调用，一个代理在回答 “班加罗尔、东京和苏黎世的天气如何” 时会这样做：

```
用户 -> LLM
LLM -> 调用 get_weather(Bengaluru)
主机 -> 运行执行器，返回结果
LLM -> 调用 get_weather(Tokyo)
主机 -> 运行执行器，返回结果
LLM -> 调用 get_weather(Zurich)
主机 -> 运行执行器，返回结果
LLM -> 最终文本回答
```

三次 LLM 往返，每次还要承担执行器延迟。大约是理想情况的 4 倍墙钟时间。

使用并行调用时：

```
用户 -> LLM
LLM -> 调用 get_weather(Bengaluru); 调用 get_weather(Tokyo); 调用 get_weather(Zurich)
主机 -> 并发运行所有三个执行器，返回三个结果
LLM -> 最终文本回答
```

一次 LLM 往返。执行器时间为三者中的最大值，而不是总和。OpenAI、Anthropic 和 Gemini 的生产基准显示，在扇出工作负载上墙钟时间减少约 60% 到 70%。

代价是关联复杂性。当三个调用乱序完成时，返回结果必须携带匹配的 `tool_call_id`，以便模型可以对齐它们。结果流式传输时，你必须在执行前将部分参数片段拼装成完整 JSON。Gemini 3 部分地通过在每次调用中添加唯一 id 来解决了现实世界中两个并行调用同名工具无法区分的问题。

## 概念

### 启用并行

- **OpenAI。** 默认开启 `parallel_tool_calls: true`。设置为 `false` 可强制串行。  
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 启用并行（Claude 3.5 及以上默认如此）。将其设为 `true` 则串行化。  
- **Gemini。** 始终支持并行；将 `tool_config.function_calling_config.mode = "AUTO"` 交由模型决定。

当工具之间存在顺序依赖（例如先 `create_file` 再 `write_file`）、一个调用的输出会影响另一个调用的输入，或速率限制器无法承受扇出流量时，应禁用并行。

### Id 关联

模型发出的每次调用都有一个 `id`。主机返回的每个结果必须包含相同的 id，否则结果会变得模糊不清。

- **OpenAI。** 每条工具角色消息上有 `tool_call_id`。  
- **Anthropic。** 每个 `tool_result` 块上有 `tool_use_id`。  
- **Gemini。** 每个 `functionResponse` 上有 `id`（Gemini 3 及以上；Gemini 2 通过名称匹配，这在同名并行调用时会出问题）。

### 并发运行调用

主机应为每个调用在独立线程、协程或远端 worker 上运行其执行器。最简单的实现使用线程池；生产环境使用 asyncio 的 `asyncio.gather` 或结构化并发。完成顺序是不可预测的 —— id 是唯一标识。

一个常见错误是按照调用列表顺序回复结果而不是按完成顺序回复。通常这样也能工作，因为模型只关心 `tool_call_id`，但如果某个结果被丢失或重复，乱序提交会使调试更困难。建议按完成顺序回复，并带上显式 id。

### 流式工具调用

当模型以流式返回时，`arguments` 会分片到达。三个并行调用的片段会在同一线路上交错到达。你需要为每个 id 提供一个累加器。

各提供者的形态：

- **OpenAI。** 每个分片位于 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。分片携带 `index`（在调用列表中的位置）。按 index 累加，首次出现时读取 `id`，并在 `finish_reason = "tool_calls"` 时解析 JSON。  
- **Anthropic。** 流事件为 `message_start`，随后每个 `tool_use` 类型的块有一个 `content_block_start`（包含 id、名称、空输入）。`content_block_delta` 事件携带 `input_json_delta` 片段。`content_block_stop` 关闭每个块。  
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）发出带有 `functionCallId` 的片段，使得调用可以干净交错。Gemini 3 之前，流式返回一次完整调用。

### 部分 JSON 与提前解析陷阱

在完整接收之前不能解析 `arguments`。部分 JSON（例如 `{"city": "Beng`）不是有效的，会抛错。正确的触发时机是提供者的结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop` 或 Gemini 的流结束事件。只有在这些信号之后才尝试 `json.loads`。更稳健的做法是使用增量 JSON 解析器，在结构完成时产出事件；OpenAI 的流式指南建议在需要展示“正在思考”指示器的交互体验中使用这一方法。基于计数括号来判断完整性并不可靠（引号内或转义内容中的大括号会导致误报），只能作为非正式的调试启发式手段。

### 乱序完成

```
call_A：快速 API，最先返回
call_B：慢速 API，第二个返回
call_C：中等 API，第三个返回
```

主机回复仍然必须引用这些 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

在 OpenAI 或 Anthropic 上，回复中的顺序对正确性并不重要。Gemini 也接受任意顺序，只要 id 匹配即可。

### 基准：顺序 vs 并行

`code/main.py` 中的测试用仿真了延迟分别为 400、600 和 800 毫秒的三个执行器。顺序执行总耗时为 1800 毫秒。并行执行耗时为 max(400, 600, 800) = 800 毫秒。差异是常数项，而非比例项，因此随着工具数量增加，节省越明显。

现实世界的注意点：并行调用会对下游 API 施加压力。对一个有速率限制的服务做 10 路扇出会失败。Phase 13 · 17 覆盖网关级别的回压；重试语义将会在未来阶段规划。

### 流式扇出的墙钟时间优化

如果模型本身以流式输出，你可以在某一次调用的 arguments 完成时就开始执行该调用，而不必等所有调用都完成。这是 OpenAI 文档中提到的优化，但并非所有 SDK 都暴露该能力。本课的实现支持这一点：当模拟流产生一个完整参数对象时，主机立即启动该调用。

## 使用方法

`code/main.py` 包含两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 以顺序和并行两种方式运行三个模拟天气调用，并打印墙钟时间。第二部分回放一个假的流式响应 —— 三个并行调用的 `arguments` 片段在同一条流上交错 —— 并用 `StreamAccumulator` 按 id 重组。无 LLM、无网络，仅含重组逻辑。

关注点：

- 顺序计时器应为 1.8 秒。并行计时器在相同假延迟下应为 0.8 秒。  
- 累加器通过为每个 id 做缓冲，即使片段乱序到达也能处理，并只在每个调用的 JSON 完整时解析。  
- 执行器在某个 id 的 arguments 完成后就启动，而不是等到所有流结束。

## 发布产物

本课会生成 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该 skill 会审核哪些工具适合并行化、哪些有顺序依赖、哪些会压垮下游速率限制 —— 并返回带有每个工具 `parallel_safe` 标志的修订注册表。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行与顺序的比率大致为 `max/sum`（实际运行会因为线程调度、序列化和实现开销而略有偏差）。在哪种延迟分布下，并行优势开始变得不显著？  
2. 扩展累加器以处理 “调用在流中途被取消” 的情况：丢弃其缓冲并发出 `cancelled` 事件。哪个提供者在文档中明确描述了这种情况？检查 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。  
3. 将线程池替换为 `asyncio.gather`。对两者进行基准测试。你应该在异步实现上看到小幅收益（由于上下文切换成本较低），但前提是执行器执行真实 I/O。  
4. 选两个不应并行化的工具（例如先 `create_file` 再 `write_file`）。向注册表添加一个 `ordering_dependency` 图，并根据该图来决定是否进行并行扇出。这是实现依赖感知调度的最小工具，后续的代理工程阶段会进一步形式化。  
5. 阅读 OpenAI 的 parallel-function-calling 一节和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 在现实情况下建议禁用并行性的那一类真实工具。（提示：对同一资源造成重大变更的操作。）

## 术语表

| 术语 | 常说的说法 | 实际含义 |
|------|-----------|----------|
| Parallel tool calls | “在一个回合内扇出” | 模型在单条助手消息中发出多个工具调用 |
| `parallel_tool_calls` | “OpenAI 的开关” | 启用或禁用多调用发射 |
| `disable_parallel_tool_use` | “Anthropic 的反向开关” | 选择退出并行的标志；默认并行启用 |
| Tool call id | “关联手柄” | 每次调用的标识符，结果消息必须回显该 id |
| Accumulator | “流缓冲” | 用于保存部分 `arguments` 片段的按 id 缓冲区 |
| Out-of-order completion | “最快先到” | 并行调用以不可预测的顺序完成；id 是粘合剂 |
| Dependency graph | “顺序约束” | 一个工具的输出作为另一个工具输入的情况；无法并行化 |
| Parse-early trap | “JSON.parse 爆炸” | 在 `arguments` 未完整时尝试解析字符串 |
| `streamFunctionCallArguments` | “Gemini 3 特性” | 带有每次调用唯一 id 的流式参数片段 |
| Completion-order reply | “不要等所有结果” | 以到达顺序回复结果，并用 id 键控它们 |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为与可选关闭标志  
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 与结果批处理  
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 的 id 关联并行调用说明  
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流式的分片参数重组指南  
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta` 与 `input_json_delta`