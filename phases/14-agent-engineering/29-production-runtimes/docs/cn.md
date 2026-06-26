# Production Runtimes: Queue, Event, Cron

> Production agents run on six runtime shapes: request-response, streaming, durable execution, queue-based background, event-driven, and scheduled. Pick the shape before you pick the framework. Observability is load-bearing at every shape.

**Type:** 学习  
**Languages:** Python（标准库）  
**Prerequisites:** 阶段 14 · 13 (LangGraph), 阶段 14 · 22 (Voice)  
**Time:** ~60 分钟

## Learning Objectives

- 能说出六种生产运行时形态，并将每种形态匹配到相应的框架 / 产品模式。
- 解释为什么持久化执行（LangGraph）对于长时程任务很重要。
- 描述事件驱动运行时以及 Claude Managed Agents 在何种场景下适用。
- 解释为什么可观测性对多步骤智能体来说是承重项目（load-bearing）。

## The Problem

Production agents fail in ways a Jupyter notebook doesn't surface: network timeouts at step 37, user hangs up mid-voice call, cron job dies on machine reboot, background worker runs out of memory. The runtime shape determines which failures are survivable.

## The Concept

### Request-response

- 同步 HTTP。用户等待完成。
- 仅适用于短任务（<30 秒）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP 访问日志 + OTel spans。

### Streaming

- 使用 SSE 或 WebSocket 实现渐进输出。
- LiveKit 将其扩展到 WebRTC 用于语音/视频（第22课）。
- 技术栈：任何支持流的框架 + 能处理 SSE/WS 的前端。
- 可观测性：按数据块的时间统计、首标记延迟、尾延迟。

### Durable execution

- 每一步后检查点保存状态；在失败时自动恢复。
- AutoGen v0.4 的 actor model 将故障隔离到单个 agent（第14课）。
- LangGraph 的核心差异化特性（第13课）。
- 在步骤数未知且恢复代价很高时至关重要。

### Queue-based / background

- 作业进入队列，工作者取出执行，结果通过 webhook 或 pub/sub 返回。
- 对于长时程智能体（每个任务数十到数百步，参见 Anthropic 的 computer use 声明）至关重要。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自研方案。
- 可观测性：队列深度、每个作业的延迟分布、DLQ 大小。

### Event-driven

- 智能体订阅触发器：新邮件、PR 打开、cron 触发等。
- Claude Managed Agents 开箱即用支持此类场景（第17课）。
- CrewAI Flows（第15课）用于构建事件驱动的确定性工作流。
- 可观测性：触发来源、事件到启动的延迟、智能体延迟。

### Scheduled

- 类 cron 的智能体周期性运行。
- 与持久化执行结合，使失败的夜间运行可以在下一个周期继续恢复。
- 技术栈：Kubernetes CronJob + 持久化框架；托管方案（Render cron、Vercel cron）。

### 2026 deployment patterns

- **CrewAI Flows** 用于事件驱动的生产环境。
- **Agno** 用于 Python 无状态 FastAPI 微服务。
- **Mastra** 用作嵌入式服务器适配器（Express、Hono、Fastify、Koa）。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（第22课）。
- **Claude Managed Agents** 用于托管的长时程异步任务。

### Observability is load-bearing

没有 OpenTelemetry GenAI spans（第23课）以及 Langfuse/Phoenix/Opik 后端（第24课），你无法调试在第 40 步失败的多步骤智能体。这不是可选项。它决定了“我们能快速调试”与“我们得从头重放并加更多日志”之间的区别。

### Where production runtimes fail

- **Wrong shape choice.** 为 5 分钟的任务选择 request-response。用户挂断；工作者堆积；重试级联。
- **No DLQ.** 队列工作者没有死信队列。失败的作业消失无踪。
- **Opaque background work.** 后台智能体运行但不导出追踪。失败在用户报告前完全不可见。
- **Skipping durable state.** 任何超过 30 秒且无法承受重启的运行都需要持久化执行。

## Build It

`code/main.py` 是一个使用标准库的多形态示例：

- Request-response 端点（普通函数）。
- Streaming 处理器（生成器）。
- 带 DLQ 的队列工作者。
- 事件触发注册表。
- 类 cron 的调度器。

运行：

```bash
python3 code/main.py
```

输出：五条 trace，展示相同任务在每种形态下的行为。相同的 agent 逻辑，不同的外围壳。持久化执行（第六种形态）在第13课通过 LangGraph 检查点机制有意单独覆盖。

## Use It

- **Request-response** 用于聊天式用户体验。
- **Streaming** 用于渐进响应。
- **Durable** 用于长时程任务。
- **Queue** 用于批处理 / 异步 / 长时运行。
- **Event** 用于智能体的响应性。
- **Cron** 用于例行维护（内存整理、评估、成本报告）。

## Ship It

`outputs/skill-runtime-shape.md` 为某项任务挑选运行时形态并连接可观测性需求。

## Exercises

1. 将你的第01课 ReAct 循环移植到你栈中的所有六种形态。哪种形态适合哪个产品面？
2. 为基于队列的示例添加 DLQ。模拟 10% 的作业失败；导出并展示 DLQ 大小。
3. 编写一个以 cron 触发的评估智能体，每晚对当天的前 20 条 trace 运行评估。
4. 实现带背压的流式传输：如果客户端很慢，则暂停智能体。这个机制如何与回合预算（turn budget）交互？
5. 阅读 Claude Managed Agents 文档。什么时候你会将自托管的长时程智能体迁移到托管服务？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Request-response | "Synchronous" | 用户等待；仅限短任务 |
| Streaming | "SSE / WS" | 渐进输出；更好的 UX；按数据块可观测延迟 |
| Durable execution | "Resume from failure" | 检查点状态；从上一步重启 |
| Queue-based | "Background jobs" | 生产者 / 工作池 / DLQ |
| Event-driven | "Trigger-based" | 智能体对外部事件做出反应 |
| DLQ | "Dead-letter queue" | 失败作业的停放区 |
| Claude Managed Agents | "Hosted harness" | Anthropic 托管的长时程异步，带缓存与压缩 |

## Further Reading

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 持久化执行细节  
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管的长时程异步  
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — “每个任务数十到数百步”  
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model 故障隔离