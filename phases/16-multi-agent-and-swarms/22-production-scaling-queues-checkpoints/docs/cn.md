# Production Scaling — Queues, Checkpoints, Durability

> 将多智能体系统扩展到数千个并发运行需要**持久化执行**。LangGraph 的运行时在每个超步后写入以 `thread_id` 为键的检查点（默认是 Postgres）；工作进程崩溃会释放租约，另一个工作进程接手并恢复。智能体可能会无限期地等待人工输入。**MegaAgent**（arXiv:2408.09955）为每个智能体运行了一个生产者-消费者队列，包含三种状态（Idle / Processing / Response）和两层协调（组内聊天 + 组间管理聊天）。在 LLM 流式场景下，Fiber/async 优于每任务线程：线程在等待令牌时 99% 时间处于空闲，fiber 在 I/O 上协作式让出。反论点：Ashpreet Bedi 的《Scaling Agentic Software》主张在负载未证明需要之前采用 **FastAPI + Postgres + 其他什么都不要**——简单架构往往能走得更远。本课构建一个持久化检查点日志、每智能体工作队列（带状态转换）、一个 async vs thread 的演示，并强调务实的“先简单”准则。

**Type:** 学习 + 构建  
**Languages:** Python（标准库，`asyncio`，`sqlite3`）  
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)  
**Time:** ~75 分钟

## 问题

一个原型多智能体系统在单台笔记本的内存事件循环中、三个智能体时工作良好。迁移到生产环境后遇到问题：

- 智能体有时运行数小时（长期研究、人工参与等待）。
- 工作进程会崩溃。重启会丢失状态。
- 峰值负载是平均值的 10 倍；需要水平扩展。
- 用户按每次智能体运行付费；需要精确一次语义用于计费。

内存事件循环不能满足这些需求。你需要在其下构建一个持久化执行层。2026 年的典型选项是：

1. 带检查点的工作流引擎（Temporal、LangGraph 运行时）。
2. 带状态存储的消息队列（Postgres + SQS/RabbitMQ）。
3. 演员模型框架（MegaAgent 的每智能体生产者-消费者）。
4. 手工实现的 FastAPI + Postgres（Bedi 的论点）。

本课构建上述每种方案的简化版。

## 概念

### 持久化执行，模式

持久化执行引擎在每个“步骤”（LangGraph 术语为超步）之后持久化完整的程序状态。崩溃时：

```
工作进程在超步中崩溃
  -> 租约超时
  -> 另一个工作进程接手该 thread_id
  -> 从上一个检查点恢复
  -> 无重复的副作用
```

要实现这一点需要满足：

- **可序列化的状态。** 所有智能体状态必须可持久化。带有活动数据库连接的函数闭包无法生存。
- **确定性恢复。** 给定相同状态和相同输入，智能体产生相同的动作（或者将 LLM 调用的决定交给外部确定性仲裁器）。
- **幂等的副作用。** 外部调用（工具调用、支付）必须是幂等的，或使用去重键。

LangGraph 在每个超步后写入检查点；Temporal 在每个 activity 后写入；Restate 使用事件源化日志。三者实现的是同一模式。

### LangGraph 的运行时

每个智能体有一个 `thread_id`；状态是一个带类型的字典；每个超步会向 checkpoints 表追加一行。恢复时，运行时从最后一个检查点开始回放，而不是从头开始。智能体可以 `interrupt()` 等待人工输入；运行时会持久化并释放 worker。当输入到达时，任何 worker 都可以恢复。

这是 2026 年的参考生产设计。

### MegaAgent 的每智能体队列

arXiv:2408.09955 描述了一个规模实验：一个集群中同时运行数千个智能体。架构如下：

```
agent i:
  state ∈ {空闲(Idle), 处理中(Processing), 响应(Response)}
  in_queue   <- 发往 agent i 的消息
  out_queue  -> 回复 + 副作用

coordinators:
  组内聊天 (组内的 agent 之间)
  组间管理聊天 (高层路由)
```

两层协调使得组内对话可以高密度进行，而组间保持稀疏——这是在数千个智能体时保持成本线性增长的模式。

### Async vs 每任务线程

LLM 调用是 I/O 绑定的。线程在等待下一个令牌时有 99% 的时间处于空闲。fibers（Python 的 `asyncio`、Go 的 goroutine、Rust 的 `tokio`）在 I/O 上协作式让出。相同的 10,000 个调用可以在单进程中轻松容纳。在 LLM-智能体规模下，async 并不是一个优化——它是架构。

例外：CPU 绑定的后处理（嵌入、分词器优化）仍然需要线程或进程。将你的 I/O 层与 CPU 层分离。

### Bedi 的反论点

"Ashpreet Bedi, Scaling Agentic Software (2026)" 认为大多数团队在未测量负载前过度设计。务实的默认配置是：

- FastAPI + Postgres。
- 每次智能体运行为一行；使用乐观并发原地更新状态。
- 后台作业通过 `pg_notify` 或简单的 Celery worker 实现。
- 在应用代码内实现重试策略。

对于可控任务下大约少于 100 个并发智能体运行，这通常就足够了。测量失败后再升级。

规则：当你遇到简单架构无法解决的具体问题时，再采用持久化执行框架。过早采用会消耗大量时间在不必要的仪式上。

### 精确一次语义

对于计费的智能体运行，你需要“有效精确一次”（至少一次投递 + 幂等消费者）。工程做法有：

- **每次运行的去重键。** 在每个副作用调用中包含去重键。
- **Outbox 模式。** 副作用先写入表，然后由独立进程执行两步都要幂等。
- **补偿事务。** 当副作用成功但其跟踪写入失败时，安排补偿。

这些是数据库工程模式，与 LLM 无关。LLM 的额外成本只是调用慢；其余都是标准分布式系统工程。

### Rainbow 部署

Anthropic 的多智能体研究系统使用“rainbow deployments”：多个版本的智能体运行时并行运行，这样长期运行的智能体在每次代码部署时不必被杀死。在一部分流量上做金丝雀（canary）测试新版本；当旧版本上的智能体运行完成后才退役旧版本。

这是面向长期运行有状态系统的标准做法；2026 年的适配是智能体可能运行数小时，因此部署周期必须兼容这一点。

### 典型的生产清单

- 持久化状态（检查点、快照，或 outbox + 可重放日志）。
- 幂等的副作用。
- 用于 LLM 调用的异步 I/O 层。
- 至少一次投递并配合去重。
- 针对有状态工作负载的 rainbow/canary 部署。
- 可观测性：每个智能体的追踪、超步审计、重试计数。

## 构建它

`code/main.py` 实现了：

- `CheckpointStore` — 基于 SQLite 的检查点日志，以 thread-id 为键。每个超步追加一行。
- `run_with_checkpoint(agent, thread_id)` — 模拟运行中崩溃；第二个 worker 从上一个检查点恢复。
- `AgentQueue` — 每智能体的 Idle / Processing / Response 状态机，带一个小的工作队列。
- `demo_async_vs_threads()` — 通过 asyncio 和线程分别运行 500 个并发模拟 “LLM 调用”；报告实时时间和峰值内存（近似）。

运行：

```
python3 code/main.py
```

预期输出：模拟崩溃后检查点恢复成功；async 版本在 < 1s 内处理 500 个并发调用；线程版本耗时数秒并且每并发单元使用数量级更大的内存。

## 使用它

`outputs/skill-scaling-advisor.md` 对持久化执行的选择提供建议：FastAPI + Postgres、LangGraph 运行时、Temporal 或自定义方案。根据负载、状态保留需求和部署频率进行校准。

## 投产建议

典型的生产加固措施：

- **先简单开始（Bedi 的规则）。** 在测量失败前使用 FastAPI + Postgres。
- **在优化前先对一切进行打点。** 每次运行的延迟直方图、每步耗时、重试计数、失败分类。
- **副作用使用 Outbox 模式。** 尤其是支付和外部 API 调用。
- **Rainbow 部署。** 在部署期间不要杀死进行中的智能体运行。
- **在你遇到特定问题时采用耐久执行引擎（Temporal / LangGraph / Restate）。** 例如：小时级的人工介入等待、跨区域协调、复杂的重试/补偿策略。
- **I/O 层使用 Async。** 仅在 CPU 密集型后处理中使用线程。

## 练习

1. 运行 `code/main.py`。确认检查点恢复工作；测量 async vs thread 的并发差异。  
2. 实现一个 **outbox** 表：每次工具调用先写入 outbox，然后一个独立的 goroutine/任务去执行。通过两次运行工具调用来验证幂等性。  
3. 模拟一个 **rainbow 部署**：两个并行运行的运行时版本；将一半的新 thread_ids 路由到每个版本；确认旧版本上的进行中线程不会被中断。  
4. 阅读 LangGraph 的运行时文档（下面有链接）。识别在手工实现 FastAPI + Postgres 时哪些运行时特性最费时去复制。那是否是采用运行时的理由，还是可以推迟实现？  
5. 阅读 MegaAgent（arXiv:2408.09955）第 3 节。两层协调（组内 + 组间管理聊天）是明确的。画出如何将其映射到带有两类队列的消息队列方案的草图。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Durable execution | "Persist the program state" | 引擎在每个超步后写状态；崩溃恢复是确定性的。 |
| Super-step | "Transactional boundary" | 检查点之间的工作单元。LangGraph 术语。 |
| thread_id | "Agent run identifier" | 绑定检查点和恢复逻辑的键。 |
| Idempotency | "Safe to retry" | 重复执行副作用与只执行一次产生相同结果。 |
| Outbox pattern | "Decouple side effects" | 将意图写入表；独立执行器执行并标记已完成。 |
| At-least-once delivery | "Possible duplicates" | 消息队列语义；去重键使消费者在效果上实现一次。 |
| Rainbow deploy | "Overlapping versions" | 长期运行工作负载期间并存的多个运行时版本。 |
| Async fiber | "Cooperative yielding" | 用户态并发；对于 I/O 绑定负载比线程更廉价。 |
| Checkpoint | "State snapshot" | 超步边界的序列化状态快照；是恢复的关键。 |

## 深入阅读

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph 运行时设计  
- [MegaAgent](https://arxiv.org/abs/2408.09955) — 每智能体生产者-消费者队列；在数千个并发智能体时的两层协调  
- [Matrix](https://arxiv.org/abs/2511.21686) — 以消息队列作为协调基底的去中心化框架  
- [Temporal docs](https://docs.temporal.io/) — 持久化执行的参考工作流引擎  
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 生产实践，包括 rainbow 部署