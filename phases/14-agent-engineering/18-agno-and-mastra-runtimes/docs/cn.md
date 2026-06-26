# Agno and Mastra: Production Runtimes

> Agno (Python) 和 Mastra (TypeScript) 是 2026 年的生产运行时配对。Agno 目标实现微秒级智能体实例化与无状态的 FastAPI 后端。Mastra 在 Vercel AI SDK 基座上交付智能体、工具、工作流、统一模型路由和复合存储。

**Type:** 学习  
**Languages:** Python, TypeScript  
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)  
**Time:** ~45 分钟

## 学习目标

- 识别 Agno 的性能目标及其适用场景。  
- 说明 Mastra 的三大原语 — Agents、Tools、Workflows — 以及支持的服务器适配器。  
- 解释为什么无状态的会话作用域 FastAPI 后端是推荐的 Agno 生产路径。  
- 针对特定技术栈（以 Python 优先或以 TypeScript 优先）选择 Agno 或 Mastra。

## 问题陈述

LangGraph、AutoGen、CrewAI 是框架化较重的工具。想要“只要智能体循环、在我的运行时里高性能运行”的团队会选择 Agno（Python）或 Mastra（TypeScript）。两者都在一定程度上放弃框架提供的高阶原语，以换取更高的速度和与周边栈更紧密的契合。

## 概念

### Agno

- Python 运行时，前身为 Phi-data。  
- “没有图、链或繁复模式 —— 只有纯 Python。”  
- 文档中的性能目标：约 2μs 智能体实例化，约 3.75 KiB 每个智能体的内存消耗，支持 ~23 个模型提供商。  
- 生产路径：无状态、会话作用域的 FastAPI 后端。每个请求启动一个新的智能体；会话状态存储在数据库中。  
- 原生多模态（文本、图像、音频、视频、文件）和具智能体特性的 RAG。

当你每秒有成千上万短生命周期智能体（如聊天聚合入口、评估流水线）时，这些速度目标就很重要。若智能体每次运行 10 分钟以上，则这些目标重要性降低。

### Mastra

- TypeScript，建立在 Vercel AI SDK 之上。  
- 三大原语：**Agents**、**Tools**（Zod 类型化）和 **Workflows**。  
- 统一模型路由 — 截至 2026 年 3 月，覆盖 94 个提供商的 3,300+ 模型。  
- 复合存储：将内存、工作流、可观测性分别发送到不同后端；建议在大规模可观测性场景下使用 ClickHouse。  
- Apache 2.0 + 源码可见的企业许可（源中含 `ee/` 目录）。  
- 提供 Express、Hono、Fastify、Koa 的服务器适配器；对 Next.js 和 Astro 有一流集成。  
- 搭载 Mastra Studio（localhost:4111）用于调试。  
- GitHub 超过 22k 星，1.0 版本时每周 npm 下载量 30 万+（2026 年 1 月数据）。

### 定位

两者都不试图成为 LangGraph。它们的竞争点是：

- 语言契合度。Agno 适合以 Python 为主的团队；Mastra 适合以 TypeScript 为主的团队。  
- 运行时易用性。Agno = 近乎零开销；Mastra = 与 Vercel 生态深度集成。  
- 可观测性。两者都可与 Langfuse/Phoenix/Opik（见第 24 课）集成，但 Mastra Studio 是一阶官方工具。

### 何时选择

- **Agno** — Python 后端、需要大量短生命周期智能体、对性能有严格要求、已采用 FastAPI 的团队。  
- **Mastra** — TypeScript 后端、部署在 Next.js / Vercel、需要统一多提供商模型路由、偏好 Zod 类型化工具。  
- **LangGraph**（第 13 课）— 当持久化状态与显式图推理比原始速度更重要时。  
- **OpenAI / Claude Agent SDK** — 当你希望使用某个提供商的产品级形态时（第 16–17 课）。

### 何处容易出问题

- 为了性能而盲目取舍。仅因为听说 Agno “2μs” 很快就选择它，而实际负载每次都是慢速智能体调用，此时开销并非瓶颈。  
- 生态锁定。Mastra 在 Vercel 上的深度集成是优势，但在其他环境可能成为劣势。  
- 企业许可混淆。Mastra 的 `ee/` 目录是源码可见但受限的企业许可，不完全等同 Apache 2.0。若计划 fork，请仔细阅读许可条款。

## 实践构建

本课主要以比较为主 — 没有单一代码工件能同时展示两套框架的全部差异。参见 `code/main.py` 中的并列示例：最小化的“运行一个智能体、流式输出、持久化会话”流程分别以 Agno 风格和 Mastra 风格实现。

运行它：

```
python3 code/main.py
```

将得到两条结构不同但功能等价的执行轨迹。

## 使用场景

- **Agno** — 适用于需要速度且符合 FastAPI 形态的 Python 后端。  
- **Mastra** — 适用于拥有众多模型提供商和工作流原语的 TypeScript 后端。  
- 两者都提供官方的可观测性钩子，并能与 Langfuse 集成。

## 部署建议

`outputs/skill-runtime-picker.md` 会基于技术栈、延迟预算和运维形态选择 Agno、Mastra、LangGraph 或某个提供商 SDK。

## 练习

1. 阅读 Agno 的文档。将 stdlib 的 ReAct 循环（第 01 课）迁移到 Agno。哪些部分消失了？哪些保留了？  
2. 阅读 Mastra 的文档。将相同的循环迁移到 Mastra。工具的类型化（Zod 与无类型）有哪些变化？  
3. 基准测试：在你的栈上测量智能体实例化延迟。Agno 的 2μs 对你的负载是否重要？  
4. 设计一次迁移：如果你在 Python 中运行的是 CrewAI，迁移到 Agno 会有哪些不兼容或中断？  
5. 阅读 Mastra 的 `ee/` 许可条款。哪些限制会影响开源分支？

## 关键术语

| 术语 | 人们如何描述 | 实际含义 |
|------|--------------|----------|
| Agno | "Fast Python agents" | 无状态、会话作用域的智能体运行时 |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + 统一模型路由 |
| Unified Model Router | "Multi-provider access" | 单一客户端访问 94 个提供商的 3,300+ 模型 |
| Composite storage | "Multiple backends" | 将内存 / 工作流 / 可观测性分别写入不同存储 |
| Mastra Studio | "Local debugger" | 用于检查智能体的本地 UI（localhost:4111） |
| Source-available | "Not OSS" | 许可允许查看源码但对商业用途有限制 |

（注：文中采用的标准术语：智能体循环 = agent loop、stateful graphs = 有状态图、护栏 = guardrails、函数调用 = function calling、思维链 = chain-of-thought、少样本 = few-shot、微调 = Fine-tuning、嵌入 = Embeddings、上下文窗口 = Context window、参与者模型 = actor model。）

## 进一步阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — 性能目标、FastAPI 集成  
- [Mastra docs](https://mastra.ai/docs) — 原语、服务器适配器、模型路由  
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图的替代方案  
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 集成中引用的可观测性比较