# Memory Blocks and Sleep-Time Compute (Letta)

> MemGPT 在 2024 年演变为 Letta。2026 年的演进增加了两个想法：模型可以直接编辑的离散功能性内存块，以及在主智能体空闲时异步整合内存的睡眠时计算智能体。这就是你如何把记忆扩展到超出单次对话范围的方式。

**Type:** 构建  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 14 · 07（MemGPT）  
**Time:** ~75 分钟

## 学习目标

- 说出 Letta 使用的三层内存（core、recall、archival）及各自角色。
- 解释内存块模式：Human block、Persona block 和用户定义块作为一等类型对象的做法。
- 描述什么是睡眠时计算，为什么它位于关键路径之外，以及为什么它可以运行比主智能体更强的模型。
- 实现一个脚本化的双智能体循环，其中主智能体提供响应，睡眠时智能体在轮次之间整合块。

## 问题

MemGPT（第 07 课）解决了虚拟内存的控制流。随之出现了三个生产问题：

1. **延迟。** 每个内存操作都位于关键路径上。如果智能体在用户等待时必须修剪、摘要或调和，尾延迟会暴涨。
2. **记忆腐烂。** 写入不断累积。被驳斥的事实依然存在。检索被陈旧内容淹没。
3. **结构丢失。** 扁平的归档存储无法表达“Human block 始终在提示中；Persona block 始终在提示中；Task block 在会话间切换”。

Letta（letta.com）是 2026 年的重写。内存块使结构显式；睡眠时计算将整合工作移出关键路径。

## 概念

### 三层

| Tier | 范围 | 存放位置 | 写入者 |
|------|-------|----------------|------------|
| Core | 始终可见 | 主提示（prompt）内部 | Agent 工具调用 + 睡眠时重写 |
| Recall | 会话历史 | 可检索 | 自动轮次日志记录 |
| Archival | 任意事实 | 向量 + KV + 图 | Agent 工具调用 + 睡眠时摄取 |

Core 是 MemGPT 的核心。Recall 是带有被驱逐尾部的会话缓冲区。Archival 是外部存储。这个划分清理了 MemGPT 两层过载的问题。

### 内存块

块（block）是 core 层中一种有类型、持久且可编辑的片段。原始 MemGPT 论文定义了两类：

- **Human block** — 关于用户的事实（姓名、角色、偏好、目标）。
- **Persona block** — 智能体的自我概念（身份、语气、约束）。

Letta 将其泛化为任意用户定义的块：用于当前目标的 `Task` block、用于代码库事实的 `Project` block、用于硬约束的 `Safety` block。每个块都有 `id`、`label`、`value`、`limit`（字符上限）、`description`（让模型知道何时编辑它）。

块可通过工具接口编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — 在块接近上限时对其进行压缩。

### 睡眠时计算

2025 年 Letta 的新增：在后台运行第二个智能体，位于关键路径之外。睡眠时智能体处理会话抄本和代码库上下文，将 `learned_context` 写入共享块，并整合或使归档记录失效。

由此产生的属性：

- **无延迟成本。** 主响应无需等待内存操作。
- **允许更强的模型。** 睡眠时智能体可以使用更昂贵、更慢的模型，因为它不受延迟约束。
- **自然的整合窗口。** 在用户不等待时进行去重、摘要、令与之矛盾的事实失效。

其形态匹配人类的工作方式：你完成任务，睡一觉，长期记忆在夜间稳定下来。

### Letta V1 与原生推理

Letta V1（`letta_v1_agent`，2026）弃用 `send_message`/心跳和内联的 `Thought:` 标记，转向原生推理。Responses API（OpenAI）和带有扩展思考的 Messages API（Anthropic）在单独的通道上输出推理，跨轮次传递（在生产中跨提供商加密）。控制循环仍然是 ReAct。思维痕迹是结构化的，而非提示层面的形态。

### 模式失效的场景

- **块膨胀。** 无限的 `block_append` 会很快触及上限。在会导致超限的写入前，给块接入一个摘要器。
- **静默漂移。** 睡眠时智能体重写了一个块，而主智能体从未注意到。对块进行版本控制并在追踪中呈现差异。
- **中毒整合。** 睡眠时智能体将攻击者可达内容整合进 core。第 27 课对睡眠时表面（surface）也适用。

## 构建它

`code/main.py` 实现了：

- `Block` — id、label、value、limit、description。
- `BlockStore` — CRUD + `near_limit(label)` 帮助函数。
- 两个脚本化的智能体 — `PrimaryAgent` 在轮次中提供响应，`SleepTimeAgent` 在轮次间进行整合。
- 一个追踪显示了一个三轮对话伴随块写入，以及一次睡眠时通过（sleep-time pass）对块进行摘要并使陈旧事实失效。

运行它：

```
python3 code/main.py
```

抄本显示了分工：主轮次快速并产生原始写入；睡眠时处理将进行压缩和清理。

## 使用场景

- **Letta**（letta.com）作为参考实现。可自托管或使用托管云服务。
- **Claude Agent SDK skills** 作为块状知识——skill 是一个命名的、版本化的、可检索的指令块，智能体按需加载。
- **自定义构建** 为希望控制存储后端的团队准备。遵循 Letta API 协议以便日后迁移。

## 发布

`outputs/skill-memory-blocks.md` 生成一个符合 Letta 形态的块系统，带有睡眠时钩子（hooks），可用于任何运行时，包括安全规则和引用（citation）连接。

## 练习

1. 增加一个 `block_summarize` 工具，当 `near_limit` 返回真时用模型生成的摘要替换块值。哪个触发阈值能在最小化摘要调用和避免块溢出之间取得平衡？
2. 在归档中实现睡眠时去重：当两条记录的文本令牌重叠度 >90% 时在睡眠轮次合并为一条。仅在睡眠通过时执行，决不在关键路径上执行。
3. 为块做版本控制。每次写入时记录旧值和差异（diff）。公开 `block_history(label)`，让运维能调试“为什么智能体忘记了 X”。
4. 将睡眠时智能体视为不受信任的写入者。当它触碰到 Persona 或 Safety block 时，要求第二个智能体复核后才提交。
5. 将示例移植到使用 Letta API（`letta_v1_agent`）。块模式有什么变化？原生推理如何改变追踪形态？

## 术语表

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Memory block | “可编辑的提示段” | 有类型的、持久的、LLM 可编辑的 core 内存片段 |
| Human block | “用户记忆” | 关于用户的事实，钉在 core 中 |
| Persona block | “智能体身份” | 自我概念、语气、约束，钉在 core 中 |
| Sleep-time compute | “异步内存工作” | 在关键路径之外进行整合的第二智能体 |
| Core / Recall / Archival | “层级” | 三层内存划分：始终可见 / 会话 / 外部 |
| Block limit | “上限” | 每个块的字符上限；强制摘要 |
| Native reasoning | “思考通道” | 提供者级别的推理输出，而非提示层面的 `Thought:` |
| Learned context | “睡眠输出” | 睡眠时智能体写入共享块的事实 |

（注：文中术语映射采用行业标准翻译，例如提示词工程 -> 提示词工程、RAG -> RAG、Embeddings -> 嵌入、Fine-tuning -> 微调、Context window -> 上下文窗口、few-shot -> 少样本、chain-of-thought -> 思维链、guardrails -> 护栏、function calling -> 函数调用、agent loop -> 智能体循环、stateful graphs -> 有状态图、actor model -> 参与者模型。）

## 延伸阅读

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — 块模式  
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — 异步整合  
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — 原生推理重写  
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — 起源