# SGLang 和 RadixAttention 在前缀密集型工作负载中的应用

> SGLang 将 KV 缓存视为一等可重用资源，存储在基数树（radix tree）中。vLLM 以 FCFS（先到先服务）调度请求，而 SGLang 的缓存感知调度器优先处理具有更长共享前缀的请求 —— 实际上是一种深度优先的基数树遍历，使得热点分支能够常驻 HBM。以 Llama 3.1 8B 在 ShareGPT 类似的 1K 提示上为例，SGLang 达到约 ~16,200 tok/s，而 vLLM 为 ~12,500，约有 29% 的优势。在前缀密集型的 RAG 工作负载上，这一优势可达 6.4x。在类语音克隆形状的工作负载上，缓存命中率超过 86%。于 2026 年已在 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS 等处部署在 400,000+ GPU。问题在于：当前缀顺序不一致时，那个 6.4x 的数字会消失 —— 顺序是工程师手中的杠杆。

**Type:** 学习  
**Languages:** Python（标准库，玩具级基数树缓存 + 缓存感知调度器）  
**Prerequisites:** 阶段 17 · 04 (vLLM Serving Internals)、阶段 14 (Agentic RAG)  
**Time:** ~75 分钟

## 学习目标

- 绘制 RadixAttention 的示意图：说明前缀如何存储在基数树中，以及 KV 块如何在以相同分支为根的序列间共享。
- 解释缓存感知调度为何比 FCFS 更适合前缀密集的流量。
- 给定前缀缓存命中率和提示长度分布，计算预期的加速比。
- 指出使得 6.4x 数字成立的提示顺序纪律（prompt-ordering discipline），以及导致丧失潜力的情况。

## 问题描述

经典的服务将每个请求的提示视为不透明的字符串。即便 5,000 个 RAG 请求全部以相同的 2,000-token 系统提示加相同的检索前缀开始，vLLM 也会为这 5,000 次都预填充那 2,000-token 前缀。GPU 在重复执行相同的工作。

观察到：在 agentic 和 RAG 工作负载中，提示几乎总是共享长前缀。系统提示、工具 schema、少样本示例、检索头、对话历史 —— 这些都会在请求之间重复。如果你把该前缀的 KV 缓存存储一次并重用，就不需要再次预填充它。

RadixAttention 正是做的这件事。令牌在基数树中被索引；每个节点拥有从根到该节点路径上的令牌序列对应的 KV 块。新请求沿树行走：任何令牌匹配的节点都会重用该节点的 KV 块。预填充成本变成与“新”后缀成比例，而不是与整个提示成比例。

挑战在于调度。如果两个请求共享 2,000-token 前缀，而第三个只共享同一前缀的 200 个 token，你希望把那两个长共享前缀的请求一起服务，这样长前缀可以常驻 HBM。FCFS 则相反 —— 它按到达顺序服务，可能在下一个长前缀请求到来之前就把热点分支驱逐掉。

## 概念

### 作为 KV 索引的基数树

基数树（紧凑前缀树）用于存储令牌序列。每个节点拥有一个令牌区间及为该区间计算出的 KV 块。子节点将序列扩展一个或多个令牌。

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

一个新请求到达，包含系统提示 + "Context: <doc A>" + "Question: Carol"。调度器沿树行走：系统前缀匹配（重用 124 个块），doc-A 分支匹配（重用 31 个块），然后仅为 "Question: Carol" 分配新的块（4 个块）。预填充成本：4 个新令牌块。没有树的话：160 个块。预填充节省约 40 倍。

### 缓存感知调度

如果缓存不停换出，基数树支持的重用就毫无意义。有两个关键策略：

1. **深度优先派发（Depth-first dispatch）**。从队列中选择下一个请求时，优先选择与当前运行集合处于同一分支根下的请求。这可以保持热点分支被固定在内存中。
2. **分支级别的 LRU，而不是块级别**。按分支整体驱逐（从最近最少使用的叶子开始），而不是单个块，这样缓存形态与基数树形态一致。

FCFS 违反了这两点。一个共享 2,000 个令牌的请求在一个共享 50 个令牌的请求之后排队时，2,000-token 分支可能被驱逐以腾出空间给 50-token 请求。

### 需要记住的基准数字

- Llama 3.1 8B、H100、ShareGPT 1K 提示：SGLang ~16,200 tok/s vs vLLM ~12,500（约 29% 优势）。
- 前缀密集型 RAG（相同系统 + 相同文档，问题各异）：SGLang 最多可达 6.4x。
- 语音克隆工作负载：前缀缓存命中率 86.4%。
- SGLang 客户的生产命中率：取决于提示纪律在 50%–99% 之间。
- 2026 年已部署在 400,000+ GPU。

### 顺序陷阱（The ordering gotcha）

6.4x 的数字依赖于一致的提示模板顺序。如果你的客户端在某些请求中构造提示为 `[system, tools, context, history, question]`，而在另一些请求中构造为 `[system, context, tools, history, question]`，基数树就无法发现共享前缀。对人类而言看起来相同的前缀，对基数树来说是两个不同的序列。

工程师的杠杆：你的提示模板就是缓存键。固定顺序。把所有不可变的内容（system、tools、schemas）放在前面。检索上下文放在后面。用户问题放最后。不要把动态内容混入可缓存的前缀中。

研究中的真实案例：将动态内容移出可缓存前缀的一个部署中，缓存命中率从 7% 一次性提升到 74%。

### RadixAttention 的适用场景与局限

适用（胜出）：
- RAG（相同检索前缀、问题变化）。
- Agent（相同工具 schema、查询变化）。
- 带有长系统提示的聊天。
- 带有重复前缀的语音 / 视觉工作负载。

不适用（吞吐回到 vLLM 水平）：
- 单次生成且提示唯一（代码补全、无系统提示的开放式聊天）。
- 每个请求都在前缀中交叉插入唯一内容的动态提示。

### 为何这是调度问题，而不仅仅是内核问题

你可以把 KV 重用实现为内核技巧。SGLang 的洞见在于：只有当调度器保持热点分支常驻时，重用才有价值。朴素的“有就重用”策略在混合负载下会让缓存频繁换出。基于基数树索引的调度器是将内核技巧转化为 29% 生产优势的关键。

### 与 vLLM 的相互作用

这两个系统并非严格的竞争对手。到 2026 年，vLLM 增加了前缀缓存（`--enable-prefix-caching`）和一个缓存感知的路由器（用 Rust 实现的 vLLM Router）。差距因此缩小但并未完全消失 —— SGLang 的整个栈是以基数为优先的；vLLM 是在其上嫁接的。对于被前缀重用主导的工作负载，SGLang 仍是默认选择。对于没有明显前缀模式的一般用途服务，vLLM 则仍然是等同或更优的选择。

```figure
roofline
```

## 使用方法

`code/main.py` 实现了一个玩具级的基数树 KV 缓存以及一个包含两种策略的调度器：FCFS 和 缓存感知。对同一工作负载分别运行两者，报告前缀缓存命中率和吞吐量差异。然后运行一个“打乱顺序”的工作负载以展示 6.4x 如何崩塌。

## 交付（Ship It）

本课会产出 `outputs/skill-radix-scheduler-advisor.md`。给定一个工作负载描述（提示模板形状、检索模式、并发租户数量），它会生成提示顺序处方并给出是否采用 SGLang 的建议（go/no-go）。

## 练习

1. 运行 `code/main.py`。比较同一工作负载下的 FCFS 与缓存感知策略。差异来自哪里 —— 预填充节省、解码节省，还是队列延迟？
2. 修改工作负载，使提示随机置换 `[system, tools, context]`。重新运行。命中率会如何变化？为什么？
3. 计算将一个 2,000-token 系统提示作为一个基数分支常驻在 Llama 3.1 8B 上的 HBM 成本。与没有前缀重用的 16 序列批次进行比较。
4. 阅读 SGLang 的 RadixAttention 论文。用三句话解释为什么在前缀密集负载下，树形 LRU 驱逐优于块形 LRU。
5. 某客户报告只有 8% 的缓存命中率。列出三种可能原因，并为每种原因给出你会执行的诊断方法。

## 术语表

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| RadixAttention | "the SGLang thing" | 将 KV 缓存按基数树索引，使共享前缀可以重用块 |
| Radix tree | "compact trie" | 每个节点拥有一段令牌范围及其对应的 KV 块的树 |
| Cache-aware scheduler | "hot-branch-first" | 优先处理与当前常驻分支共享的请求的调度器 |
| Prefix-cache hit rate | "how much of your prompt was free" | 前缀缓存命中率：从重用的 KV 块提供的提示令牌比例 |
| FCFS | "first-come first-served" | 默认调度，会破坏前缀局部性 |
| Branch-level LRU | "evict the leaf" | 与基数形态匹配的分支级驱逐策略 |
| Prompt template ordering | "the cache key" | 提示组件的顺序决定了基数树能共享的内容 |
| System prompt pinning | "resident prefix" | 将不可变的系统部分固定以避免驱逐抖动 |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) — 源代码与文档。  
- [SGLang documentation](https://sgl-project.github.io/) — RadixAttention 与调度细节。  
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — 设计参考。  
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — 基准数据与调度器原理。  
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM 的类似前缀实现，供比较参考。