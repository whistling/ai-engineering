# Prompt Caching and Semantic Caching Economics

> **Pricing snapshot dated 2026-04.** 下文的数值声明基于本课发布时截取的厂商费率表；在下游引用前请核对链接文档以获取最新价格。

> 缓存发生在两个层面。L2（提供方级）提示/前缀缓存通过重用注意力 KV 来复用重复前缀 —— Anthropic 的提示缓存文档宣称在长提示上可节省最高 90% 成本并降低 85% 延迟；对 Claude 3.5 Sonnet，缓存读取为 $0.30/M 而新鲜读取为 $3.00/M，TTL 为 5 分钟，并且 1 小时 TTL 选项存在 2x 的写入溢价（docs.anthropic.com，2026-04）。OpenAI 的提示缓存对 ≥1024 token 的提示会自动应用，缓存输入大约比新鲜输入便宜 90%（platform.openai.com，2026-04）；具体的按模型缓存费率取决于实时费率表。L1（应用级）语义缓存在嵌入相似命中时完全跳过 LLM。厂商的“95% 准确率”指的是匹配正确性，而非命中率 —— 报告的生产命中率从 10%（开放式聊天）到 70%（结构化 FAQ）不等；没有厂商发布官方基线，因此把这些视为社区遥测而非保证。生产中的陷阱：并行化会破坏缓存（在第一次缓存写入完成前发出的 N 个并行请求会将开销放大若干倍），而前缀中包含动态内容会完全阻止缓存命中。ProjectDiscovery 报告通过把动态文本移出可缓存前缀，将命中率从 7% 提升到 74%（2025-11）。

**Type:** 学习  
**Languages:** Python（标准库，玩具两层缓存模拟器）  
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang RadixAttention)  
**Time:** ~60 分钟

## Learning Objectives

- 区分 L2 提示/前缀缓存（提供方的 KV 重用）和 L1 语义缓存（在相似提示上绕过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记以及两个 TTL 选项（5 分钟 vs 1 小时）及其价格乘数。
- 根据命中率、提示/响应构成和 token 价格计算预期的月度节省。
- 指出会将账单放大 5–10x 的并行化反模式以及导致命中率崩塌的动态内容反模式。

## The Problem

你为 RAG 服务添加了提示缓存，但账单没有下降。你测得的命中率是 7%。你的提示看似静态，但其实不是 —— 系统提示包含按分钟格式化的当前时间、请求 ID，以及为多样性而随机重排的示例。每个请求都会写入一个新的缓存条目，读取为零。

另外，你的 agent 在每个用户问题上并行运行十个工具调用。所有十个请求在第一次缓存写入完成前都到达提供方。十次写入，零次读取。你的账单变成了“启用缓存”应该达到的 5–10 倍。

缓存是一个协议，而不是一个标志。两个层面，两个不同的失效模式。

## The Concept

### L2 — provider prompt/prefix caching

提供方存储可缓存前缀的注意力 KV，并在下一次匹配该前缀的请求中重用它。你只需支付一次写入成本，读取几乎免费。

**Anthropic (Claude 3.5 / 3.7 / 4 系列)**：在请求中使用显式的 `cache_control` 标记。你标注哪些区块是可缓存的。TTL：5 分钟（写入成本为基础价的 1.25x）或 1 小时（写入成本为基础价的 2x）。缓存读取：Claude 3.5 Sonnet 为 $0.30/M，而新鲜读取为 $3.00/M —— 便宜约 10 倍（docs.anthropic.com，截止 2026-04）。各模型费率不同（Opus/Haiku 单独公布）；始终交叉检查实时定价页面。

**OpenAI**：对 ≥1024 token 的提示自动进行缓存（platform.openai.com，2026-04）。无显式标志。缓存输入相对于新鲜输入大约便宜 10 倍，适用于当前 gpt-4o/gpt-5 的费率表。文档或发行说明未发布官方命中率基线；社区报告在经过精心提示设计后命中率聚集在 30–60% 左右。使用 `usage.cached_tokens` 来度量你自己的情况。

**Google (Gemini)**：通过显式 API 提供上下文缓存；1M-token 的上下文意味着缓存收益更大。

**自托管（vLLM、SGLang）**：Phase 17 · 06 涵盖 RadixAttention —— 在你自己的算力上有相同的模式。

### L1 — app-level semantic caching

在真正调用 LLM 之前，对提示做哈希、生成嵌入，并查找相似的缓存请求（余弦相似度超过阈值，通常为 0.95+）。命中则返回缓存响应。未命中则调用 LLM 并缓存结果。

开源方案：Redis 向量相似性、GPTCache、Qdrant。商业方案：Portkey Cache、Helicone Cache。

厂商的准确率声明指的是返回的缓存响应在语义上适当的频次 —— 而不是你的命中频次。生产中的命中率示例：

- 开放式聊天：10–15%。
- 结构化 FAQ / 支持：40–70%。
- 代码问答：20–30%（小的变体会破坏命中）。
- 语音代理重复提示：50–80%（语音归一化固定集合）。

### The parallelization anti-pattern

你的 agent 并行发起 10 个工具调用。所有 10 个调用都包含相同的 4K-token 系统提示。Anthropic 的缓存写是按请求进行的；第一次缓存写在提供方看到提示后大约 300 ms 完成。请求 2–10 在同一毫秒窗口到达，因而每个都看到缓存未命中。你支付了 10 次写入溢价，读取折扣为 0。

修复方法：使用“顺序首个 + 批发”——先单独发起请求 1，等其缓存被填充后再发起 2–10。会使第一个工具调用增加约 300 ms 延迟，但可节省 5–10x 的费用。

### The dynamic content anti-pattern

你的系统提示看起来像：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每次请求都是唯一的。每次请求都会写入。零命中。

修复方法：把所有真正静态的内容放入可缓存前缀；把动态内容放在缓存边界之后：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 就是通过这种方式把缓存命中率从 7% 提升到 74%，并发布了拆解方法。

### Stack batch + cache for overnight workloads

批量 API（Phase 17 · 15）在 24 小时周转时提供 50% 的折扣。在此之上叠加缓存输入还能再获得约 10x 的折扣。夜间分类、标注和报表生成工作负载，通过层叠这些策略可降至同步未缓存成本的约 10%。

### Numbers you should remember

定价点抓取自 2026-04 的链接厂商文档，并会每几个月漂移 —— 在依赖它们之前请重新检查。

- Anthropic 缓存读取：Claude 3.5 Sonnet 为 $0.30/M，大约比新鲜输入便宜 10 倍（docs.anthropic.com）。
- Anthropic 缓存写入溢价：5 分钟 TTL 为 1.25x，1 小时 TTL 为 2x。
- OpenAI 自动缓存：适用于 ≥1024 token 的提示；在当前费率表上，缓存输入约为新鲜输入的 10%（platform.openai.com）。
- 语义缓存命中率（社区报告）：大约 10%（开放式聊天）；结构化 FAQ 可达 ~70%。不是厂商文档化的基线。
- ProjectDiscovery：通过将动态内容移出前缀，命中率从 7% 提升到 74%（project blog，2025-11）。
- 并行化反模式：当 N 个并行请求错过第一次缓存写入时，典型报告为账单膨胀 5–10x。

## Use It

`code/main.py` 模拟了混合工作负载下的 L1 + L2 缓存。报告命中率、账单，并展示并行化惩罚。

## Ship It

本课会产出 `outputs/skill-cache-auditor.md`。给定提示模板和流量，该审计会检查可缓存性并推荐重构方案。

## Exercises

1. 运行 `code/main.py`。切换并行化标志。账单变化多少？
2. 你的系统提示包含日期。把它移出来。展示前后命中率的数学计算。
3. 在给定你的请求到达率下，计算 1 小时 TTL（2x 写入）与 5 分钟 TTL（1.25x 写入）的盈亏平衡点。
4. 语义缓存在 0.95 阈值时命中 20%。在 0.85 时命中 50%，但你看到不正确的缓存响应。选择合适的阈值并给出理由。
5. 你为每个用户问题并行批处理 10 个子查询。改写以对缓存友好而不增加端到端延迟。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| L2 prompt cache | "prefix cache" | Provider stores KV for repeated prefix |
| `cache_control` | "Anthropic cache marker" | Explicit attribute marking cacheable blocks |
| Cache write premium | "write tax" | Extra cost for first miss-to-cache (1.25x or 2x) |
| L1 semantic cache | "embedding cache" | App-level hash-and-embed before calling LLM |
| GPTCache | "LLM caching lib" | Popular OSS L1 cache library |
| Cache hit rate | "hits / total" | Fraction of requests served from cache |
| Parallelization anti-pattern | "the N-write trap" | N parallel requests miss cache N times |
| Dynamic content trap | "the time-in-prompt trap" | Dynamic bytes in prefix kill hit rate |
| RadixAttention | "intra-replica cache" | SGLang's prefix-cache implementation |

（注：表中术语列保持为原文术语以便与开发/文档中的引用保持一致。）

## Further Reading

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 官方 `cache_control` 语义和 TTL 说明。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动缓存行为及适用性说明。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)