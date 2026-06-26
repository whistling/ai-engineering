# Prompt Caching and Context Caching

> Your system prompt is 4,000 tokens. Your RAG context is 20,000 tokens. You send both with every request. You also pay for both — every time. Prompt caching lets the provider keep that prefix warm on their side and bill you 10% of the normal rate on reuse. Used correctly, it cuts inference cost by 50–90% and first-token latency by 40–85%.

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 11 · 01 (提示词工程), Phase 11 · 05 (上下文工程), Phase 11 · 11 (缓存与成本)
**Time:** ~60 分钟

## 问题

一个编程代理在对话的每一轮都向 Claude 发送相同的 15,000 令牌系统提示。以 $3/M 输入令牌、20 轮为例，单是输入成本就是 $0.90 —— 还没算任何用户的实际消息。把这个乘以每天 10,000 次会话，账单达到 $9,000/天，而这些文本从未改变。

你无法在不损害质量的情况下压缩提示。你也无法避免在每一轮发送它——模型每次都需要它。唯一的办法是停止为提供者已经见过的前缀支付全价。

那个办法就是提示词缓存（prompt caching）。Anthropic 在 2024 年 8 月推出了它（2025 年推出了可扩展到 1 小时的 TTL 变体），OpenAI 在同年晚些时候实现了自动化，Google 在 Gemini 1.5 同时发布了显式的上下文缓存，现在三家都在其前沿模型上把它作为一等公民功能提供。

## 概念

![提示词缓存：写一次，读便宜](../assets/prompt-caching.svg)

**机制。** 当一次请求的前缀与最近的一次请求匹配时，提供者会返回上一次运行的 KV-cache，而不是重新对这些令牌进行编码。第一次写入需要支付一个小的写入溢价，之后每次读取都能享受大幅折扣。

**到 2026 年的三种提供商风格。**

| 提供商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存 |
|--------|----------|----------:|---------:|---------|-----------|
| Anthropic | 在内容块上用显式的 `cache_control` 标记 | 输入费用减少 90% | 加收 25% | 5 分钟（可扩展到 1 小时） | 1,024 令牌（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入费用减少 50% | 无 | 最长 1 小时（best-effort） | 1,024 令牌 |
| Google (Gemini) | 显式的 `CachedContent` API | 按存储计费；读取约为正常输入的 ~25% | 按 令牌·小时 收取存储费 | 用户设置（默认 1 小时） | 4,096 令牌（Flash），32,768（Pro） |

**不变原则。** 三家都只缓存前缀。如果任意一个令牌在请求之间不同，从第一个不同的令牌开始之后的所有内容都算 miss。把稳定的部分放在顶部，把可变的部分放在底部。

### 缓存友好的布局

```
[system prompt]          <-- 缓存此处
[tool definitions]       <-- 缓存此处
[few-shot examples]      <-- 缓存此处
[retrieved documents]    <-- 如果重用则缓存，否则不要
[conversation history]   <-- 缓存到上一个回合
[current user message]   <-- 永远不缓存（每次不同）
```

如果违反顺序——把用户消息放在系统提示上面、把动态检索插在 few-shots 之间——缓存就永远无法命中。

### 收支平衡计算

Anthropic 的 25% 写入溢价意味着一个被缓存的块需要在 TTL 内被读取至少两次才能净省钱。1 次写入 + 1 次读取平均每次请求花费 0.675x（节省 32%）；1 次写入 + 10 次读取平均每次请求花费 0.205x（节省 80%）。经验法则：把你预计在 TTL 内至少会重用 3 次的内容放进缓存。

## 实现

### 步骤 1：使用显式标记的 Anthropic 提示词缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 将该内容块存储 5 分钟。在该窗口内重用会命中；过期后再次写入并重新付费。

**响应的使用字段（usage fields）：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 首次写入按 1.25x 计费
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 按 0.1x 计费
```

在 CI 中检查这两个字段——如果 Across requests `cache_read_input_tokens` 始终为零，说明你的缓存键在漂移。

### 步骤 2：一小时扩展 TTL

对于长时间运行的批处理任务，默认的 5 分钟会在作业之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 会使写入溢价翻倍（写入成本为基线的 1.5x 而不是 1.25x），但在任何在 TTL 内重用前缀超过 5 次的批处理中回本很快。

### 步骤 3：OpenAI 的自动缓存

OpenAI 不需要你做任何配置。任何超过 1,024 令牌并与最近请求匹配的前缀会自动获得 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 长且稳定
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 折扣部分
```

同样适用缓存友好的布局规则。有两件事会让 OpenAI 的缓存失效而不会让 Anthropic 的缓存失效：改变 `user` 字段（作为缓存键的一部分）和工具的重排序。

### 步骤 4：Gemini 的显式上下文缓存

Gemini 将缓存视为你创建并命名的一级对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 按令牌·小时收取存储费用，只要缓存存在就会收费，读取时大约按正常输入费率的 25% 收费。当你在多天内在许多会话之间重用同一个巨大提示时，这是正确的形态。

### 步骤 5：在生产中测量命中率

参见 `code/main.py`，其中有一个模拟的三厂商会计程序，它跟踪写入/读取/未命中次数并计算每 1K 请求的混合成本。以目标命中率为门槛进行部署 —— 大多数生产环境下 Anthropic 在预热后应当看到 >80% 的读取比例。

## 到 2026 年仍会出现的陷阱

- **把动态时间戳放在顶部。** 顶部写着 `"Current time: 2026-04-22 15:30:02"` 会导致每次请求都 miss。把时间戳放到缓存分界点之下。
- **工具重排序。** 以稳定的顺序序列化工具 —— 部署之间的字典重排会破坏所有命中。
- **自由文本的近似重复。** "You are helpful." 与 "You are a helpful assistant." —— 仅 1 字节不同就会导致完全 miss。
- **块太小。** Anthropic 强制最小为 1,024 令牌（Haiku 为 2,048）。更小的块会静默地不被缓存。
- **盲目的成本仪表板。** 将“输入令牌”拆分为已缓存与未缓存。否则流量下降会被误判为缓存命中。

## 使用场景与选择

2026 年的缓存栈：

| 情形 | 选择 |
|------|------|
| 系统提示稳定且大于 10k、很多轮次的代理 | Anthropic 的 `cache_control`，5 分钟 TTL |
| 在 30+ 分钟内的批处理作业重用同一前缀 | Anthropic，`ttl: "1h"` |
| 无自建基础设施的无服务器 GPT-5 端点 | OpenAI 自动缓存（只要你的前缀稳定且足够长） |
| 多天重用大型代码/文档语料库 | Gemini 的显式 `CachedContent` |
| 跨提供商回退方案 | 在各提供商之间保持相同的可缓存前缀布局，以便任意命中都有效 |

将其与语义缓存（Phase 11 · 11）结合用于用户消息层：提示词缓存处理“逐字相同”的重用，语义缓存处理“语义相同”的重用。

## 部署（Ship It）

保存为 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. 简单：对 Claude 运行一个带 5,000 令牌系统提示的 10 回合对话。先不使用 `cache_control`，然后再使用。报告每种情况下的输入令牌账单。
2. 中等：编写一个测试框架，给定一个提示模板和请求日志，计算每个提供商的预期命中率和美元节省（Anthropic 5 分钟、Anthropic 1 小时、OpenAI 自动、Gemini 显式）。
3. 困难：构建一个布局优化器：给定一个提示和一组标记为 `stable=True/False` 的字段，重写提示以在不丢失信息的情况下把单个缓存分界点放在最大的缓存友好位置。在真实的 Anthropic 端点上进行验证。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|---------|--------|
| Prompt caching | "Makes long prompts cheap" | 为匹配的前缀重用提供者侧的 KV-cache；重复输入令牌享受 50–90% 的折扣。 |
| `cache_control` | "The Anthropic marker" | 声明“直到这里为止都是可缓存”的内容块属性；`{"type": "ephemeral"}`。 |
| Cache write | "Paying the premium" | 首次填充缓存的请求；在 Anthropic 上以约 1.25x 的输入费率计费，在 OpenAI 上免费。 |
| Cache read | "The discount" | 匹配前缀的后续请求；在 Anthropic 上按 10% 计费，在 OpenAI 上按 50%，在 Gemini 上约 25%。 |
| TTL | "How long it lives" | 缓存保持热态的秒数；Anthropic 默认 5 分钟（可扩展到 1 小时），OpenAI best-effort 最长 1 小时，Gemini 可由用户设置。 |
| Extended TTL | "1-hour Anthropic cache" | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价为 2x，但对于批量重用通常值得。 |
| Prefix match | "Why my cache missed" | 只有从起始到分界点每个字节完全相同时缓存才会命中。 |
| Context caching (Gemini) | "The explicit one" | Google 的命名、按存储计费的缓存对象；适合对大型语料进行多天重用。 |

## 进一步阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1 小时 TTL、收支平衡表。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 与存储定价。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — 启动文章及延迟数据。
- Phase 11 · 05 (上下文工程) — 在何处切分提示以便缓存落地。
- Phase 11 · 11 (缓存与成本) — 将提示词缓存与用户消息上的语义缓存配对。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — 解释了 KV-cache 内存模型以及为什么缓存前缀比重新计算便宜约 10 倍。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill 是提示词缓存的捷径；解释了为什么缓存命中时 TTFT 大幅下降而 TPOT 不变。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — 提示词缓存与预测性解码、Flash Attention、MQA/GQA 等一起作为降低推理成本的手段；阅读以了解其他三项技术。
