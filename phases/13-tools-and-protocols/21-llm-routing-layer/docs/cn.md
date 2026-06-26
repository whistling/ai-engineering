# LLM 路由层 — LiteLLM、OpenRouter、Portkey

> 供应商锁定很昂贵。不同的工具调用工作负载适合不同模型。路由网关提供统一的 API 表面、重试、故障切换、成本跟踪和护栏。到 2026 年，三种原型占主导：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（2026 年 3 月开源、适用于生产）。本课命名决策标准并演示一个 stdlib 路由网关实现。

**Type:** 学习  
**Languages:** Python（stdlib、路由 + 故障切换 + 成本跟踪）  
**Prerequisites:** Phase 13 · 02（函数调用），Phase 13 · 17（网关）  
**Time:** ~45 分钟

## 学习目标

- 区分自托管、托管和生产级路由选项。  
- 实现一个按优先级顺序在提供者失败时进行重试的回退链。  
- 跟踪跨提供者的每次请求成本和令牌使用。  
- 针对给定的生产约束在 LiteLLM、OpenRouter 和 Portkey 之间做出选择。

## 问题描述

当需要提供者路由时的场景：

1. **成本。** Claude Sonnet 的费用是 Haiku 的 3 倍。对于分流任务（triage），Haiku 足够；对于合成任务，Sonnet 值得使用。按请求路由。  

2. **故障切换。** OpenAI 遭遇小时级故障。所有请求都失败。你希望自动故障切换到 Anthropic 而无需重新部署。  

3. **延迟。** 实时聊天 UI 需要尽快拿到第一个 token（time-to-first-token）。批量摘要则不需要。按延迟 SLA 路由。  

4. **合规。** 欧盟用户必须保持在欧盟区域。按区域路由。  

5. **试验。** 在相同工作负载上对两个模型做 A/B 测试。按测试桶路由。  

为每个集成手动编写这些逻辑很重复。路由网关提供统一的 OpenAI 兼容 API 并处理其余事务。

## 概念

### OpenAI 兼容的代理形状

大家都使用 OpenAI 的格式。路由网关暴露 `/v1/chat/completions`，接受 OpenAI schema，并在内部代理到 Anthropic / Gemini / Cohere / Ollama / 任何后端。客户端无需关心。

### 模型别名

不用在代码中写 `claude-3-5-sonnet-20251022`，你写 `our_smart_model`。网关将别名映射到真实模型。Anthropic 发布 Claude 4 时，你仅在服务端更改别名；代码不需改动。

### 回退链

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关在配置中定义此类规则。重试计入预算，以免回退级联无限增加成本。

### 语义缓存

完全相同或近似相同的提示命中缓存而不是去提供者。对重复的 agent 循环可以节省 30% 到 60%。键基于嵌入；近似相同的提示共享缓存槽。

### 护栏

网关级别：

- **PII 抹除。** 在发送提示前使用正则或基于 ML 的方式处理。  
- **策略违规检测。** 拒绝包含禁止内容的提示。  
- **输出过滤。** 清洗完成功以防泄露。  

Portkey 和 Kong 都内置了有明确立场的护栏。LiteLLM 则把这些功能作为可选项。

### 每密钥速率限制

一个 API key = 一个团队。每密钥预算防止某个团队消耗共享配额。大多数网关支持此功能。

### 自托管与托管的权衡

| Factor | LiteLLM (self-hosted) | OpenRouter (managed) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| Code | 开源，Python | 托管 SaaS | 开源（2026 年 3 月）+ 托管 |
| Setup | 部署代理 | 注册账号 | 二者皆可 |
| Providers | 100+ | 300+ | 100+ |
| Billing | 使用自己的密钥计费 | OpenRouter 积分计费 | 使用自己的密钥计费 |
| Observability | OpenTelemetry | 仪表盘 | 完整 OTel + PII 抹除 |
| Best for | 有 SRE 团队并需要数据主权的团队 | 想快速原型且无需运维的团队 | 需要开箱即用护栏和合规的生产环境 |

当你有 SRE 团队并希望数据主权时，LiteLLM 是首选。想要单一订阅并免运维则选 OpenRouter。需要内置护栏与合规则选 Portkey。

### 成本跟踪

每个请求携带字段 `provider`、`model`、`input_tokens`、`output_tokens`。乘以由网关维护的每模型每 token 价格表。并可按用户 / 团队 / 项目汇总。

### MCP 与路由的结合

网关可以同时路由 LLM 调用和 MCP 采样请求。当采样请求的 modelPreferences 偏好特定模型时，网关会将其翻译到正确的后端。这就是 Phase 13 · 17（MCP 网关）和本课的路由网关有时合并为同一服务的地方。

### 路由策略

- **静态优先级。** 列表中的第一个；失败时回退。  
- **负载均衡。** 轮询或加权。  
- **成本感知。** 在满足延迟 / 质量的模型中选择最便宜的。  
- **延迟感知。** 选择最近 N 分钟内最快的模型。  
- **任务感知。** 提示分类器将编码任务路由到某个模型，将摘要任务路由到另一个模型。

## 使用示例

`code/main.py` 在 ~150 行内实现了一个路由网关：接受 OpenAI 形状的请求，翻译为各提供者的存根调用，运行优先回退链，跟踪每次请求成本，并在输入上执行 PII 抹除。用三种场景运行：正常请求、主提供者宕机触发回退、PII 泄露被抹除拦截。

值得关注的点：

- `ROUTES` dict：别名 -> 按优先级排序的具体提供者列表。  
- 回退循环在 5xx 错误时重试。  
- 成本追踪器将令牌使用乘以每模型费率。  
- PII 抹除器在转发前清理类似 SSN 模式的敏感信息。

## 部署产出

本课会产出 `outputs/skill-routing-config-designer.md`。给定工作负载配置（延迟、成本、合规），该技能选出 LiteLLM / OpenRouter / Portkey 并生成一份路由配置。

## 练习

1. 运行 `code/main.py`。触发故障场景；确认回退落在第二个提供者并且成本正确计入。  

2. 添加语义缓存：对提示做 SHA256，作为查找键；缓存命中则立即返回。测量重复调用时的成本节省。  

3. 添加提示分类器，将以 "code ..." 开头的提示路由到偏向智能性的别名，将以 "summarize ..." 开头的提示路由到偏向速度的别名。  

4. 设计按团队的预算：每个团队有每月消费上限；一旦超额网关拒绝请求。选择执行粒度（按请求或窗口化）。  

5. 并列阅读 LiteLLM、OpenRouter 和 Portkey 文档。分别列出每个项目中另外两个项目所不具备的一项功能。

## 关键术语

| 术语 | 大家怎么说 | 它真正的含义 |
|------|-----------|--------------|
| 路由网关（Routing gateway） | “LLM 代理” | 在多个提供者前的统一 API 层 |
| OpenAI 兼容（OpenAI-compatible） | “使用 OpenAI 的 schema” | 接受 `/v1/chat/completions` 形状并翻译到任意后端 |
| 模型别名（Model alias） | “our_smart_model” | 代码中使用的名字，网关将其映射到具体模型 |
| 回退链（Fallback chain） | “重试列表” | 按顺序尝试的提供者列表 |
| 语义缓存（Semantic caching） | “提示嵌入缓存” | 键是提示的嵌入；近似重复共享缓存命中 |
| 护栏（Guardrails） | “输入/输出过滤” | 抹除 PII、拒绝策略违规 |
| 每密钥速率限制（Per-key rate limit） | “团队预算” | 绑定到 API key 的配额 |
| 成本跟踪（Cost tracking） | “每次请求花费” | 令牌使用 × 模型价格的汇总 |
| LiteLLM | “开源代理” | 可自托管的开源路由网关 |
| OpenRouter | “托管 SaaS” | 托管的网关，采用积分计费 |
| Portkey | “生产选项” | 开源 + 托管，内置护栏与合规功能 |

## 延伸阅读

- [LiteLLM — docs](https://docs.litellm.ai/) — 自托管路由网关  
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — 托管路由 SaaS 快速入门  
- [Portkey — docs](https://portkey.ai/docs) — 带护栏的生产路由方案  
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 决策指南  
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — 供应商调研