# AI Gateways — LiteLLM, Portkey, Kong AI Gateway, Bifrost

> 网关位于你的应用与模型提供方之间。核心功能包括提供方路由、回退、重试、限流、密钥引用、可观测性、护栏。2026 年市场划分：**LiteLLM** 是 MIT 许可证的开源项目，支持 100+ 提供方、兼容 OpenAI，但在约 ~2000 RPS 时会出现瓶颈（8 GB 内存，在已发布基准测试下会出现级联故障）；最适合 Python、<500 RPS、开发/原型阶段。**Portkey** 定位为控制平面（护栏、PII 清洗、越狱检测、审计轨迹），自 2026 年 3 月开源为 Apache 2.0，额外延迟约 20–40 ms，生产层 $49/月。**Kong AI Gateway** 构建于 Kong Gateway 之上 — Kong 在同等 12 CPU 基准上：比 Portkey 快 228%，比 LiteLLM 快 859%；定价 $100/模型/月（Plus tier 最多 5 个模型）；如果你已经采用 Kong，则适合企业级使用。**Bifrost**（Maxim AI）——支持可配置回退策略的自动重试，在 OpenAI 返回 429 时回退到 Anthropic 是一个典型方案。**Cloudflare / Vercel AI Gateways** —— 托管、零运维、基础重试。数据驻留是自托管决策的主要驱动因素；Portkey 和 Kong 位于中间，提供 OSS + 可选托管。

**Type:** 学习  
**Languages:** Python（标准库、简易网关路由模拟器）  
**Prerequisites:** Phase 17 · 01（托管 LLM 平台）、Phase 17 · 16（模型路由）  
**Time:** ~60 分钟

## 学习目标

- 列举六项核心网关功能（路由、回退、重试、限流、密钥、可观测性、护栏）。  
- 将四个 2026 年的网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到其规模上限和适用场景。  
- 引述 Kong 的基准（比 Portkey 快 228%，比 LiteLLM 快 859%）并解释这对 >500 RPS 的重要性。  
- 根据数据驻留和运维预算选择自托管还是托管方案。

## 问题

你的产品同时调用 OpenAI、Anthropic 和一个自托管的 Llama。每个提供方有不同的 SDK、错误模型、速率限制和认证方案。你需要故障转移（例如 OpenAI 返回 429 时尝试 Anthropic）、单一凭据存储、统一的可观测性和按租户的限流。

在应用层重做这些功能会把每个服务与每个提供方耦合。网关层将这些功能合并到一个进程和一个 API（通常兼容 OpenAI）下，然后再扇出到各个提供方。

## 概念

### 六项核心功能

1. **Provider routing（提供方路由）** — 将 OpenAI、Anthropic、Gemini、自托管等放在同一 API 之后。  
2. **Fallback（回退）** — 在 429、5xx 或质量失败时切换到其他提供方。  
3. **Retries（重试）** — 指数退避、有限次尝试。  
4. **Rate limits（限流）** — 按租户、按密钥、按模型的限流。  
5. **Secret references（密钥引用）** — 运行时从 Vault 拉取凭据（绝不在应用中存储）。  
6. **Observability（可观测性）** — OTel + GenAI 属性（Phase 17 · 13）+ 成本归因。  
7. **Guardrails（护栏）** — PII 清洗、越狱检测、允许话题过滤。

### LiteLLM — MIT 开源、Python

- 支持 100+ 提供方，兼容 OpenAI，提供路由配置、回退、基础可观测性。  
- 在 Kong 的基准测试中在约 2000 RPS 时出现瓶颈；占用约 8 GB 内存，在持续负载下会出现级联故障。  
- 最适合：Python 应用、<500 RPS、开发/预发布网关、实验性路由。  
- 成本：OSS 免费；有云端免费层。

### Portkey — 控制平面定位

- 自 2026 年 3 月起采用 Apache 2.0 开源许可。支持护栏、PII 清洗、越狱检测、审计轨迹。  
- 每请求增加约 20–40 ms 的延迟。  
- 生产层 $49/月，包含数据保留与 SLA。  
- 最适合：受监管行业，需要护栏 + 可观测性的一体化解决方案。

### Kong AI Gateway — 面向规模的方案

- 构建在 Kong Gateway（成熟的 API 网关产品，基于 lua+OpenResty）之上。  
- Kong 在等效 12 CPU 的基准测试中：比 Portkey 快 228%，比 LiteLLM 快 859%。  
- 定价：$100/模型/月，Plus 层最多 5 个模型。  
- 最适合：已经使用 Kong 的团队；>1000 RPS；愿意付费授权。

### Bifrost（Maxim AI）

- 支持可配置回退的自动重试。  
- 在 OpenAI 返回 429 时回退到 Anthropic 是一个典型做法。  
- 新兴厂商；商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管、零运维。提供基础的重试和可观测性。  
- 最适合：在 Cloudflare/Vercel 上边缘部署的 JavaScript 应用。  
- 相比 Kong/Portkey，在护栏和限流上功能有限。

### 自托管 vs 托管

数据驻留是决策的关键驱动。医疗和金融通常默认自托管（LiteLLM、Portkey OSS 或 Kong）。消费级产品通常默认托管（Cloudflare AI Gateway）或采用中间层（Portkey 托管）。混合策略：对受监管租户自托管，对其他租户使用托管服务。

### 延迟预算

- LiteLLM：典型额外延迟 5–15 ms。  
- Portkey：额外延迟 20–40 ms。  
- Kong：额外延迟 3–8 ms。  
- Cloudflare/Vercel：额外延迟 1–3 ms（边缘优势）。

网关延迟会直接加到 TTFT（首个标记到达时间）。若 TTFT P99 要求 < 100 ms 的 SLA，推荐 Kong 或 Cloudflare。若 TTFT P99 < 500 ms，任意网关都可满足。

### 限流语义很重要

简单的令牌桶（token-bucket）可应对中等规模。多租户场景需要滑动窗口（sliding-window）+ 突发容忍 + 按租户分层。LiteLLM 内置令牌桶；Kong 提供滑动窗口；Portkey 提供分层限流。

### 网关 + 可观测性 + 路由是可组合的

Phase 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产环境中通常是同一层。选择一个同时覆盖三者的工具，或将它们合理地接合：到 2026 年，大多数部署会把 Helicone（可观测性）或 Portkey（护栏）与 Kong（规模）结合用于分工。

### 你应记住的数字

- LiteLLM：在 ~2000 RPS 时出现瓶颈，内存约 8 GB。  
- Portkey：额外延迟 20–40 ms；自 2026 年 3 月起采用 Apache 2.0。  
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。  
- Kong 定价：$100/模型/月，Plus 层最多 5 个模型。  
- Cloudflare/Vercel：边缘额外延迟 1–3 ms。

## 试用

`code/main.py` 模拟了在注入 429/5xx 故障时，跨三家提供方的路由与回退。会报告延迟、重试率和回退命中率。

## 交付

本课产出 `outputs/skill-gateway-picker.md`。根据规模、运维姿态、合规性、延迟预算，选定一个网关。

## 练习题

1. 运行 `code/main.py`。配置回退顺序 OpenAI→Anthropic→自托管。在提供方错误率为 5% 时，预期命中率是多少？  
2. 你的 SLA 要求 TTFT P99 < 200 ms，基线为 300 ms。哪些网关仍能满足预算？  
3. 一家医疗客户要求自托管 + PII 清洗 + 审计。选择 Portkey OSS 还是 Kong？  
4. 比较 LiteLLM 与 Kong：团队应在多少 RPS 的天花板上迁移？  
5. 为多租户 SaaS 设计一个限流策略：免费层、试用层、付费层。使用令牌桶还是滑动窗口？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "API broker" | 位于应用与提供方之间的进程 |
| LiteLLM | "the MIT one" | Python OSS，支持 100+ 提供方，约 2K RPS 时会瓶颈 |
| Portkey | "guardrails gateway" | 控制平面 + 可观测性，Apache 2.0 开源 |
| Kong AI Gateway | "the scale one" | 构建于 Kong Gateway，基准测试领先 |
| Bifrost | "Maxim's gateway" | 自动重试 + 在 OpenAI 429 时回退到 Anthropic 的方案 |
| Cloudflare AI Gateway | "edge managed" | 边缘部署的托管网关，零运维 |
| PII redaction | "data scrub" | 在发送给模型前用正则/NER 等方式掩码个人敏感信息 |
| Jailbreak detection | "prompt injection guard" | 对用户输入运行分类器以检测提示词注入（越狱） |
| Audit trail | "regulated log" | 每次 LLM 调用的不可变记录 |
| Token-bucket | "simple rate limit" | 令牌桶限流（基于补充令牌） |
| Sliding-window | "precise rate limit" | 滑动窗口限流；更公平、更精确 |

## 延伸阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)  
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)  
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)  
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)  
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)  
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)