# 推理平台经济学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 到 2026 年，推理市场不再只是按 GPU 时间出租。它分化为定制硅（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal）和以 API 为先的市场（Replicate、DeepInfra）。Fireworks 在 2026-05-01 将按 GPU 每小时价格上调 $1，而处理 10T+ tokens/天并获得 $40 亿估值说明基于流量的模型是可行的。Baseten 在 2026 年 1 月完成 $3 亿美元的 E 轮融资，估值 $50 亿。竞争定位规则很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业打磨，Modal 优化 Python 原生开发体验，Replicate 优化多模态覆盖，Anyscale 优化分布式 Python。本课程给你一张矩阵，可以直接交给创始人。

**Type:** 学习  
**Languages:** Python（stdlib，示例每次调用经济学比较器）  
**Prerequisites:** Phase 17 · 01（托管 LLM 平台），Phase 17 · 04（vLLM 服务内部原理）  
**Time:** ~60 分钟

## 学习目标

- 说出三个市场细分（定制硅、GPU 平台、以 API 为先）并将每个厂商映射到对应细分。
- 解释为什么“按标记（per-token）”的 API 定价模型更趋近于服务引擎的成本曲线，而不是硬件本身的成本曲线。
- 计算至少三家厂商在某工作负载下的有效每次请求成本，并解释什么时候按分钟（Baseten、Modal）优于按标记计费。
- 指出哪种平台是给定工作负载（无服务器突发、稳定高吞吐、微调变体、多模态）的默认首选。

## 问题

你评估了托管的超大规模平台，决定需要更窄、更快的提供商 —— 为低延迟选 Fireworks，为模型目录选 Together，为微调定制模型选 Baseten。现在你有六个真实选项，但定价页面无法直接比较。Fireworks 显示 $/M tokens；Baseten 显示 $/minute；Modal 显示 $/second；Replicate 显示 $/prediction。没有对工作负载建模，你无法进行头对头比较。

更糟的是，每个定价页背后的商业模型不同。Fireworks 在共享 GPU 上运行自家引擎 FireAttention；按标记收费反映了他们的利用率曲线。Baseten 提供 Truss + 专用 GPU；按分钟反映排他性。Modal 是真正的 Python 无服务器 —— 按秒计费并有亚秒冷启动。相同的输出（LLM 响应），三种不同的成本函数。

本课程对六家厂商建模并告诉你何时各自胜出。

## 概念

### 三个细分

- 定制硅（Custom silicon） — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。在同一模型上通常比基于 GPU 的集群解码快 5–10 倍。按标记价格更高（Groq 在 2025 年末 Llama-70B 上约 $0.99/M），但在对延迟极度敏感的用例中无可匹敌。Groq 是语音代理和实时翻译的生产首选。
- GPU 平台 — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（2026 年的 H100、H200、B200）或偶尔的 AMD 上。处于“原始 GPU 出租”（RunPod、Lambda）与“超大规模管理服务”（Bedrock）之间的经济层。
- 以 API 为先的市场（API-first marketplaces） — Replicate、DeepInfra、OpenRouter、Fal。目录广泛，按预测或按秒计费，强调快速接入。

### Fireworks —— 延迟优化的 GPU 平台

- FireAttention 引擎（自研）；宣称在等效配置下延迟比 vLLM 低 4x。
- 非交互工作负载有批处理层，约为无服务器费率的 ~50%。
- 微调模型与基础模型同价——相对于对 LoRA 收费的供应商，这是实实在在的差异化。
- 2026 年中：自 2026-05-01 起将按需 GPU 租赁价格提高 $1/小时。大规模可协商量价。
- 财务信号：估值 $40 亿，日处理 10T+ tokens。

### Together —— 目录广度优化

- 200+ 模型，包括上游发布后数日内的开源版本。
- 在等效 LLM 模型上比 Replicate 便宜 50–70% —— “AI 原生云”的定位是由流量和目录驱动。
- 在一个 API 中同时提供推理、微调和训练。

### Baseten —— 企业打磨优化

- Truss 框架：将模型打包、依赖、密钥、服务配置写在一个清单中。
- GPU 范围从 T4 到 B200。按分钟计费，并有合理的冷启动缓解机制。
- SOC 2 Type II、HIPAA 就绪。为金融科技和医疗保健常见的选择。
- 估值 $50 亿，2026 年 1 月 E 轮 $3 亿（CapitalG、IVP、NVIDIA 参与）。

### Modal —— Python 原生优化

- 纯 Python 的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰函数，一条命令部署。
- 按秒计费。冷启动 2–4s（有预热）；小模型 <1s。
- 2025 年 B 轮 $8,700 万，估值 $11 亿。在独立调查中开发者体验评分最高。

### Replicate —— 多模态广度

- 按预测计费。图像、视频、音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS 插件）。
- 在 LLM 按标记费率上竞争力较弱，但在多模态种类上取胜。

### Anyscale —— Ray 原生

- 构建在 Ray 之上；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）。
- 最适合将推理步骤作为更大图中一个节点的分布式 Python 工作负载。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 紧密集成。

### 按标记（per-token）与按分钟（per-minute）——何时哪种更优

按标记适合对延迟不敏感且突发的工作负载 —— 你只为实际使用付费。按分钟适合高且可预测的利用率 —— 当你把 GPU 饱和时可以击败按标记计费。

粗略规则：当某台专用 GPU 的持续利用率超过 ~30% 时，按分钟（Baseten、Modal）开始优于按标记（Fireworks、Together）。低于该值时，按标记胜出，因为你避免为空闲付费。

### 定制引擎才是真正护城河

每个平台都声称优于 vLLM 和 SGLang 的自家引擎。FireAttention、RayTurbo、Baseten 的推理栈。所谓自研引擎很大程度上是营销 —— 诚实的表述是 vLLM + SGLang 占据了生产开源推理的约 80%，平台层的差异化更多体现在开发体验、归因和 SLA 上。

### 你应该记住的数据

- Fireworks GPU 租赁：自 2026-05-01 起每小时加价 $1。
- Fireworks 宣称：在等效配置下延迟比 vLLM 低 4x。
- Together：在 LLM 上比 Replicate 便宜 50–70%。
- Baseten 估值：$50 亿（2026 年 1 月 E 轮 $3 亿）。
- Modal 估值：$11 亿（2025 年 B 轮）。
- 当持续利用率超过 ~30% 时，按分钟优于按标记。

```figure
cost-per-token
```

## 使用方法

`code/main.py` 会在一个合成工作负载上比较这六家厂商的定价模型。它会报告 $/day 和 有效 $/M tokens。运行它来找到按标记与按分钟的盈亏平衡点。

## 交付物

本课会生成 `outputs/skill-inference-platform-picker.md`。给定工作负载配置、SLA 和预算，选择主要的推理平台并指出备选。

## 练习

1. 运行 `code/main.py`。对于一个在单台 H100 上运行的 70B 模型，当持续利用率达到多少时 Baseten（按分钟）会比 Fireworks（按标记）更划算？自行推导交叉点并与经验法则比较。
2. 你的产品同时提供图像生成、聊天和语音转文字。为每种模态选择平台，并命名一种统一它们的网关模式（gateway pattern）。
3. Fireworks 将你主要模型的价格提高 $1/小时。若 40% 的流量转入批处理层（批处理折扣 50%），对混合成本的影响如何建模？
4. 一个受监管客户要求 SOC 2 Type II + HIPAA + 专用 GPU。哪三家平台可行？哪家在 FinOps（财务运维）方面胜出？
5. 比较 Llama 3.1 70B 在 Fireworks serverless、Together on-demand、Baseten dedicated 和 Replicate API 下每 1,000 次预测的成本。在哪种情况下在 10 次/天 时最便宜？在 10,000 次/天 时又如何？

## 关键术语

| 术语 | 常说的话 | 实际含义 |
|------|---------|---------|
| 定制硅（Custom silicon） | “非 GPU 芯片” | Groq LPU、Cerebras WSE、SambaNova RDU —— 针对解码优化 |
| FireAttention | “Fireworks 引擎” | 自研的 attention 内核；宣称比 vLLM 延迟低 4x |
| Truss | “Baseten 的格式” | 模型打包清单；包含依赖、密钥与服务配置 |
| 按标记（Per-token） | “API 定价” | 按消耗的 token 计费；避免为空闲付费 |
| 按分钟（Per-minute） | “专用计费” | 按 GPU 墙钟时间计费；高利用率时占优 |
| 按预测（Per-prediction） | “Replicate 定价” | 按模型调用计费；在图像/视频领域常见 |
| RayTurbo | “Anyscale 引擎” | 基于 Ray 的专有推理引擎；在 Ray 集群上与 vLLM 竞争 |
| 批处理层（Batch tier） | “50% 折扣” | 非交互队列以降低费率处理；Fireworks、OpenAI 常见 |
| 微调按基础费率（Fine-tuned at base rate） | “Fireworks LoRA” | 将 LoRA/微调后的请求按基础模型费率计费（差异化点） |

## 拓展阅读

- [Fireworks Pricing](https://fireworks.ai/pricing) — 按标记费率、批处理层、GPU 租赁。  
- [Baseten Pricing](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业层。  
- [Modal Pricing](https://modal.com/pricing) — 按秒 GPU 费率与免费层。  
- [Together AI Pricing](https://www.together.ai/pricing) — 模型目录与按标记费率。  
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo 与托管 Ray 定价。  
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 比较评估。  
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 厂商格局。