# 托管型 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大超大云厂商，三种截然不同的策略。AWS Bedrock 是模型市场——通过一个 API 提供 Claude、Llama、Titan、Stability、Cohere。Azure OpenAI 是与 OpenAI 的独家合作并提供 Provisioned Throughput Units（PTU，预配置吞吐单元）以获得专用容量。Vertex AI 则以 Gemini 为先，主打最佳的长上下文和多模态能力。2026 年 Artificial Analysis 在等效 Llama 3.1 405B 部署上测得 Azure OpenAI 的中位 TTFT（首令牌时间）约为 ~50 ms，而 Bedrock 约为 ~75 ms——PTU 解释了这一差距，因为专用容量优于共享按需容量。决策规则不是“哪个最快”，而是“哪个模型目录和 FinOps 界面更适合我的产品”。本课教你把权衡写下来，用事实而非直觉来选择。

**Type:** 学习  
**Languages:** Python（stdlib，玩具级成本与延迟比较器）  
**Prerequisites:** 阶段 11（LLM Engineering），阶段 13（Tools & Protocols）  
**Time:** ~60 分钟

## 学习目标

- 说出三种平台策略（市场型 vs 独家型 vs Gemini 优先）并将每种策略匹配到产品使用场景。
- 解释 Azure OpenAI 的 Provisioned Throughput Units（PTU）能为你带来什么，以及为什么在 405B 级别上按需的 Bedrock 通常比 Azure 慢约 ~25 ms。
- 绘制每个平台的 FinOps 归因面（Bedrock 的 Application Inference Profiles vs Vertex 的按团队项目 vs Azure 的 scope + PTU 预留）。
- 写出一条“至少两家提供商”政策，并解释为什么在 2026 年单厂商锁定是昂贵的错误。

## 问题背景

你为产品选择了 Claude 3.7 Sonnet。现在需要部署它。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock，或者通过一个网关。直接 API 最简单；Bedrock 增加了 BAA、VPC 终端节点、IAM 和 CloudWatch 的归因能力。网关则增加了故障切换、统一计费和跨提供商的速率限制。

更深层的问题是目录（catalog）。如果你的产品需要同时使用 Claude、Llama 和 Gemini，你无法只从一个地方买到它们，除非那个地方同时是 Bedrock、Vertex 和 Azure OpenAI。各超大云并不可互换——它们分别押注了谁将拥有模型层。

本课映射出三种押注、延迟差异、FinOps 差异以及锁定风险。

## 概念

### 三种策略

**AWS Bedrock**——市场型。包含 Claude（Anthropic）、Llama（Meta）、Titan（AWS 一线）、Stability（图像）、Cohere（嵌入）、Mistral，以及图像和嵌入子目录。一个 API、一个 IAM 界面、一个 CloudWatch 导出。Bedrock 的赌注是客户更看重多样性而非只要单一模型。

**Azure OpenAI**——独家合作。你可以在 Azure 数据中心使用 GPT-4 / 4o / 5 / o-series、DALL·E、Whisper，并在 Azure 上对 OpenAI 模型做微调。"Azure OpenAI Service" 目录中没有非 OpenAI 的模型——那些会进入 Azure AI Foundry（独立产品）。Azure 的赌注是 OpenAI 将继续领先，客户希望对这段特定关系拥有企业级控制。

**Vertex AI**——以 Gemini 为先，其次是其他模型。提供 Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，以及 Model Garden（第三方模型）。Vertex 押注多模态长上下文——1M 令牌的 Gemini 上下文是其差异点。

### 大规模下的延迟差异

Artificial Analysis 运行持续基准测试。在等效 Llama 3.1 405B 的部署（共享按需）上，Azure OpenAI 的中位首令牌时间（TTFT）约为 50 ms；Bedrock 约为 75 ms。差距不是 AWS 的失误，而是容量模型的不同。Azure 销售 PTU（预配置吞吐单元），它为你的租户预留 GPU 容量。Bedrock 的等价物（Provisioned Throughput）也存在，但通常起价约 $21/小时/单元，多数客户仍然使用共享按需。

按需共享容量会与所有其他客户的流量竞争。专用容量则不会。如果你的产品 SLA 要求 P99 TTFT < 100 ms，你要么在 Azure 上购买 PTU，要么购买 Bedrock 的 Provisioned Throughput，或接受默认的波动。

### 预配置吞吐的经济学

Azure PTU：一块预留的推理计算。对于可预测的工作负载，可比按需节省约 70%（上限）。费用按小时固定——即使空闲也要为预留付费。通常在约 40–60% 的持续利用率下达到盈亏平衡。

Bedrock Provisioned Throughput：$21–$50/小时，具体取决于模型和区域。相同的数学模型——盈亏平衡通常在峰值利用的一半左右。需要按月承诺。

Vertex 的预配置容量按 Gemini SKU 出售；定价随模型和地区变化，且公开信息较少。

### FinOps 面（真正的差异点）

**Bedrock 的 Application Inference Profiles** 在市场中提供了最清晰的归因。为 profile 打上 `team`、`product`、`feature` 标签；将所有模型调用路由至该 profile；CloudWatch 会在无需后处理的情况下按 profile 划分成本。该功能在 2025 年加入，仍然是超大云原生中最细粒度的。

**Vertex** 的归因是按团队划分项目并在所有资源上打标签。通常把每个团队建为一个 GCP 项目，在所有资源上打标签，并用 BigQuery Billing Export + DataStudio 做汇总。工作量更大，但 BigQuery 可以对计费数据做任意 SQL 查询。

**Azure** 依赖订阅/资源组 scope 加 tags，并把 PTU 预留作为一级费用对象。标签是从资源组继承的，不是基于每次请求，因此要做每次请求的归因需要 Application Insights 的自定义指标或一个会打标头的网关。

模式是：Bedrock 原生最干净，Vertex 通过 BigQuery 最灵活，Azure 除非做额外埋点否则最难透明化。

### 锁定风险是 2026 年的隐患

当单一模型主导时，将所有业务绑在一家厂商上还可以。但在 2026 年前沿模型每月移动——一个季度是 Claude 3.7，下个季度是 Gemini 2.5，再下个季度是 GPT-5。绑定到一个平台就等于封闭了三分之二的前沿模型。

工作团队采用的模式：对任何产品关键的 LLM 调用至少采用两家提供商。常见搭配是 Bedrock + Azure OpenAI——一个提供 Claude，另一个提供 GPT，之间做故障切换并通过同一网关接入。成本上行可以忽略，因为网关会做最优路由；在发生中断（例如 2025 年 1 月的 Azure OpenAI 事件、AWS us-east-1 故障）时，可用性提升是决定性的。

### 数据驻留、BAA 与受监管行业

Bedrock：多数区域都有 BAA；提供 VPC 终端节点；有护栏（guardrails）。常见的金融科技默认选择。  
Azure OpenAI：支持 HIPAA、SOC 2、ISO 27001；支持欧盟数据驻留；企业受监管的默认选择。  
Vertex：支持 HIPAA、GDPR，按区域提供数据驻留；依赖 Google Cloud 的合规栈。

三者都满足基础合规清单。差异体现在数据保留策略、日志处理以及是否会对你的流量做滥用监控（大多数为默认开启；企业可选择退出）。

### 你应记住的数字

- Azure OpenAI 在等效 Llama 3.1 405B 上的中位 TTFT：~50 ms（有 PTU）。
- Bedrock 按需的中位 TTFT：~75 ms。
- Bedrock Provisioned Throughput：$21–$50/小时/单元。
- Azure PTU 盈亏平衡：约 40–60% 的持续利用率。
- 在高利用率下，PTU 对比按需的最大节省可达 70%。

## 使用指南

`code/main.py` 比较三大平台在合成工作负载下的表现——它对按需与 PTU 的经济学、TTFT 方差和成本归因可观察性建模。运行它可以看到 PTU 在何处划算，以及市场型模型广度何时超过了 TTFT 差距的价值。

## 部署建议

本课会输出 `outputs/skill-managed-platform-picker.md`。给定一个工作负载画像（需要的模型、TTFT SLA、日流量、合规需求），它会推荐主平台、备选平台以及 FinOps 观测/埋点计划。

## 练习

1. 运行 `code/main.py`。对于 70B 级模型，Azure PTU 在何种持续利用率下优于按需？计算盈亏平衡并与宣传的 40–60% 区间比较。  
2. 你的产品需同时使用 Claude 3.7 Sonnet 和 GPT-4o。设计一个两家提供商的部署方案——各自放在哪个超大云，前端网关选什么，故障切换策略如何。  
3. 一家受监管的医疗客户要求 BAA、美国东部数据驻留，并且 P99 TTFT < 100 ms。选择一个平台并用三个具体特性进行论证。  
4. 发现本月 Bedrock 账单在流量未变的情况下暴增 4 倍。没有 Application Inference Profiles 的情况下你如何定位根因？有 Profiles 的情况下需要多长时间？  
5. 阅读 Azure OpenAI 与 Bedrock 的定价页面。对于每月 1 亿令牌的 Claude 工作负载，哪种更便宜——直接调用 Anthropic API、Bedrock 按需，还是 Bedrock 的 Provisioned Throughput？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|----------|
| Bedrock | “AWS 的 LLM 服务” | 覆盖 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | “Azure 的 ChatGPT” | 在 Azure 数据中心提供的 OpenAI 独家模型，带企业级控制 |
| Vertex AI | “Google 的 LLM” | 以 Gemini 为先的平台，并通过 Model Garden 提供第三方模型 |
| PTU | “专用容量” | Provisioned Throughput Unit（预配置吞吐单元）——按小时计价的预留推理 GPU |
| Application Inference Profile | “Bedrock 的打标” | 面向每个产品的成本/使用轮廓，CloudWatch 原生支持 |
| Model Garden | “Vertex 的目录” | Vertex AI 的第三方模型区，独立于 Gemini |
| Two-provider minimum | “LLM 冗余” | 对每条关键 LLM 路径至少跨 ≥2 家超大云运行的策略 |
| BAA | “HIPAA 的 paperwork” | Business Associate Agreement；处理 PHI（受保护健康信息）时所需；三家厂商均提供 |
| Abuse monitoring | “日志审查” | 提供商侧对提示词/输出的安全扫描；企业通常可选择退出 |

## 延伸阅读

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — 官方费率表与 Provisioned Throughput 定价。  
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU 经济学与费率表。  
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 分层与 Model Garden 的附加费用。  
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — 在各提供商间持续更新的延迟与吞吐基准。  
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架。  
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 成本归因机制对比。