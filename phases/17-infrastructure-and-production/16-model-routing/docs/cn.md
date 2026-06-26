# 模型路由：作为成本降低原语

> 一个动态代理会评估每个请求（任务类型、token 长度、嵌入相似度、置信度），并先向廉价模型发送简单查询，将复杂请求升级到前沿模型。这也称为模型级联。生产案例显示在相同性能下，美国/英国/欧盟部署可以节省 20–60% 的成本；在高流量 SaaS 上路由效率提高 30% 会转化为六位数的年节省。到 2026 年的背景是 LLM 推理价格每年约下降 ~10x —— 从 2022 年末到 2026 年，GPT-4 级别的 token 价格从 $20/M 降到约 $0.40/M。大部分下降来自更好的服务栈（Phase 17 · 04-09），而非硬件。路由是把这种价格下降转化为利润而不回退产品质量的方式。失败模式是廉价模型漂移：路由把 40% 的流量推给了更弱的模型，推理任务质量下降 3–5%，一个季度内没人注意到。用在线质量指标为路由设门槛，而不仅仅依赖离线评估集。

**Type:** 学习  
**Languages:** Python（标准库，玩具级级联路由器模拟器）  
**Prerequisites:** Phase 17 · 01（托管 LLM 平台），Phase 17 · 19（AI 网关）  
**Time:** ~60 分钟

## 学习目标

- 解释模型级联：先用廉价模型并做置信度检查，置信度低时升级到前沿模型。  
- 列举四个路由信号（任务分类、提示词长度、与已知困难集合的嵌入相似度、首轮的自信度）。  
- 计算目标路由拆分下的期望加权成本与可容忍的质量损失。  
- 说出用于捕捉廉价模型逐渐失效的漂移监控指标（在线质量门）。

## 问题

你的服务在 GPT-5 上每月花费 $80k。分析显示 70% 的查询很简单： “巴黎现在几点？”、“改写这句话”。一款 Haiku 级模型以约 3% 的成本可以完美处理这些。30% 的请求需要 GPT-5 的推理能力 —— 编码、数学、多步规划。

如果你把 70% 路由到廉价模型、30% 路由到昂贵模型，账单大约下降 ~65%，产品质量保持不变。这就是路由。难点在于在不回退质量的前提下构建路由代理。

## 概念

### 四个路由信号

1. **任务分类**：简单 / 复杂 / 代码生成 / 数学 / 聊天。可以是基于规则的分类器、小型 LLM（Haiku 级，$0.25/M），或将嵌入与标注桶做相似度比对。输出：route = cheap / balanced / frontier。

2. **提示词长度**：提示词 >4K token 常常需要前沿模型以保证连贯性。提示词 <500 token 通常不需要。

3. **与已知困难集合的嵌入相似度**：如果查询与某个已知困难桶的嵌入接近（余弦 > 0.88），则直接升级到前沿模型。

4. **首轮的自信度**：先发给廉价模型；如果模型的对数概率（log-probs）显示低置信度，或它拒绝回答，或输出犹豫/回避措辞，则在前沿模型上重试。对约 10% 的流量会增加 P95 延迟，但在另外 90% 的流量上节省 50%+ 成本。

### 三种模式

**Pre-route**（前置路由，先分类）：增加约 5–10ms 的延迟；总体最快。

**Cascade**（级联，先廉价再升级）：中位延迟约 1.2x（廉价运行 + 验证），升级时约 2x。质量下限最好。

**Ensemble route**（集合路由，平行运行并选优）：质量最高、成本最高；仅在关键 A/B 测试时使用。

### 实现

AI 网关（Phase 17 · 19）通常暴露路由能力。LiteLLM 有带回退和成本路由的 `router` 配置。Portkey 有 guard + routing。Kong AI Gateway 支持基于插件的路由。OpenRouter 的模型市场暴露推荐 API。

开源项目：RouteLLM（LMSYS）、Not Diamond（商业）、Prompt Mule。

### 2026 年的价格曲线

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x 更便宜 |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | 新的档位 |

大部分改进来自服务效率 —— Phase 17 · 04-09 的核心经验转化为提供方的成本下降。路由能让你在应用层捕获这些收益，而不必等所有用户都迁移到廉价层。

### 漂移是真正的风险

你的路由把 40% 的流量发给廉价模型。六个月内，任务分布发生变化（用户变得更复杂、问更长的问题）。路由器没察觉，因为其分类器是在第一季度数据上训练的。质量静默下降，没有人强烈抱怨。你在竞品基准中发现被落下。

用在线质量指标为路由设门槛：

- 每条路由的用户点赞 / 踩。  
- 针对每条路由对保留样本（5%）运行自动化 LLM 判定器。  
- 升级率：若级联的上行率 >30%，说明廉价模型被过度路由。  
- 每条路由的拒绝率。

### 应记住的数据

- 2026 年同质量下的路由节省：案例为 20–60%。  
- LLM 价格 2022–2026 年下降：整体约每年 10x。  
- GPT-4 级别 2022 vs 2026：$20/M → $0.40/M。  
- 级联的延迟影响：中位约 1.2x，升级时约 2x（约 10% 的流量会被升级）。

## 使用方法

`code/main.py` 模拟了在混合工作负载下的 pre-route、cascade 和 ensemble。报告加权成本、质量损失和升级率。

## 交付

本课产出 `outputs/skill-router-plan.md`。给定工作负载和质量预算，选定路由模式和信号。

## 练习

1. 运行 `code/main.py`。在什么准确率下限（accuracy floor）级联优于前置路由？  
2. 你的用户群是 30% 企业（复杂查询）、70% 免费层（简单查询）。设计路由拆分。使用哪个在线指标为其设门槛？  
3. 某一路由质量下降 2% 但节省 40%。是否允许上线？这取决于产品 —— 两种立场都要论证。  
4. 基于 OpenAI / Anthropic API 的 logprobs 实现置信度检查。你一开始会设定什么阈值？  
5. 六个月内，升级率从 8% 上升到 22%。诊断三种可能原因并给出每种的修复方案。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | "cost broker" | 每个请求动态选择模型 |
| Model cascade | "cheap-first escalate" | 先运行廉价模型，低置信时回退到前沿模型 |
| Pre-route | "classify first" | 前端分类器；不重新运行模型 |
| Ensemble route | "parallel pick" | 并行运行多个模型，由奖励模型（reward-model）选最佳 |
| Escalation rate | "uprouted %" | 发生升级的级联系统请求的比例 |
| RouteLLM | "LMSYS router" | 开源路由库 |
| Not Diamond | "commercial router" | 商业化模型路由产品 |
| Drift | "cheap creep" | 分布漂移，路由器未能察觉的变化 |
| Online quality gate | "live check" | 对在线流量做自动化 LLM 判定的实时抽样检查 |

## 延伸阅读

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)  
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)  
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)  
- [Not Diamond — model routing](https://www.notdiamond.ai/)  
- [OpenRouter](https://openrouter.ai/) — 多模型网关，带路由原语。