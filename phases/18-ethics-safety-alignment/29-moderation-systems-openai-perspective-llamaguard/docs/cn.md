# 审核系统 — OpenAI、Perspective、Llama Guard

> 生产环境的审核系统将第12-16课中定义的安全策略落地。OpenAI Moderation API：`omni-moderation-latest`（2024，基于 GPT-4o）在一次调用中对文本和图像进行分类；在多语言测试集上比前一代提高了 42%；响应模式返回 13 个类别的布尔值 — harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用可以隐藏延迟；被标记时显示占位响应。Llama Guard 3/4（第16课）：14 个 MLCommons 危害类别、Code Interpreter 滥用、8 种语言（v3）、多图像（v4）。Perspective API（Google Jigsaw）：在“以 LLM 作为审核器”浪潮之前的毒性评分系统；主要是单维度毒性及其 severe-toxicity/insult/profanity 变体；是内容审核研究的基线。弃用：Azure Content Moderator 于 2024 年 2 月弃用，2027 年 2 月退役，被 Azure AI Content Safety 替代。

**Type:** 构建  
**Languages:** Python (stdlib，三层审核框架)  
**Prerequisites:** 第18阶段 · 第16课（Llama Guard / Garak / PyRIT）  
**Time:** ~60 分钟

## 学习目标

- 描述 OpenAI Moderation API 的类别分类法以及它如何不同于 Llama Guard 3 的 MLCommons 集合。  
- 描述三层审核模式（输入、输出、自定义）并指出每层的一种失效模式。  
- 阐明 Perspective API 作为 LLM 之前时代的基线地位，以及为什么它在研究中仍被使用。  
- 说明 Azure 的弃用时间线。

## 问题背景

第12-16课描述了攻击与防御工具。第29课涵盖将防御在用户接触产品的表面进行部署的审核系统。三层模式是 2026 年的默认配置。

## 概念

### OpenAI Moderation API

`omni-moderation-latest`（2024）。基于 GPT-4o。在一次调用中对文本和图像进行分类。对大多数开发者免费。

类别（响应模式中的 13 个布尔值）：
- harassment、harassment/threatening  
- hate、hate/threatening  
- self-harm、self-harm/intent、self-harm/instructions  
- sexual、sexual/minors  
- violence、violence/graphic  
- illicit、illicit/violent

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余类别为文本专属。

在示例代码 `code/main.py` 的教学框架中，我们为教学简化将 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别合并到其顶级父类别中。生产代码应使用完整的 13 类模式。

在多语言测试集上比前一代 moderation 端点提高了 42%。按类别提供评分；应用按阈值设置策略。

### Llama Guard 3/4

在第16课中介绍。14 个 MLCommons 危害类别（组织方式与 OpenAI 的 13 类布尔响应不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）本地支持多模态，12B 参数。

OpenAI 与 Llama Guard 的分类法存在重叠但也有分歧。OpenAI 将 “illicit” 当作一个广泛类别；Llama Guard 则将“暴力犯罪”和“非暴力犯罪”分开。部署时依据自身政策-分类法的匹配度进行选择。

### Perspective API（Google Jigsaw）

在“以 LLM 作为审核器”浪潮之前（2020 年之前）的毒性评分系统。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。以单维度的主评分（TOXICITY）为主，并有子维度变体。

之所以在内容审核研究中被广泛采用，原因是 API 稳定、文档完善，并且有多年的校准数据。对于现代与 LLM 相近的用例，Llama Guard 或 OpenAI Moderation 通常更合适。

### 三层模式

1. **输入审核。** 在生成前对用户提示进行分类。若被标记则拒绝。延迟：一次分类器调用。  
2. **输出审核。** 在交付前对模型输出进行分类。若被标记则替换为拒绝响应。延迟：生成后的一次分类器调用。  
3. **自定义审核。** 领域特定规则（正则、允许列表、业务策略）。可在输入或输出阶段运行。

这三层按设计为顺序执行：输入审核必须在生成前完成，输出审核在生成后运行。并行性适用于同一层内——在同一文本上并发运行多个分类器（例如 OpenAI Moderation + Llama Guard + Perspective）可以隐藏各分类器的延迟。作为可选优化，可以在输入审核完成前显示占位响应（“请稍候，正在检查...”），并将 token-1 的流式推送延后。标记行为可配置：拒绝、清理、升级至人工审核。

### 失效模式

- **仅输入。** 无法捕获输出的幻觉（第12-14课中编解码攻击可以绕过输入分类器）。  
- **仅输出。** 允许任意输入到达模型；增加成本；将内部推理暴露给攻击者。  
- **仅自定义。** 无法跨类别稳健覆盖；正则脆弱且易被绕过。

分层是默认做法。多重保障。

### Azure 弃用

Azure Content Moderator：于 2024 年 2 月弃用，2027 年 2 月退役。由 Azure AI Content Safety 替代，后者基于 LLM 并与 Azure OpenAI 集成。迁移是 2024–2027 年针对 Azure 部署的现场级项目。

### 在第18阶段中的位置

第16课覆盖了红队场景中的审核工具。第29课覆盖已部署的审核。第30课以当前的双重用途能力证据结束。

## 实践

`code/main.py` 构建了一个三层审核框架：输入审核器（关键词 + 类别得分）、输出审核器（对输出进行相同的分类）、自定义审核器（领域规则）。你可以运行输入并观察每层捕获了什么。

## 交付产物

该课产出 `outputs/skill-moderation-stack.md`。针对给定部署，建议一个审核堆栈配置：输入使用哪个分类器、输出使用哪个分类器、哪些自定义规则、以及对边缘情况的裁定方式。

## 练习

1. 运行 `code/main.py`。对一条良性、边缘和有害输入通过所有三层运行。报告每条在哪一层触发。  
2. 在框架中扩展一个类似 Perspective-API 的毒性评分用于特定类别。比较其阈值行为与类别得分的差异。  
3. 阅读 OpenAI Moderation API 文档和 Llama Guard 3 的类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。识别三个无法清晰映射的类别。  
4. 为代码助手部署（例如 GitHub Copilot）设计一个审核堆栈。识别最相关和最不相关的类别，并提出自定义规则。  
5. Azure Content Moderator 将于 2027 年 2 月退役。制定迁移至 Azure AI Content Safety 的计划。识别迁移中风险最高的要素。

## 关键术语

| 术语 | 人们如何称呼 | 实际含义 |
|------|---------------|---------|
| OpenAI Moderation | "omni-moderation-latest" | 基于 GPT-4o 的 13 类（文本）分类器，支持部分多模态 |
| Perspective API | "Google Jigsaw toxicity" | LLM 之前时代的毒性评分基线 |
| Llama Guard | "MLCommons 14-category" | Meta 的危害分类器（v3：8B 文本，8 种语言；v4：12B 多模态） |
| 输入审核 | "pre-generation filter" | 在模型调用前对用户提示进行分类的过滤器 |
| 输出审核 | "post-generation filter" | 在交付前对模型输出进行分类的过滤器 |
| 自定义审核 | "domain rules" | 部署特定规则（正则、允许列表、策略） |
| 分层审核 | "all three layers" | 标准的生产部署模式 |

## 进一步阅读

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 端点  
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 仓库  
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性评分  
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 的替代方案