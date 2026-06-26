# Agent Observability: Langfuse, Phoenix, Opik

> 到 2026 年，三个开源的智能体可观测性平台占据主导地位。Langfuse (MIT) — 每月 600 万+ 安装，提供追踪 + 提示词管理 + 评估 + 会话回放。Arize Phoenix (Elastic 2.0) — 深度的智能体专用评估、RAG 相关性、OpenInference 自动注入。Comet Opik (Apache 2.0) — 自动化提示词优化、护栏、LLM 作为裁判的幻觉检测。

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 阶段 14 · 23（OTel GenAI）
**Time:** ~45 分钟

## 学习目标

- 说出三个顶级开源智能体可观测性平台及其许可证。
- 区分每个平台最擅长的领域：Langfuse（提示词管理 + 会话）、Phoenix（RAG + 自动注入评测）、Opik（优化 + 护栏）。
- 解释为何 89% 的组织在 2026 年报告已经部署了智能体可观测性。
- 实现一个基于 stdlib 的 trace->仪表盘流水线并带有 LLM 作为裁判的评估。

## 问题是什么

OTel GenAI（第 23 课）给出了模式（schema）。你仍然需要一个平台来摄取 spans、运行评估、存储提示词版本并呈现回归。三家候选者各自强调生命周期的不同部分。

## 概念

### Langfuse (MIT)

- 每月 600 万+ SDK 安装，19k+ GitHub stars。
- 功能：追踪、带版本控制与 playground 的提示词管理、评估（LLM 作为裁判、用户反馈、定制评估）、会话回放。
- 2025 年 6 月：原本商业的模块（LLM 作为裁判、注释队列、提示词实验、Playground）以 MIT 许可证开源。
- 最擅长：端到端可观测性，结合紧密的提示词管理闭环。

### Arize Phoenix (Elastic License 2.0)

- 更深的智能体专用评估：trace 聚类、异常检测、用于 RAG 的检索相关性评估。
- 原生的 OpenInference 自动注入（auto-instrumentation）。
- 可与托管的 Arize AX 配合用于生产。
- 不提供提示词版本控制 —— 定位为一种与更广平台并行的漂移/行为回归工具。
- 最擅长：RAG 相关性、行为漂移、异常检测。

### Comet Opik (Apache 2.0)

- 通过 A/B 实验实现自动化提示词优化。
- 护栏（PII 脱敏、话题约束）。
- LLM 作为裁判的幻觉检测（LLM-judge hallucination detection）。
- Comet 自身测评基准：Opik 的日志 + 评估耗时 23.44 秒，Langfuse 为 327.15 秒（约 14 倍差距）——对厂商基准应持参考性看法。
- 最擅长：优化闭环、自动化实验、护栏执行。

### 行业数据

根据 Maxim（2026 年现场分析）：89% 的组织已经部署了智能体可观测性；质量问题是生产环境的主要障碍（32% 的受访者指出）。

### 如何选择

| Need | Pick |
|------|------|
| All-in-one with prompt management | Langfuse |
| Deep RAG evaluation + drift | Phoenix |
| Automated optimization + guardrails | Opik |
| Open licensing, no ELv2 | Langfuse (MIT) or Opik (Apache 2.0) |
| Datadog / New Relic integration | Any — they all export OTel |

### 这个模式的陷阱

- **没有评估策略。** 只有追踪而没有评估只是昂贵的日志记录。
- **自建的 LLM 作为裁判缺乏落地依据。** CRITIC 模式（第 05 课）适用 —— 裁判需要外部工具来进行事实核验。
- **提示词版本未与 trace 关联。** 当生产回归发生时，你无法通过二分法定位导致问题的提示词版本。

## 构建它

`code/main.py` 实现了一个 stdlib 的 trace 收集器 + LLM 作为裁判的评估器：

- 摄取符合 GenAI 形状的 spans。
- 按会话分组，标记失败运行（触发护栏、低置信度评估）。
- 一个脚本化的 LLM 裁判，按评分量表对智能体响应打分。
- 类似仪表盘的汇总：失败率、主要失败原因、评估分布。

运行命令：

```
python3 code/main.py
```

输出：每个会话的评估分数和失败分类，结果类似 Langfuse/Phoenix/Opik 展示的内容。

## 使用方法

- **Langfuse** 可自托管或使用云；通过 OTel 或他们的 SDK 接入。
- **Arize Phoenix** 可自托管；自动注入 OpenInference。
- **Comet Opik** 可自托管或云端使用；提供自动化优化闭环。
- **Datadog LLM Observability** 适合已经运行 Datadog 的混合运维+ML 团队。

## 部署

`outputs/skill-obs-platform-wiring.md` 选择一个平台，并将追踪 + 评估 + 提示词版本接入现有智能体。

## 练习

1. 导出一周的 OTel traces 到 Langfuse 云（免费层）。哪些会话失败？原因是什么？
2. 为你的领域编写一个 LLM 裁判量表（事实正确性、语气、范围遵从）。在 50 条 trace 上测试。
3. 对比 Langfuse 的提示词版本控制与 Phoenix 的 trace 聚类。哪一个能更快告诉你哪里出了问题？
4. 阅读 Opik 的护栏文档。将一个 PII 脱敏护栏接入你的某次智能体运行。
5. 在你的语料上对三者做基准测试。忽略厂商发布的数字；自己度量。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tracing | "Spans collector" | 摄取 OTel / SDK spans；按会话索引 |
| Prompt management | "Prompt CMS" | 版本化的提示词并与 traces 绑定 |
| LLM-as-judge | "Automated eval" | 由独立的 LLM 根据量表对智能体输出打分（自动化评估） |
| Session replay | "Trace playback" | 对过去运行逐步回放以便调试 |
| RAG relevancy | "Retrieval quality" | 检索到的上下文是否匹配查询 |
| Trace clustering | "Behavioral grouping" | 将相似运行聚类以发现漂移 |
| Guardrail enforcement | "Policy at log time" | 在记录时对内容执行 PII/毒性/范围检查 |

## 深入阅读

- [Langfuse docs](https://langfuse.com/) — 追踪、评估、提示词管理
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — 自动注入、漂移检测
- [Comet Opik](https://www.comet.com/site/products/opik/) — 优化 + 护栏
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 三个平台通用的 schema