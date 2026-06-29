# LLM 评估 — RAGAS、DeepEval、G-Eval

> 精确匹配（Exact-match）和 F1 无法捕捉语义等价。人工复核无法规模化。LLM 作为裁判（LLM-as-judge）是生产环境答案 —— 只要经过足够的校准以信任其数值。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (问答), Phase 5 · 14 (信息检索)
**Time:** ~75 分钟

## 问题

你的 RAG 系统回答： "June 29th, 2007."
金标准参考答案是： "June 29, 2007."
Exact Match 得分 0，F1 约 75%。人工判定得分为 100%。

现在扩大到 10,000 个测试用例。再乘以每一次检索器、分块、提示词或模型的变更。你需要一个能理解语义、在大规模下廉价运行、不对回归撒谎，并能显式暴露正确失败模式的评估器。

到 2026 年，有三套框架覆盖这个问题。

- **RAGAS。** Retrieval-Augmented Generation ASsessment。四个 RAG 指标（faithfulness、answer-relevance、context-precision、context-recall），后端使用 NLI + LLM 裁判。研究支持、轻量级。
- **DeepEval。** 面向 LLM 的 pytest。包含 G-Eval、任务完成度、幻觉、偏见等度量。适合 CI/CD。
- **G-Eval。** 一种方法（也是 DeepEval 的一个度量）：带有思维链的 LLM-as-judge，自定义判定标准，返回 0-1 分。

三者都依赖 LLM 作为裁判。本课旨在建立对该方法及其信任层的直觉。

## 概念

![四个评估维度，LLM 作为裁判架构](../assets/llm-evaluation.svg)

**LLM-as-judge（LLM 作为裁判）。** 用一个 LLM 替代静态度量来根据评分量表为输出打分。对于 (query, context, answer)，提示一个裁判 LLM：“就保真性打 0-1 分。”返回分数。

为何可行：LLM 以极低的成本近似人工判断。以 GPT-4o-mini 为例，约 $0.003 每次评分，能让 1000 个样本的回归评估在 5 美元以内完成。

为何会无声失败：

1. 裁判偏差。裁判偏好更长的答案、偏好来自同一家模型族的答案、偏好与提示风格一致的答案。
2. JSON 解析失败。错误的 JSON → NaN 分数 → 在聚合中被静默排除。RAGAS 用户深有体会。用 try/except + 明确失败模式来保护。
3. 随模型版本漂移。升级裁判会改变所有度量。固定裁判模型与版本。

RAG 的四个维度。

| Metric | Question | Backend |
|--------|----------|---------|
| Faithfulness | 答案中的每一条陈述是否来自检索到的上下文？ | 基于 NLI 的蕴含检测 |
| Answer relevance | 答案是否回应了问题？ | 从答案生成假设问题；与真实问题比较 |
| Context precision | 在检索到的分块中，哪些是相关的？ | LLM 裁判 |
| Context recall | 检索是否返回了所有必要的信息？ | 用 LLM 裁判对金标准答案进行比对 |

**G-Eval。** 定义一个自定义标准：“答案是否引用了正确的来源？”框架会自动展开为带思维链的评估步骤，然后打 0-1 分。适用于 RAGAS 未覆盖的领域特定质量维度。

**校准。** 在信任原始裁判分数之前，务必与人工标签计算相关性。运行 100 个手工标注样本。绘制裁判分数对人工分数的图。计算 Spearman rho。如果 rho < 0.7，则裁判量表需要改进。

## 构建

### 步骤 1：用 NLI 做保真性（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` 是任意可调用对象：prompt str -> generated str。
# 示例：llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案拆成原子陈述。对每条陈述使用 NLI 与检索到的上下文比对。保真性 = 被支持的陈述占比。

### 步骤 2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: 任意实现 .encode(texts, normalize_embeddings=True) -> ndarray 的模型
# 例如：encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的疑问与实际问题不同，相关性就会下降。

### 步骤 3：G-Eval 自定义度量

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

evaluation_steps 即评分量表。显式步骤比隐式的 “打 0-1 分” 提示更稳定。

### 步骤 4：CI 门控

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件发布。每次 PR 运行。在回归时阻止合并。

### 步骤 5：从零实现一个玩具评估

见 `code/main.py`。仅使用标准库的保真性（答案陈述与上下文重叠）和相关性（答案标记与问题标记重叠）近似实现。非生产级，仅展示形态。

## 陷阱

- 无校准。与人工标签 Spearman 相关为 0.3 的裁判是噪声。在发布前要求校准运行。
- 自我评估。用同一 LLM 生成并让其评分会使得分偏高 10-20%。裁判应使用不同模型族。
- 对比顺序偏差。在成对判断时，裁判倾向于偏好第一个选项。总是随机化顺序并对两种顺序都运行。
- 原始聚合掩盖失败。均值 0.85 往往掩盖 5% 的灾难性失败。务必查看底部分位（bottom quantile）。
- 金数据集变坏（Golden dataset rot）。未经版本管理的评测集随时间漂移会破坏纵向比较。每次更改都要打 tag。
- LLM 成本。在大规模下，裁判调用主导成本。使用满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small 是常见选择。

## 使用场景

2026 年栈：

| Use case | Framework |
|---------|-----------|
| RAG 质量监控 | RAGAS（4 个指标） |
| CI/CD 回归门控 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 内的 G-Eval |
| 在线实时流量监控 | RAGAS 的无参考（reference-free）模式 |
| 人工参与抽检 | LangSmith 或 Phoenix 的标注界面 |
| 红队 / 安全评估 | Promptfoo + DeepEval |

典型组合：监控用 RAGAS，CI 用 DeepEval，新增维度用 G-Eval。三个一起跑；它们之间有建设性的分歧。

## 交付内容

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: 设计一个带有校准裁判和 CI 门控的 LLM 评估方案。
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. 简单：在 10 个已知存在幻觉的 RAG 示例上使用 RAGAS。验证保真性（faithfulness）指标能捕捉到每一处幻觉。
2. 中等：手工标注 50 个问答的正确性（0-1）。用 G-Eval 打分。测量裁判与人工的 Spearman rho。
3. 困难：用 DeepEval 构建一个 pytest CI 门控。有意回归检索器，验证门控失败。通过对最低 10% 的阈值检查来添加底部分位告警。

## 关键术语

| Term | 大家如何说 | 它实际是什么意思 |
|------|-----------|------------------|
| LLM-as-judge | 用 LLM 打分 | 根据量表提示一个裁判模型，对输出打 0-1 分。 |
| RAGAS | RAG 的度量库 | 开源的评估框架，包含 4 个无参考的 RAG 指标。 |
| Faithfulness | 答案是否有依据？ | 答案陈述被检索到的上下文所蕴含的分数占比。 |
| Context precision | 检索到的分块是否相关？ | top-K 分块中实际有用的分数占比。 |
| Context recall | 检索是否找回了所有信息？ | 金标准答案的陈述被检索到的分块覆盖的分数占比。 |
| G-Eval | 自定义的 LLM 裁判 | 量表 + 带思维链的评估步骤 + 0-1 分。 |
| Calibration | 信任但要验证 | 裁判分数与人工分数之间的 Spearman 相关性。 |

（注：文中术语采用常用中英文术语映射，例如 RAG、G-Eval、嵌入（Embeddings）、少样本（few-shot）、思维链（chain-of-thought）、函数调用（function calling）、投机性解码（speculative decoding）、位置嵌入（positional embeddings）、自注意力（self-attention）、指令微调（instruction tuning）、分布式训练（distributed training）、模型上下文协议（Model Context Protocol）等。）

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 开源生产栈文档。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 关于偏差、校准与局限性的研究。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 将 RAGAS、DeepEval、Phoenix 等整合的统一框架。