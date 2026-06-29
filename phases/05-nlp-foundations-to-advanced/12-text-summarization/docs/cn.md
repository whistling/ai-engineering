# 文本摘要

> 抽取式系统告诉你文档里“说了什么”。生成式系统告诉你作者“意思是什么”。不同任务，不同陷阱。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 5 · 02（BoW + TF-IDF），Phase 5 · 11（机器翻译）  
**Time:** ~75 分钟

## 问题

一篇 2,000 字的新闻文章出现在你的信息流中。你需要 120 字来概括它。你可以从文章中挑出三句最重要的句子（抽取式），也可以用自己的话重写内容（生成式）。两者都叫“摘要”，但它们是完全不同的问题。

抽取式摘要是一个排序问题。对每句句子打分，返回得分最高的前 k 条。输出通常是语法正确的，因为是逐字摘取的。风险是遗漏分布在整篇文章中的信息。

生成式摘要是一个生成问题。一个 transformer 在输入条件下产出新文本。输出通常流畅且压缩率高，但可能会杜撰原文中没有的事实。风险是自信地造假（hallucination）。

本课同时构建两者，并展示各自的失败模式。

## 概念

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**抽取式。** 将文章视为一个图，节点是句子，边表示相似度。在图上运行 PageRank（或类似算法）给句子打分，看哪个句子与其它句子连接得最紧密。得分最高的句子就是摘要。典型实现是 **TextRank**（Mihalcea 和 Tarau，2004）。

**生成式。** 对 transformer 编码器-解码器（BART、T5、Pegasus）在文档-摘要对上进行微调。在推理时，模型阅读文档并通过交叉注意力逐步生成摘要。Pegasus 特别使用 gap-sentence 的预训练目标，使其在少量微调下在摘要任务上表现优异。

用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）进行评估。ROUGE-1 和 ROUGE-2 评估一元/二元词的重合，ROUGE-L 评估最长公共子序列（LCS）。数值越高越好，但 ROUGE-L 到 40 被认为“好”，50 被认为“优秀”。论文通常同时报告这三项。使用 `rouge-score` 包。

## 实现

### 步骤 1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

两点值得说明。相似度函数使用对数归一化的词重叠，这是原始 TextRank 的变体。用 TF-IDF 向量的余弦相似度也可以。阻尼系数 0.85 和迭代次数是 PageRank 的默认值。

### 步骤 2：基于 BART 的生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料上已经微调过，可以直接生成新闻风格的摘要。对于其它领域（学术论文、对话、法律），使用对应的 Pegasus 检查点或在目标数据上进行微调。

### 步骤 3：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

务必使用词干化（stemming）。否则“running”和“run”会被计为不同的词，ROUGE 会低估重合度。

### 超越 ROUGE（2026 年的摘要评估）

ROUGE 在过去二十年里一直是主导指标，但在 2026 年单靠它已不够。一项对大规模 NLG 论文的元分析显示：

- **BERTScore**（基于语境化嵌入的相似度）在 2023 年后广泛流行，现在在大多数摘要论文中与 ROUGE 一并报告。
- **BARTScore** 将评估视为生成问题：用预训练的 BART 在给定源文档的情况下计算摘要的似然性作为分数。
- **MoverScore**（在语境化嵌入上的 Earth Mover's Distance）在 2025 年的摘要基准中登顶，因为它比 ROUGE 更好地捕捉语义重合。
- **FactCC** 和基于 QA 的忠实性检测在 2021–2023 年常见，现在常被 **G-Eval**（一个基于 GPT-4 的提示链，使用思维链推理来评分连贯性、一致性、流利度、相关性）取代或补充。
- 当评分细则设计良好时，**G-Eval** 和类似的 LLM-评审方法与人工判断的一致率约为 80%。

生产环境建议：报告 ROUGE-L 以便与历史工作比较，使用 BERTScore 衡量语义重合，使用 G-Eval 评估连贯性与事实性。并用 50–100 条人工标注摘要进行校准。

### 步骤 4：事实性问题

生成式摘要容易出现幻觉。抽取式摘要在幻觉风险上低得多，因为输出是逐字摘取自源文，但如果源句被断章取义、过时或被断句引用，也可能误导读者。这是生产系统在合规相关内容上仍偏好抽取式方法的最主要原因。

需要命名的幻觉类型：

- **实体替换（Entity swap）。** 源文写的是 “John Smith”，摘要写成 “John Brown”。
- **数字漂移（Number drift）。** 源文写 “25,000”，摘要写成 “25 million”。
- **极性翻转（Polarity flip）。** 源文写 “拒绝了要约”，摘要写成 “接受了要约”。
- **事实捏造（Fact invention）。** 源文未提及 CEO，摘要却写 CEO 批准了某事。

有效的评估方法：

- **FactCC。** 一个二分类器，训练目标是判断源句与摘要句之间的蕴涵关系，预测是否为事实性正确。
- **基于 QA 的事实性检测。** 对源文提出问题，答案应在源文中；若摘要支持不同答案则标记为问题。
- **实体级 F1。** 比较源文与摘要中的命名实体。仅出现在摘要中的实体是可疑的。

对于任何面向用户且事实性重要的场景（新闻、医疗、法律、金融），抽取式是更安全的默认选择。生成式需要在流程中加入事实性检查门控。

## 使用建议

2026 年栈：

| 用例 | 推荐 |
|------|------|
| 新闻，3–5 句摘要，英语 | `facebook/bart-large-cnn` |
| 学术论文 | `google/pegasus-pubmed` 或微调过的 T5 |
| 多文档、长文本 | 任何具备 32k+ 上下文窗口的 LLM，使用提示工程（prompting） |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，固有低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

当计算资源不是限制条件时，具备长上下文的 LLM 在 2026 年通常超过专用模型。但代价是成本和可重复性；专用模型能提供更一致的输出。

## 上线交付

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1. 简单。对 5 篇新闻文章运行 TextRank。将前 3 句与参考摘要比较。测量 ROUGE-L。在 CNN/DailyMail 风格的文章上你应该看到 30–45 的 ROUGE-L。
2. 中等。实现实体级事实性检测：从源文和摘要中抽取命名实体（spaCy），计算摘要中包含的源实体召回率和摘要实体相对于源文的精确率。高精确低召回意味着安全但过于简洁；低精确意味着出现幻觉实体。
3. 困难。对 50 篇 CNN/DailyMail 文章比较 BART-large-CNN 与一个 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（按实体 F1）以及每次摘要的成本。记录各自胜出的场景。

## 术语释义

| 术语 | 常说法 | 实际含义 |
|------|--------|---------|
| 抽取式（Extractive） | Pick sentences | 从源文中逐字返回句子。不会杜撰（相对而言）。 |
| 生成式（Abstractive） | Rewrite | 在源文条件下生成新文本。可能会出现幻觉。 |
| ROUGE | Summary metric | 系统输出与参考摘要之间的 N-gram / LCS 重合度。 |
| TextRank | Graph-based extractive | 在句子相似度图上运行 PageRank。 |
| 事实性（Factuality） | Is it right | 摘要陈述是否被源文支持。 |
| 幻觉（Hallucination） | Made-up content | 摘要中源文不支持的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式的经典论文。  
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文。  
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 与 gap-sentence 目标。  
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文。  
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 关于生成式摘要事实性问题的综述论文。