# 长上下文评估 — NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称支持 1000 万 token 的上下文。在 100 万 token 时，8-needle MRCR 的准确率降到 26.3%。宣传的 ≠ 可用的。长上下文评估能告诉你实际能在生产中使用的模型容量。

**Type:** 学习
**Languages:** Python
**Prerequisites:** Phase 5 · 13（问答）, Phase 5 · 23（分块策略）
**Time:** ~60 分钟

## 问题

你有一份 200 页的合同。模型宣称支持 100 万 token 的上下文。你把合同整段贴进去并问：“终止条款是什么？”模型给出了答案 —— 但回答来自封面页，因为终止条款位于 120k tokens 深处，超出了模型实际关注的范围。

这是 2026 年的上下文容量差距。规格表写着 1M 或 10M，但现实往往只有 60–70% 可用，而且“可用”还取决于具体任务。

- 检索（在干草堆里找单根针）：在前沿模型上，直到宣传最大值几乎可以完美完成。
- 多跳 / 聚合：在大多数模型中，超过 ~128k 后性能急剧下降。
- 基于分散事实的推理：是最先失败的任务。

长上下文评估衡量这些维度。本课介绍这些基准、它们各自衡量什么，以及如何为你的领域构建定制的“找针”测试。

## 概念

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023)。** 在长上下文中将一个事实（“魔法词是 菠萝”）放在受控深度处。询问模型去检索它。对深度 × 长度做扫掠。原始的长上下文基准。前沿模型现在在此基准上趋于饱和；这是一个必要但不充分的基线。

**RULER（Nvidia, 2024）。** 包含 13 类任务，分属于 4 个类别：检索（单键 / 多键 / 多值）、多跳追踪（变量追踪）、聚合（常见词频统计）、问答。支持可配置的上下文长度（4k 到 128k+）。能揭示那些在 NIAH 上饱和但在多跳任务上失败的模型。在 2024 年发布时，17 个声称支持 32k+ 的模型中，只有一半在 32k 时保持了质量。

**LongBench v2（2024）。** 包含 503 道选择题，上下文长度从 8k 到 2M 单词不等，涵盖六类任务：单文档 QA、多文档 QA、长上下文的 in-context 学习、长对话、代码仓库、长结构化数据。是评估现实世界长上下文行为的生产级基准。

**MRCR（Multi-Round Coreference Resolution）。** 大规模多轮指代消解。8-needle、24-needle、100-needle 变体。揭示模型在注意力退化前能同时处理多少事实。

**NoLiMa。** “非词汇型针”。针和查询之间没有字面重合；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 将许多文档串联起来，询问任意文档中的问题。测试选择性注意力。

**BABILong。** 在大量无关干扰中嵌入 bAbI 推理链。测试在干草堆中的推理能力，而不仅仅是检索。

### 实际要报告的内容

- **宣传的上下文窗口。** 规格表上的数字。
- **有效检索长度。** 在某个阈值（例如 90%）下通过 NIAH 的长度。
- **有效推理长度。** 在相同阈值下，多跳或聚合任务的通过长度。
- **退化曲线。** 按任务类型绘制准确率随上下文长度变化的曲线。

给你的规格表两个数字：检索有效长度和推理有效长度。通常推理有效长度是宣传窗口的 25–50%。

## 构建方法

### 第 1 步：为你的领域构建定制 NIAH

参见 `code/main.py`。骨架如下：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

在 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k} 上扫掠。绘制热图。这就是你目标模型的 NIAH 卡片。

### 第 2 步：多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像 “三个魔法词是什么？” 这类问题需要检索所有三项。单针成功并不能预测多针成功。

### 第 3 步：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要串联三个赋值。前沿模型在 128k 时的准确率常降至 50–70%。

### 第 4 步：在你的环境中运行 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按类别报告准确率。汇总分数常会掩盖显著的任务级差异。

## 陷阱

- 仅做 NIAH 评估。仅在 1M token 上通过 NIAH 并不意味着多跳任务能通过。始终运行 RULER 或自定义多跳测试。
- 深度采样不均匀。许多实现只在 depth=0.5 测试。请在 0、0.25、0.5、0.75、1.0 上测试 —— “中间丢失”效应是真实存在的。
- 填充文本与针的词汇重叠。如果针与填充共享关键词，检索会变得太简单。使用 NoLiMa 风格的不重叠针。
- 忽视延迟。预填充 1M token 的提示需要 30–120 秒。请同时测量首个 token 的时间（time-to-first-token）和准确率。
- 直接信任厂商自报数据。OpenAI、Google、Anthropic 都会发布自己的分数。请在你的用例上独立重跑。

## 使用场景

2026 年推荐栈：

| Situation | Benchmark |
|-----------|-----------|
| 快速健康检查 | 在 3 个深度 × 3 个长度上做定制 NIAH |
| 生产模型选择 | 在目标长度上运行 RULER（13 个任务） |
| 现实世界 QA 质量 | LongBench v2 的 single-doc-QA 子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话 / 会话 | 在目标长度上运行 MRCR 8-needle |
| 模型升级回归检测 | 固定的本地 NIAH + RULER 测试套件，在每次模型更新时运行 |

生产经验法则：在达到预期长度之前，绝不要仅凭模型卡就信任一个上下文窗口。至少要有 NIAH + 一个推理任务的测试。

## 上线交付

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习

1. 简单。构建一个 3 个深度（0.25、0.5、0.75）× 3 个长度（1k、4k、16k）的 NIAH。在任意模型上运行。将通过率绘制为 3×3 热图。
2. 中等。添加一个 3-needle 变体。在每个长度上测量能否检索全部 3 项。与同一长度下的单针通过率比较。
3. 困难。构建一个嵌入在 64k 填充中的 3 跳变量追踪任务（X1 → X2 → X3）。在 3 个前沿模型上测量准确率。报告每个模型的有效推理长度。

## 术语表

| 术语 | 常说法 | 实际含义 |
|------|--------|---------|
| NIAH | Needle in haystack | 在填充文本中种下一条事实，询问模型检索它。 |
| RULER | NIAH 的加强版 | 覆盖检索 / 多跳 / 聚合 / QA 的 13 种任务类型。 |
| Effective context | 实际容量 | 在阈值之上仍保持准确率的长度。 |
| Lost in the middle | 深度偏差 | 模型对长输入中间部分的内容关注不足。 |
| Multi-needle | 多条事实同时存在 | 多个植入点；考察注意力切换能力，而不仅是单次检索。 |
| MRCR | 多轮指代 | 8、24 或 100 针的指代测试；揭示注意力饱和。 |
| NoLiMa | 非词汇型针 | 针与查询之间没有字面 token 重合；需要语义推理。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 现实世界的长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的针。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 干草堆中的推理。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏差论文。
