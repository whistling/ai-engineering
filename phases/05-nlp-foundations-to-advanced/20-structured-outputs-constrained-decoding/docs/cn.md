# 结构化输出与受限解码

> 请求 LLM 返回 JSON。大多数情况下能得到 JSON。在生产环境中，“大多数”就是问题。受限解码通过在采样前编辑 logits，将“大多数”变为“始终”。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 5 · 17（聊天机器人）, Phase 5 · 19（子词标记化）  
**Time:** ~60 分钟

## 问题

一个分类器向 LLM 提示：“返回 {positive, negative, neutral} 之一。”模型返回了：“The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...”。你的解析器崩溃了。分类器的 F1 为 0.0。

自由生成不是契约，它只是建议。生产系统需要契约。

到 2026 年，存在三层方法。

1. **提示（Prompting）。** 用礼貌的方式要求。“只返回 JSON 对象。”在前沿模型上大约有效 80%，在小模型上更差。
2. **原生结构化输出 API。** OpenAI 的 `response_format`、Anthropic 的 tool use、Gemini 的 JSON 模式。在支持的 schema 上可靠。但被厂商锁定。
3. **受限解码（Constrained decoding）。** 在每个生成步骤修改 logits，使模型无法输出无效 token。通过构造保证 100% 有效。可用于任何本地模型。

本课将建立对这三种方法的直觉，并说明何时选用哪一种。

## 概念

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**受限解码的工作原理。** 在每个生成步骤，LLM 对整个词表（约 10 万个 token）产生一个 logit 向量。一个 *logit 处理器* 位于模型与采样器之间。它根据目标文法的当前位置（JSON Schema、正则、上下文无关文法等）计算哪些 tokens 有效，并将所有无效 token 的 logits 设为负无穷。对剩余 logits 的 softmax 只会将概率质量分配给有效的延续。

2026 年的实现方式：

- **Outlines。** 将 JSON Schema 或正则编译为有限状态机（FSM）。每个 token 都能做 O(1) 的“下一个有效 token”查找。基于 FSM，因此递归 schema 需要展开（flatten）。
- **XGrammar / llguidance。** 上下文无关文法引擎。能处理递归 JSON Schema。解码开销接近零。OpenAI 在 2025 年的结构化输出实现中给了 llguidance 署名。
- **vLLM guided decoding。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，后端可以是 Outlines、XGrammar，或 lm-format-enforcer。
- **Instructor。** 基于 Pydantic 的包装器，适用于任何 LLM。对验证失败进行重试。跨提供商，但不修改 logits —— 它依赖重试 + 结构化输出友好的提示。

### 反直觉的结果

受限解码通常比非受限生成更快。有两个原因。首先，它缩小了下一个 token 的搜索空间。其次，聪明的实现会对强制输出的 token（比如骨架字符串 `{"name": "`）跳过逐字生成（这些字节是确定的）。

### 会让你付出代价的陷阱

字段顺序很重要。把 `answer` 放在 `reasoning` 之前，模型会在进行思考之前就确定答案。JSON 合法，但答案错了。没有验证能发现它。

```json
// 错误
{"answer": "yes", "reasoning": "because ..."}

// 正确
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序是逻辑，不是格式。

## 实现

### 第 1 步：从头实现基于正则的受限生成

参见 `code/main.py` 的独立 FSM 实现。核心思想在 30 行内：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪到目前为止已满足文法的哪些部分。`valid_tokens(state, tokenizer)` 计算哪些词表 token 可以推进 FSM 而不离开可接受路径。

### 第 2 步：使用 Outlines 支持 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

永远无验证错误。FSM 使得无效输出不可达。

### 第 3 步：Instructor 实现跨提供商的 Pydantic 流程

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

机制不同。Instructor 不接触 logits。它将 schema 格式化进提示、解析输出，并在验证失败时重试（默认 3 次）。适用于任意提供商。重试会增加延迟和成本。跨提供商可移植性是其卖点。

### 第 4 步：原生厂商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务器端受限解码。在受支持的 schema 上，可靠性与 Outlines 相当。无需本地模型管理。但会被厂商锁定。

## 常见陷阱

- **递归 schema。** Outlines 会将递归展开到固定深度。树结构输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **超大枚举。** 10,000 项的 enum 编译慢或超时。改用检索器：先预测 top-k 候选项，然后约束到这些候选。
- **文法过于严格。** 强制 `date: "YYYY-MM-DD"` 的正则会导致模型无法为缺失日期输出 `"unknown"`。模型会发明一个日期。允许 `null` 或使用哨兵值。
- **过早承诺。** 见上面的字段顺序陷阱。总是把 reasoning（推理）放在前面。
- **没有 schema 的厂商 JSON 模式。** 纯 JSON 模式只保证 JSON 语法，不保证符合你的用例。始终提供完整的 schema。

## 使用建议

2026 年的技术栈：

| 情况 | 选择 |
|-----------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生厂商结构化输出 |
| 任意提供商，Pydantic 工作流，可容忍重试 | Instructor |
| 本地模型，需要 100% 有效、扁平 schema | Outlines（FSM） |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM guided decoding |
| 可接受重试的批处理 | Instructor + 最便宜的模型 |

## 交付示例

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: 选择结构化输出方法、schema 设计与验证计划。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

给定一个用例（提供商、延迟预算、schema 复杂度、故障容忍度），输出：

1. 机制。选择 原生厂商结构化输出、Instructor 重试、Outlines FSM 或 XGrammar CFG。并给出一句话理由。
2. Schema 设计。字段顺序（先推理，后答案）、针对“未知”的可空字段、枚举 vs 正则、必需字段。
3. 失败策略。最大重试次数、备用模型、优雅的 `null` 处理、分布式外样本拒绝策略。
4. 验证计划。Schema 合规率（目标 100%）、语义有效性（LLM 评估）、字段覆盖率、延迟 p50/p99。

拒绝任何把 `answer` 或 `decision` 放在推理字段之前的设计。拒绝在没有 schema 的情况下使用裸 JSON 模式。对递归 schema，标记为仅限 FSM 的库。
```

## 练习

1. 简单。对一个小型开源模型（例如 Llama-3.2-3B）在不使用受限解码的情况下，针对 `Review(sentiment, confidence, evidence_span)` 进行提示。在 100 条评论上测量能被解析为有效 JSON 的比例。
2. 中等。同一语料在 Outlines JSON 模式下运行。比较合规率、延迟和语义准确性。
3. 困难。为电话号码（`\d{3}-\d{3}-\d{4}`）从头实现一个基于正则的受限解码器。验证在 1000 次采样中 0 个无效输出。

## 关键词

| 术语 | 大家通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Constrained decoding | 强制有效输出 | 在每个生成步骤屏蔽无效 token 的 logits。 |
| Logit processor | 约束器 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的文法表示；O(1) 的“下一个有效 token”查找。 |
| CFG | 上下文无关文法 | 能处理递归的文法；比 FSM 更慢但更有表达力。 |
| Schema field order | 字段顺序重要吗？ | 是 —— 第一个字段会导致承诺；总是把推理放在答案之前。 |
| Guided decoding | vLLM 的命名 | 相同概念，集成在推理服务器中。 |
| JSON mode | OpenAI 的早期版本 | 保证 JSON 语法；不保证匹配你的 schema。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文。  
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 快速的基于 CFG 的受限解码。  
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推理服务器集成。  
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API 参考与注意事项。  
- [Instructor library](https://python.useinstructor.com/) — Pydantic + 跨提供商重试。  
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 对 6 个受限解码框架的基准测试。