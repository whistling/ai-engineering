# Structured Output — JSON Schema, Pydantic, Zod, Constrained Decoding

> “礼貌地要求模型返回 JSON” 在前沿模型上仍然有 5% 到 15% 的失败率。结构化输出通过受限解码填补了这部分缺口：模型在生成时会被字面上禁止发出会违反模式的 token。OpenAI 的严格模式、Anthropic 的带模式类型的工具使用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 和 Zod 的 `.parse` 是同一思路的五种表面形式。本课将构建模式校验器和严格模式合约，供每条生产级抽取流水线使用。

**Type:** 构建  
**Languages:** Python（stdlib，JSON Schema 2020-12 子集）  
**Prerequisites:** Phase 13 · 02（函数调用深入）  
**Time:** ~75 分钟

## 学习目标

- 使用正确的约束（`enum`、`min`/`max`、`required`、`pattern`）为抽取目标编写 JSON Schema 2020-12。
- 解释为什么严格模式与受限解码相比，“生成后校验”给出的保证不同。
- 区分三种失败模式：解析错误、模式违反、模型拒绝（refusal）。
- 部署带有类型化修复和类型化拒绝处理的抽取流水线。

## 问题描述

一个代理需要把采购订单邮件的自由文本转换为 `{customer, line_items, total_usd}`。有三种方法。

**方法一：通过提示要求返回 JSON。** “以 JSON 回复，字段为 customer、line_items、total_usd。” 在前沿模型上通常能成功 85% 到 95%。但会失败于六类情况：缺少大括号、尾随逗号、类型错误、虚构字段、被令牌限制截断、以及泄露的散文如 "Here is your JSON:"。

**方法二：生成后校验。** 自由生成，解析，基于模式校验，失败则重试。可靠但昂贵 —— 每次重试都要付费，而且截断类 bug 每次都会多出一次额外对话回合成本。

**方法三：受限解码。** 提供方在解码时强制执行模式。无效的 token 会在采样分布中被屏蔽。输出被保证能解析并保证通过校验。失败简化为一种模式：拒绝（模型判断输入无法符合模式）。

每家到 2026 年的前沿提供商都以某种形式实现了方法三。

- **OpenAI。** 使用 `response_format: {type: "json_schema", strict: true}`，如果模型拒绝会在响应中带上 `refusal` 字段。
- **Anthropic。** 在 `tool_use` 输入上有模式强制；`stop_reason: "refusal"` 并不是标志，但以 `end_turn` 且没有工具调用作为信号。
- **Gemini。** 在请求层面支持 `responseSchema`；到 2026 年，Gemini 在部分类型上提供基于 token 的语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 会产生一个结构化的 `RunResult`，其类型为 `InvoiceModel`。
- **Zod（TypeScript）。** 运行时解析器用于校验提供方输出对应 Zod 模式；可配合 OpenAI 的 `beta.chat.completions.parse` 使用。

共同点：定义一次模式，然后端到端强制执行。

## 概念

### JSON Schema 2020-12 —— 通用语法

每个提供方都接受 JSON Schema 2020-12。你最常用的构造：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子模式的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数值）、`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：对每个数组元素应用的子模式。
- `additionalProperties`：`false` 禁止额外字段（默认值随模式/实现而异）。

OpenAI 的严格模式增加了三项要求：每个属性必须列在 `required` 中、到处设置 `additionalProperties: false`，并且不允许未解析的 `$ref`。如果违反这些规则，API 会在请求时返回 400。

### Pydantic，Python 绑定

Pydantic v2 可以通过 `model_json_schema()` 从类结构生成 JSON Schema。Pydantic AI 将此包裹起来，使你写出：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

代理框架会把该模式在边缘翻译成 OpenAI 的严格模式、Anthropic 的 `input_schema` 或 Gemini 的 `responseSchema`。模型的输出会以类型化的 `Invoice` 实例返回。校验错误会抛出带有类型化错误路径的 `ValidationError`。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TypeScript 的等价物。OpenAI 的 Node SDK 提供 `zodResponseFormat(Invoice)`，它会把 Zod 模式翻译成 API 的 JSON Schema 有关负载。

### 拒绝（Refusals）

严格模式无法强制模型一定回答。如果输入无法满足模式（“邮件是一首诗，而不是发票”），模型会发出 `refusal` 字段并说明原因。你的代码必须把这当作一等公民来处理，而不是错误。拒绝也可作为安全信号：当被要求从受保护内容中抽取信用卡号时，模型会返回附带安全理由的拒绝。

### 公开实现的受限解码技术

开源权重实现通常使用三种技术。

1. 语法（文法）级解码（如 `outlines`、`guidance`、`lm-format-enforcer`）：从模式构建一个确定性有限自动机（DFA）；在每一步屏蔽会违反 DFA 的 token。
2. 使用 JSON 解析器进行 logits 屏蔽：以流式 JSON 解析器与模型同步；在每个步骤计算合法的下一个 token 集合并屏蔽其它 token。
3. 带验证器的投机性解码：廉价的草稿模型提出 token，验证器强制执行模式。

商业提供商在后台选择其中一种实现。到 2026 年，短小结构化输出的速度通常优于纯生成，长文本时速度大致相当。

### 三种失败模式

1. 解析错误。输出不是有效的 JSON。在严格模式下不会发生。在非严格提供商上仍然可能发生。
2. 模式违反。输出能解析但违反了模式。在严格模式下不会发生。在非严格模式下常见。
3. 拒绝。模型决定不回答。这必须作为一种类型化结果来处理。

### 重试策略

当你不在严格模式下（Anthropic 的工具使用、非严格 OpenAI、较早版本的 Gemini）时，恢复模式为：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

一次重试通常足够。三次重试可以捕获弱模型的抖动。超过三次通常意味着模式本身有问题：对于某些输入模型无法满足，这时需要修正提示或模式。

### 小模型支持

受限解码同样适用于小模型。一个带有文法强制的 3B 参数开源模型在结构化任务上的表现可能优于一个 70B 参数的未经约束提示模型。这就是结构化输出在生产环境中重要的原因：它把可靠性与模型规模解耦。

## 使用说明

`code/main.py` 提供了一个使用标准库实现的最小 JSON Schema 2020-12 校验器（支持 `type`、`required`、`enum`、`min`/`max`、`pattern`、`items`、`additionalProperties`）。它封装了一个 `Invoice` 模式，并把一个伪 LLM 输出送入校验器，演示解析错误、模式违反和拒绝路径。在生产中可以把伪输出替换为任何提供方的真实响应。

注意事项：

- 校验器会返回带路径和消息的类型化 `[ValidationError]` 列表。这就是你希望在重试提示中显现的形状。
- 拒绝分支不会重试。它会记录并返回类型化的拒绝。Phase 14 · 09 会把拒绝作为安全信号使用。
- 对抗性测试输入触发 `additionalProperties: false` 检查，展示了严格模式如何关上虚构字段的大门。

## 部署成果

本课会生成 `outputs/skill-structured-output-designer.md`。给定一个自由文本的抽取目标（发票、支持工单、简历等），该技能会产出一个兼容严格模式的 JSON Schema 2020-12，以及一个与之对应的 Pydantic 模型，并在代码中留有类型化拒绝和重试处理的桩代码。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 为负数。确认校验器使用 `minimum` 约束拒绝该输入并指向相应路径。

2. 扩展校验器以支持带有鉴别器（discriminator）的 `oneOf`。常见情况：`line_item` 要么是产品要么是服务，由 `kind` 标记。严格模式对此有细微规则；检查 OpenAI 的结构化输出指南以获得细节。

3. 用 Pydantic BaseModel 写出相同的 Invoice 模式，并比较 `model_json_schema()` 的输出与手写模式。找出 Pydantic 默认设置的那一个字段，而手写模式中没有设置。

4. 测量拒绝率。构造十个不可抽取的输入（歌曲歌词、数学证明、空邮件等），用真实提供商的严格模式运行它们。统计拒绝与幻觉输出的数量。这就是你关于拒绝感知重试的实测基线。

5. 通读 OpenAI 的结构化输出指南。找出严格模式明文禁止但普通 JSON Schema 允许的那一个构造。然后设计一个非必要地使用了该构造的模式，再把它重构为严格兼容的形式。

## 术语表

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "The schema spec" | IETF-draft 模式方言，所有现代提供商都支持 |
| Strict mode | "Guaranteed schema" | OpenAI 的一个标志，通过受限解码强制执行模式 |
| Constrained decoding | "Logit masking" | 在解码时屏蔽非法下一个 token 的强制执行 |
| Refusal | "Model declines" | 当输入无法匹配模式时的类型化结果 |
| Parse error | "Invalid JSON" | 输出无法解析为 JSON；严格模式下不可能发生 |
| Schema violation | "Wrong shape" | 能解析但违反了类型/必需字段/枚举/范围等 |
| `additionalProperties: false` | "No extras allowed" | 禁止未知字段；在 OpenAI 严格模式中是必需的 |
| Pydantic BaseModel | "Typed output" | 会导出并校验 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript output type" | 用于提供方输出校验的 TypeScript 运行时模式 |
| Grammar enforcement | "Open-weights constrained decode" | 基于 FSM 的 logits 屏蔽，如 outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式、拒绝与模式要求  
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布文章，解释解码保证  
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 序列化到各提供商的类型化 `output_type` 绑定说明  
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 官方规范发布说明  
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明及严格模式注意事项