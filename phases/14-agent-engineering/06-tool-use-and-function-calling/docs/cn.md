# 工具使用与函数调用

> Toolformer（Schick 等，2023）开启了自监督的工具标注。Berkeley Function Calling Leaderboard V4（Patil 等，2025）设定了 2026 年的门槛：40% agentic，30% multi-turn，10% live，10% non-live，10% hallucination。单回合已接近解决。记忆、动态决策和长时程工具链尚未解决。

**Type:** 构建  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 14 · 01 (智能体循环), Phase 13 · 01 (函数调用深入)  
**Time:** ~60 分钟

## 学习目标

- 解释 Toolformer 的自监督训练信号：只有当执行工具能降低下一个 token 的损失时才保留工具注释。
- 说出 BFCL V4 的五个评估类别及每个类别的衡量对象。
- 实现一个带有模式验证、参数强制转换和执行沙箱的 stdlib 工具注册表。
- 诊断 2026 年仍未解决的三大问题：长时程工具串联、动态决策和记忆。

## 问题描述

早期的工具使用问题是：模型能否预测一个正确的函数调用？现代的工具使用问题是：模型能否在 40 步的链路中串联工具，具备记忆、部分可观测性、从工具失败中恢复，并且不产生不存在工具的幻觉？

Toolformer 建立了基线：模型可以通过自监督学习何时调用工具。BFCL V4 定义了 2026 年的评估目标。它们之间的差距就是生产环境中智能体所处的空间。

## 概念

### Toolformer（Schick 等，NeurIPS 2023）

思路：让模型在预训练语料上自我标注候选 API 调用。对每个候选执行该调用；只有当包含工具结果能降低下一个 token 的损失时才保留该标注。然后在过滤后的语料上微调。

覆盖的工具：计算器、问答系统、搜索引擎、翻译、日历。自监督信号完全基于工具是否有助于预测文本——无人工标注。

规模效应：随着模型规模增大，工具使用能力出现。小模型在添加工具注释时可能受损；大模型则获益。这就是为什么到 2026 年前沿模型通常内置强大的工具使用能力，而多数 7B 模型仍需显式的工具使用微调才能可靠。

### Berkeley Function Calling Leaderboard V4（Patil 等，ICML 2025）

BFCL 是 2026 年事实上的评估基准。V4 组成：

- **Agentic (40%)** — 完整的智能体轨迹：记忆、多轮、动态决策。
- **Multi-Turn (30%)** — 带工具链的交互式对话。
- **Live (10%)** — 用户提交的真实提示（分布更难）。
- **Non-Live (10%)** — 合成测试用例。
- **Hallucination (10%)** — 检测何时不该调用工具。

V3 引入了基于状态的评估：在工具序列执行后，检查 API 的实际状态（例如“文件是否被创建？”），而不是仅匹配工具调用的 AST。V4 增加了网络搜索、记忆和格式敏感性类别。

2026 年的关键发现：单回合函数调用已近乎解决。失败集中在记忆（跨回合携带上下文）、动态决策（基于先前结果选择工具）、长时程链路（20+ 步后漂移）以及幻觉检测（在无合适工具时拒绝调用）。

### 工具 schema

每个提供者都有一个 schema。细节不同但结构相同：

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。两者都接受 JSON Schema。描述是关键信息——模型读取这些描述来选择正确的工具。糟糕的工具描述是错误选择工具的头号根本原因。

### 参数验证

不要信任任何工具调用。验证内容包括：

1. **类型强制转换。** 模型可能返回字符串 "5" 而 schema 要求 int。若无歧义则强制转换；若不明确则拒绝。
2. **枚举验证。** 如果 schema 规定 `status in {"open", "closed"}`，而模型输出 `"in_progress"`，应以可描述的错误拒绝。
3. **必填字段。** 缺少必填字段 -> 立即返回错误观察给模型，而不是崩溃。
4. **格式验证。** 日期、邮箱、URL — 使用具体解析器验证，而不是仅用正则。

每次验证失败都应返回结构化的 observation，以便模型能按正确的形状重试。

### 并行工具调用

现代提供者支持在一次 assistant 回合中并行调用多个工具。循环流程：

1. 模型发出 3 个带独立 `tool_use_id` 的工具调用。
2. 运行时执行它们（若相互独立则并行）。
3. 每个结果作为带有对应 `tool_use_id` 的 `tool_result` 块返回。

工程规则：把关联 ID 当作关键负载——换错它们会把错误的结果路由到错误的工具。

### 沙箱

工具执行是沙箱边界。见 Lesson 09 了解详情。简短版本：每个工具都应指定读/写面、网络访问、超时、内存上限。通用的 `run_shell(cmd)` 是危险信号；具体的 `git_status()` 更安全。

```figure
tool-routing
```

## 构建它

`code/main.py` 实现了一个生产形态的工具注册表：

- JSON Schema 子集验证器（仅 stdlib）。
- 带描述、输入 schema、超时和执行器的工具注册。
- 参数强制转换和枚举验证。
- 带关联 ID 的并行工具分发。
- 将错误观察作为结构化字符串返回。

运行它：

```
python3 code/main.py
```

追踪展示了一个小型智能体在一个回合中调用三个工具，其中一个故意传入格式错误的调用被拒绝并返回了描述性错误，模型可以据此采取行动。

## 使用它

每个提供者都有自己的工具 schema —— Anthropic、OpenAI、Gemini、Bedrock。若需多提供者支持，请使用翻译层（OpenAI Agents SDK、Vercel AI SDK、LangChain tool adapter）。如果工具使用对产品至关重要，请在发布前用 BFCL 对你的智能体进行评测。

## 发布它

`outputs/skill-tool-registry.md` 会为给定任务域生成工具目录、schema 和注册表。包括描述质量检查（每个工具的描述是否告诉模型何时使用它？）。

## 练习

1. 添加一个“no-op”工具，允许模型显式拒绝使用任何其他工具。在类似 BFCL 的幻觉测试上进行衡量。
2. 为 int-as-string 和 float-as-string 实现参数强制转换。强制转换在哪些情况下开始掩盖真实错误？
3. 为每个工具添加超时和断路器（在 3 次连续失败后 60 秒内拒绝该工具）。这会改变模型如何恢复吗？
4. 阅读 BFCL V4 描述。选择一个类别（例如“multi-turn”），用你的智能体运行 10 个示例提示。报告通过率。
5. 将 stdlib 验证器移植到 Pydantic 或 Zod。Pydantic/Zod 抓到了 toy 版遗漏了的哪些问题？

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 函数调用 (Function calling) | "工具使用" | 带验证 schema 的结构化输出工具调用 |
| Toolformer | "自监督工具标注" | Schick 2023 — 保留那些其结果能降低下一个 token 损失的工具调用 |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 基准：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination |
| 工具 schema (Tool schema) | "模型的函数签名" | name、description、参数的 JSON Schema |
| `tool_use_id` | "关联 ID" | 将工具调用与其结果绑定；对并行分发至关重要 |
| 幻觉检测 (Hallucination detection) | "知道何时不该调用" | V4 类别：在没有合适工具时拒绝调用 |
| 参数强制转换 (Argument coercion) | "字符串到整型的修复" | 对可预测的 schema 不匹配做窄范围修复；若有歧义则拒绝 |
| 沙箱 (Sandboxing) | "工具执行边界" | 每个工具的读/写面、网络、超时、内存上限 |

## 延伸阅读

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — 自监督工具标注
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 2026 评测基准
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Agent SDK 中的生产工具 schema
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 函数工具类型与 Guardrails