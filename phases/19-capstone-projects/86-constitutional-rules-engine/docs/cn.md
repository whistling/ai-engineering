# Capstone 86 — Constitutional Rules Engine

> 规则由名称、谓词和解释组成。若缺少这三项中的任何一项，那就是一种“感觉”（vibe），而不是规则。

**Type:** 构建  
**Languages:** Python、YAML  
**Prerequisites:** 第18阶段安全课程，第19阶段 Track A 第25-29课  
**Time:** ~90 分钟

## 问题描述

分类器覆盖可识别的失败。规则引擎覆盖契约性（contractual）的那些。一个编写代码助手的团队想要这样的约束：“所有包含代码的回复必须以可运行的代码块或明确的假设结尾。” 一个运行客服机器人的团队希望“所有拒绝回复必须提供下一步建议。” 这些约束不适合自然的分类器目标。它们是关于回复、对话和系统策略的谓词，并且需要对非工程人员可读。

诚实的表示方式是声明式文件。宪章以 YAML 存放在代码旁边，纳入版本控制，并有独立的审查流程。每条规则包含 `name`、`predicate`、`severity` 和 `explanation` 模板。引擎加载文件，对候选输出评估每条规则，并为触发的每条规则返回一个结构化的 `Violation`。本 capstone 中的规则引擎用 `all_of`、`any_of` 和 `not_` 组合谓词，因此单条规则可以表达“如果回复包含代码，则必须以可运行的代码块结束并且不得引用仅限内部使用的库”。

课程的另一半是修订。一个只能阻断的规则引擎只是半成品。一个能提出修复的规则引擎在操作上更有用：助手起草回复，引擎标记违规，修复器生成修订回复，引擎确认修订满足规则。本课提供一个最小修复器（按规则的正则替换）和一个结构化差异（逐行的新增、删除、编辑），用于比较草稿与修订版。

## 概念

```mermaid
flowchart LR
  D[草稿响应] --> RE[规则引擎]
  RE -->|违规| F[修复器]
  F --> R[修订后响应]
  R --> RE2[规则引擎 第二轮检查]
  RE2 -->|判决| OUT[接受或上报]
  D -.->|差异| R
```

一条规则的结构如下

```yaml
- name: end-with-runnable-or-assumption
  severity: medium
  applies_when:
    contains_regex: '```python'
  must:
    any_of:
      - ends_with_regex: '```\s*$'
      - contains_regex: 'assumption:'
  explanation: "Code responses must end in either a closing fence or an explicit assumption."
  fix:
    append_if_missing: "\n\nAssumption: example inputs are valid."
```

谓词是原子性的：`contains_regex`、`not_contains_regex`、`ends_with_regex`、`starts_with_regex`、`max_words`、`min_words`。组合谓词有 `all_of`、`any_of`、`not_`。引擎先评估 `applies_when`；如果规则不适用，则将违规记录为 `not_applicable`。否则引擎评估 `must` 并产生 `pass` 或 `violation`。

严重性有 `low`、`medium`、`high`，与第85课一致。下游关卡（第87课）将 `high` 级别的规则违规视为与分类器给出的 `high` 判定相同：阻断。

修复器是一组声明式操作：`append_if_missing`、`prepend_if_missing`、`replace_regex`。每个操作按规则名称映射到一个变换。修复器故意限制为局部编辑；结构性重写属于单独的拒绝并提供帮助层，这里不涵盖。

差异对比在原始文本和修订文本之间计算。它是一个包含 `Change` 记录的列表，带有 `op`（add、remove、edit）和相关文本。下游关卡可以记录差异，以便人工审查者随时间评估修复器的行为。

## 构建说明

`code/rules.yml` 保存宪章。`code/main.py` 中的加载器接受 YAML 文件（当 PyYAML 可用时）或 JSON 文件（内置）。课程随附的 `rules.yml` 将被课程测试通过这两种路径解析。`code/main.py` 定义了 `Engine` 和 `Fixer` 类以及一个 `diff` 函数。组合谓词以递归方式评估，并在 `any_of` 上进行短路（short-circuiting）。

随课提供的宪章包括：

- `no-empty-refusal`（medium）- 拒绝回复必须包含建议或重定向之一
- `end-with-runnable-or-assumption`（medium）- 代码回复必须正确结束
- `no-pii-in-examples`（high）- 示例数据不得包含邮箱或电话格式
- `cite-when-asserting-fact`（low）- 以 "According to" 开头的行必须包含括号内引用
- `no-internal-library-leak`（high）- 输出中不得出现 `internal-only` 或 `policybot-internal` 这些词
- `bounded-length`（low）- 回复长度不得超过 800 字

## 使用方法

运行： `python3 main.py`。演示把三个草稿回复运行通过引擎，打印违规项，运行修复器，打印差异，并写出 `outputs/rules_report.json`。其中一个测试用例在草稿中没有代码块，报告显示该规则为 `not_applicable`，以便团队看到引擎明确地评估了该规则。

## 发布

`outputs/skill-constitutional-rules-engine.md` 记录了规则语法和修复器操作。

## 练习

1. 添加一条规则：当提示中提到安全（safety）时，要求每个回复包含短语 "If this is urgent"。使用组合谓词实现。
2. 将正则修复器替换为带命名插槽的模板修复器，并展示以新设计重写的一条规则。
3. 添加一个度量端点：给定一批草稿，返回按规则的违规率，以便团队查看哪条规则过度触发。

## 术语表

| 术语 | 常用含义 | 精确定义 |
|---|---|---|
| constitution | 模糊的政策文档 | 一个包含规则、谓词、严重性和说明的 YAML 文件 |
| predicate | 检查 | 一个从文本到布尔值的可调用，原子或通过 `all_of`/`any_of`/`not_` 组合 |
| violation | 失败 | 一个结构化记录，包含规则名、严重性、解释和匹配片段 |
| fixer | 修复器 | 类比为模型微调：一个确定性的按规则将草稿映射为修订的变换 |
| diff | 字符串比较 | 一个按 add、remove、edit 分类的结构化操作列表，用于比较草稿与修订版本 |

## 延伸阅读

第87课将该引擎与输入侧检测器和输出侧分类器组合成单一的安全关卡。