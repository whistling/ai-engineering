---
title: Structured Prompting (结构化提示)
type: Concept
last_updated: 2026-07-04
sources:
  - "few_shot_cot_cn.md"
---
# Structured Prompting (结构化提示)

### Core Mechanism
Structured Prompting uses clear structural delimiters like XML tags, Markdown headers, or custom delimiters to prevent the model from confusing instructions, context, and input data.

### Templates
* **XML Tags** (Highly recommended for Claude, works well globally):
```xml
<context>
您正在审查一个拉取请求。
代码库使用 TypeScript 和 React。
</context>
<task>
审查以下差异，查找错误、安全问题和风格违规。
</task>
<diff>
{diff_content}
</diff>
```

* **Markdown Headers**:
```markdown
## 角色
高级安全工程师。
## 任务
分析此 API 端点的漏洞。
## 输入
{api_code}
```

## Outgoing Links
*None*

## References
- [[index|Wiki Index]]
