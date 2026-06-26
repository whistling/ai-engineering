# Browser Agents and Long-Horizon Web Tasks

> ChatGPT agent（2025 年 7 月）将 Operator 与 Deep Research 合并成一个浏览器/终端代理，并在 BrowseComp 上创下 68.9% 的 SOTA。OpenAI 在 2025 年 8 月 31 日关闭了独立的 Operator——向产品层整合。Anthropic 收购 Vercept 后，将 Claude Sonnet 在 OSWorld 的表现从不足 15% 提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修正了原始 WebArena 中约 11.3 百分点的假阴性率，并发布了 258 个任务的 Hard 子集。数据是真实的；攻击面也是：OpenAI 的应急负责人公开表示，针对浏览器代理的间接提示注入“不是一个可以被完全修补的漏洞”。2025–2026 年已记录的攻击示例包括：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）以及 Perplexity Comet 中的一键劫持。

**Type:** 学习  
**Languages:** Python (标准库, 间接提示注入攻击面模型)  
**Prerequisites:** Phase 15 · 10 (权限模式), Phase 15 · 01 (长视野代理)  
**Time:** ~45 分钟

## 问题描述

浏览器代理是一类长视野代理，它会读取不可信内容并做出有后果的操作。代理访问的每个页面都是用户未直接编写的输入；页面上的每个表单都可能成为潜在的命令通道。2025–2026 年的攻击语料表明这并非假设性问题：Tainted Memories 允许攻击者通过精心构造的页面将恶意指令绑定到代理的记忆中；HashJack 将命令隐藏在代理访问的 URL 片段中；Perplexity Comet 的劫持可以在一次点击中生效。

防御形势令人不安。OpenAI 的应急负责人把隐蔽部分说出来了：间接提示注入“不是一个可以被完全修补的漏洞”。原因在于攻击存在于代理的“读取-行动”边界，这是一个架构上模糊的界面——原则上，模型读取的每个 token 都可能被解释为一条指令。

本课旨在命名该攻击面、介绍基准生态（BrowseComp、OSWorld、WebArena-Verified），并建模一个最小的间接提示注入场景，帮助你在第 14 和第 18 课中推理实际防御方案。

## 概念

### 2026 年的格局（每个系统一句话）

**ChatGPT agent（OpenAI）。** 于 2025 年 7 月推出。统一了 Operator（浏览）和 Deep Research（多小时研究）。独立的 Operator 在 2025 年 8 月 31 日停服。在 BrowseComp 上达到 68.9% 的 SOTA；在 OSWorld 和 WebArena-Verified 上也有强表现。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept，重点放在计算机使用能力上。将 Claude Sonnet 在 OSWorld 的表现从 <15% 推升到 72.5%。Claude Computer Use 以工具 API 的形式发布。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use 集成了计算机使用控制；FSF v3（2026 年 4 月，第 20 课）专门跟踪 ML 研发领域的自治能力。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个广为人知的问题：原始 WebArena 存在约 11.3% 的假阴性（被标记为失败但实际上已解决）。Verified 版本使用人工策划的成功判据重新评分，并增加了 258 个任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| Benchmark | 测量内容 | 时间尺度 |
|---|---:|---|
| BrowseComp | 在开放网络上在时间压力下查找特定事实 | 分钟级 |
| OSWorld | 代理操作完整桌面（鼠标、键盘、终端） | 数十分钟 |
| WebArena-Verified | 在模拟网站上的事务性网页任务 | 分钟级 |
| Hard subset | WebArena-Verified 中带有多页状态转换的任务 | 数十分钟 |

不同的维度。高 BrowseComp 分数说明代理能找到事实；但并不意味着代理能预订机票。OSWorld 的分数更接近“它能在我的桌面上工作吗”。WebArena-Verified 更接近“它能完成一个流程吗”。任何生产决策都需要选择与任务分布匹配的基准。

### 攻击面（命名）

1. 间接提示注入（Indirect prompt injection）。不可信页面内容包含指令；代理读取并执行它们。公开示例：2024 年 Kai Greshake 等人，2025 年 Tainted Memories 论文，2026 年 HashJack（Cato Networks）。
2. URL 片段 / 查询注入（URL fragment / query injection）。被爬取 URL 的 `#fragment` 或查询字符串包含命令。虽然不被可见渲染，但仍在代理的上下文中。
3. 记忆绑定攻击（Memory-binding attacks）。页面指示代理写入持久记忆（参见 Lesson 12 的持久状态）。下次会话时，记忆触发载荷而无需可见触发器。
4. 针对已认证会话的 CSRF 型攻击（CSRF-shaped attacks on authenticated sessions）。Tainted Memories 类别：代理在某处已登录；攻击者页面发出状态变更请求，代理以用户的 cookie 执行。
5. 一键劫持（One-click hijack）。一个视觉上无害的按钮携带后续载荷，代理随即执行。Comet 类攻击。
6. 代理宿主表面的 Content-Security-Policy 孔洞（Content-Security-Policy holes in the agent's host surface）。渲染层和工具层本身可能成为攻击向量；浏览器嵌入到浏览器代理的堆栈攻击面很广。

### 为什么“不能被完全修补”

该攻击与代理能力同构。代理必须读取不可信内容才能完成工作。代理读取的任何内容都可能包含指令；代理执行的任何指令都可能与用户真正的意图不一致。防御（信任边界、分类器、工具白名单、对有后果操作的人类在环审查）能够提高攻击成本并缩小影响范围，但无法完全消除这类攻击。

这一论证模式与 Lob 定理（Lesson 8）相同：代理无法证明下一个 token 是安全的；它只能建立一个系统，使不安全的 tokens 更容易被检测到。

### 实际可交付的防御姿态

- 读/写边界（Read / write boundary）。读取永远不产生后果。写入（提交表单、发布内容、调用有副作用的工具）仅当发起内容来自信任边界外时，需要重新获得人工批准。
- 每个任务的工具白名单（Tool allowlist per task）。代理可以浏览；但除非明确为任务启用了该工具，否则它不能发起电汇等敏感操作。第 13 课讨论预算。
- 会话隔离（Session isolation）。浏览器代理会话使用受限凭据运行。禁止使用生产认证、个人邮箱。保留每次 HTTP 请求的日志以便审计。
- 内容清理器（Content sanitizer）。在将获取的 HTML 拼接进模型上下文前，剥离已知的恶意模式。（能减少简单攻击；无法阻止复杂载荷。）
- 对有后果操作的人类在环（HITL on consequential actions）。采用 propose-then-commit 模式（Lesson 15）。
- 记忆金丝雀（Canary tokens on memory）。若记忆项触发，用户会看到（Lesson 14）。

## 使用示例

`code/main.py` 模拟了一个小型浏览器代理对三页合成页面的运行。一个页面是良性页面，一个在可见文本中包含直接的提示注入代码块，另一个在 URL 片段中包含注入（不可见但在代理上下文中）。脚本展示了 (a) 一个天真的代理会做什么，(b) 读/写边界能拦下什么，(c) 清理器能拦下什么，(d) 哪些情况两者都无法拦下。

## 部署文件

`outputs/skill-browser-agent-trust-boundary.md` 限定了一个拟议的浏览器代理部署：涉及哪些信任区、被授权写入什么、以及在首次运行前必须部署哪些防御。

## 练习

1. 运行 `code/main.py`。识别清理器能拦下但读/写边界无法拦下的攻击，以及只有读/写边界能拦下的攻击。

2. 扩展清理器以检测一类 HashJack 风格的 URL 片段注入。在带有合法片段的良性 URL 上测量误报率。

3. 选择你熟悉的一个真实浏览器代理工作流（例如“预订机票”）。列出其中的每次读取和每次写入。标注哪些写入需要人类在环并说明原因。

4. 阅读 WebArena-Verified 的 ICLR 2026 论文。指出原始 WebArena 在评分上不可靠的一个任务类别，并解释 Verified 子集如何解决该问题。

5. 为浏览器代理设计一个记忆金丝雀。你会存储什么、放在哪里、以及是什么触发报警？

## 关键术语

| 术语 | 公众说法 | 实际含义 |
|---|---|---|
| 间接提示注入（Indirect prompt injection） | “页面里有恶意文本” | 代理读取的页面中包含不可信指令，代理将其执行 |
| Tainted Memories | “记忆攻击” | 代理将攻击者提供的指令写入持久记忆；在下次会话触发 |
| HashJack | “URL 片段攻击” | 有效载荷藏在 URL 的片段/查询字符串中，出现在代理上下文但不被可见渲染 |
| 一键劫持（One-click hijack） | “坏按钮” | 可见的交互控件携带后续载荷，代理执行该后续操作 |
| BrowseComp | “网页搜索基准” | 在开放网络上查找特定事实；分钟级时间尺度 |
| OSWorld | “桌面基准” | 完整的操作系统控制；多步 GUI 任务 |
| WebArena-Verified | “修正后的网页任务基准” | ServiceNow 重新评分的 WebArena，包含 Hard 子集 |
| 读/写边界（Read/write boundary） | “副作用门控” | 读取不产生后果；若发起内容来源于信任边界外，则写入需重新获得批准 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — 合并 Operator 与 Deep Research；BrowseComp SOTA。  
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 的谱系及后续成为 ChatGPT agent 的架构。  
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始基准。  
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 的修正子集论文。  
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含针对计算机使用代理的攻击面讨论。