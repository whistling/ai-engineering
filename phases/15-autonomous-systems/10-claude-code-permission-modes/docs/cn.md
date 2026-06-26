# Claude Code 作为自主代理：权限模式与自动模式

> Claude Code 暴露了七种权限模式。“plan” 在每个操作前请求批准，“default” 仅对有风险的操作请求批准，“acceptEdits” 自动批准文件写入但仍确认 shell 执行，而 “bypassPermissions” 则批准一切。自动模式（2026 年 3 月 24 日）用一个两阶段并行安全分类器替代了逐项批准流程：对每个动作运行一次单 token 的快速检查；被标记的动作会触发一轮思维链式的深入复审。通过 `max_turns` 和 `max_budget_usd` 强制执行操作预算。自动模式作为研究预览发布 —— Anthropic 明确指出该分类器本身不足以完全保障安全。

**Type:** Learn  
**Languages:** Python（stdlib，二阶段分类器模拟器）  
**Prerequisites:** Phase 15 · 01（长期代理），Phase 15 · 09（编码代理生态）  
**Time:** ~45 分钟

## 问题陈述

在你的机器上运行的自治编码代理属于一种特殊的安全类别。攻击面包括代理能接触到的一切——文件系统、网络、凭证、剪贴板、任何浏览器标签页、任何打开的终端。Bruce Schneier 等人已公开指出：计算机使用代理并不是聊天机器人的一次“功能更新”，它们是具有全新风险谱的新型工具。

Claude Code 的权限系统是 Anthropic 的应对方案。它不是一个简单的“自主 / 非自主”开关，而是由七个模式构成的能力阶梯：plan → default → acceptEdits → … → bypassPermissions。每种模式在速度与每次操作审核之间做不同的权衡。自动模式（2026 年 3 月）增加了一个两阶段分类器：对于分类器判断为安全的动作，将审批移出用户关键路径；对于被标记的动作则保留复审层。

工程问题是：该系统能捕获哪些风险，会遗漏哪些风险，以及针对特定任务应选择哪种模式？

## 概念

### 七种权限模式

| Mode | Behavior | When to use |
|---|---|---|
| `plan` | 代理提出一个计划；用户批准整个计划；每个动作在执行前都要审核 | 不熟悉的任务；与生产相关的代码；第一次在某个仓库上使用代理 |
| `default` | 代理执行动作；对任何“有风险”的操作（shell 执行、破坏性操作、网络调用）提示用户 | 大多数交互式编码会话 |
| `acceptEdits` | 自动批准文件写入；仍对 shell 执行和网络调用提示 | 大规模跨文件的重构 |
| `acceptExec` | 在经策划的白名单内自动批准 shell 命令；文件写入自动批准 | 紧凑的内部循环，其中每个 shell 命令都是 `npm test` 或类似操作 |
| `autoMode` | 两阶段安全分类器；被标记的动作会升级到复审 | 在受限工作区内的长期无人值守运行 |
| `yolo` | 跳过大多数提示；仍执行工具白名单/黑名单 | 短暂的沙箱、CI 作业、研究脚本 |
| `bypassPermissions` | 批准一切 | 文档中标注为“仅在你愿意丢弃的短期容器内使用” |

（上表中的名称与公开的 Claude Code 文档一致；“yolo” 是对较宽松中间地带的非正式简称。）

### 自动模式一页概览

自动模式（于 2026 年 3 月 24 日发布）是第一个将逐项批准职责委托给模型的权限模式。结构如下：

1. **Stage 1 — 单 token 分类器。** 对每个拟执行动作运行。成本低。返回“safe”或“flag”。被设计成与主代理循环并行运行，因此被批准的动作不会在队列中等待。
2. **Stage 2 — 次级策略/安全复审。** 仅对被标记的动作运行。对该动作、会话当前状态以及声明的任务进行聚焦的策略与安全评估。在置信度低时升级为人工 HITL（人类在回路中）审查。

预算控制与分类器并存：

- `max_turns` — 会话中的总迭代次数。
- `max_budget_usd` — 会使会话中止的美元上限。
- 每种工具的调用次数上限（例如不超过 N 次 `WebFetch` 调用等）。

### 系统能捕获的情况

- 直接的提示注入（prompt injection）到工具输入中，且被注入的指令映射到已知的高风险动作模式。
- 重复的工具循环——分类器可以检测到第 N+1 次动作几乎与第 N 次相同，连续发生多次。
- 在原本仅限文件编辑的会话中出现明显超出范围的 shell 命令。

### 系统可能漏掉的情况

- 细微的提示注入，会在不产生单个被标记动作的情况下调节行为。间接提示注入并非一个可完全修补的漏洞（参见 OpenAI 准备性负责人，2025，关于浏览器代理的讨论 —— 见第 11 课）。
- 语义层面的不当行为。每个单独的动作看起来都安全，但组合起来的轨迹可能有害。分类器对单个动作做判断；它不会重新推导用户的意图。
- 通过合法渠道的数据外泄。将数据写入你拥有的文件，然后 `git push` 到公共仓库，是一系列被允许的动作，但其组合存在问题。

### 研究预览定位

Anthropic 将自动模式作为研究预览发布。文档明确表示分类器只是一个层，非最终解决方案：用户应把自动模式与预算、白名单、隔离工作区和轨迹审计（第 12–16 课）结合使用。预览定位也反映出离线评估与实际部署间的差距（第 1 课）——在用户上下文模糊的真实会话中，通过离线评测通过的分类器可能会表现不同。

### 该阶梯在你工作流中的位置

- 不熟悉的任务：从 `plan` 开始。阅读计划比回滚一次糟糕的运行更便宜。
- 已知的重构：`acceptEdits` 可以节省大量确认点击。
- 无人值守的后台运行：只有在你已经衡量了爆炸半径的工作区内才使用 `autoMode`（无凭证、无生产挂载、无未经你选择的出口）。
- 临时容器：仅当容器及其凭证可丢弃时，才可接受 `yolo` / `bypassPermissions`。

```figure
autonomy-oversight
```

## 使用方法

`code/main.py` 模拟了两阶段分类器。Stage 1 是对拟执行动作的廉价关键字规则；Stage 2 是较慢的多规则复审器。驱动程序注入一段短的合成轨迹（安全操作、一次提示注入尝试、一个重复循环），并展示分类器在哪里捕获以及哪里遗漏。

## 上线部署

`outputs/skill-permission-mode-picker.md` 将任务描述匹配到合适的权限模式、预算上限和所需的隔离措施。

## 练习

1. 运行 `code/main.py`。哪一类合成动作从未被 Stage 1 标记但总是被 Stage 2 捕获？哪一类既未被捕获也未被标记？

2. 扩展 Stage 1 的规则集以捕获一种已知的恶意形态（例如 `curl $ATTACKER/exfil`）。在良性动作样本上测量误报率。

3. 阅读 Anthropic 的“How the agent loop works”文档。列出在 `default` 模式下代理默认会触及的每一个外部状态。在无人值守运行 `autoMode` 之前，你需要分别对哪些状态进行额外门控？

4. 设计一个 24 小时无人值守运行的预算：`max_turns`、`max_budget_usd`、每个工具的上限、白名单。为每个数字给出理由。

5. 描述一条轨迹，其中每个独立动作都被 Stage 1 和 Stage 2 批准，但组合行为仍然不对齐。（第 14 课讨论了如何通过断路器与金丝雀令牌来应对这类问题。）

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Permission mode | "How much the agent can do" | 七种命名策略之一，用于控制每次动作的批准 |
| plan mode | "Ask before anything" | 代理写出计划；用户在执行前批准 |
| acceptEdits | "Let it write files" | 自动批准文件写入；shell 执行仍需提示 |
| autoMode | "Auto approvals" | 两阶段安全分类器；被标记的动作会升级复审 |
| bypassPermissions | "Full YOLO" | 批准一切；旨在用于临时容器 |
| Stage 1 classifier | "Fast token check" | 对拟执行动作的单 token 规则；并行运行 |
| Stage 2 classifier | "Deep review" | 对被标记动作进行的思维链式深度复审 |
| Research preview | "Not GA" | Anthropic 对那些其失败模式仍在绘制中的功能所做的定位 |

## 延伸阅读

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 权限模式、预算、动作格式。  
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管服务执行模型。  
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — 功能面与自动模式发布说明。  
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 形成分类器判断的基于理由的层。  
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 关于长期权限设计的内部视角。