# 技能库与终身学习（Voyager）

> Voyager（Wang 等，TMLR 2024）将可执行代码视为一种技能。技能是有名称的、可检索的、可组合的，并通过环境反馈进行改进。这是 Claude Agent SDK skills、skillkit，以及 2026 年技能库模式的参考架构。

**Type:** 构建  
**Languages:** Python（stdlib）  
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)  
**Time:** ~75 分钟

## 学习目标

- 说出 Voyager 的三大组成部分 —— 自动课程、技能库、迭代提示机制 —— 以及每个部分的作用。
- 解释为什么 Voyager 将动作空间设为代码，而不是原语命令。
- 实现一个 stdlib 技能库，具备注册、检索、组合和基于失败的精炼能力。
- 将 Voyager 的模式映射到 2026 年的 Claude Agent SDK skills 和 skillkit 生态系统。

## 问题背景

每次会话都从头重建所有能力的智能体会犯三个错误：

1. **浪费 token。** 每个任务都重新触发相同的推理。
2. **丢失进展。** 在会话 A 学会的修正不会迁移到会话 B。
3. **在长时程组合上失败。** 复杂任务需要能力层次结构；一次性提示无法表达这些层次。

Voyager 的答案是：将每个可复用的能力作为一个以名称标识的代码块存储在库中，按相似性检索，可与其他技能组合，并通过执行反馈进行改进。

## 概念

### 三个组成部分

Voyager（arXiv:2305.16291）围绕以下内容组织智能体：

1. **自动课程（Automatic curriculum）。** 一个基于好奇心的提议器根据智能体当前的技能集合和环境状态挑选下一个任务。探索是从底层向上推进的。
2. **技能库（Skill library）。** 每个技能都是可执行代码。任务成功时会添加新技能。技能按查询与描述的相似度被检索。
3. **迭代提示机制（Iterative prompting mechanism）。** 失败时，智能体会收到执行错误、环境反馈和自我验证的输出，然后对技能进行改进。

在 Minecraft 的评估（Wang 等，2024）中：与基线相比，生成的独特物品数提升 3.3 倍、制作石制工具更快 8.5 倍、制作铁质工具更快 6.4 倍、地图遍历时间更长 2.3 倍。数据是 Minecraft 特定的，但该模式可以迁移到其他场景。

### 动作空间 = 代码

大多数智能体发出原语命令。Voyager 发出 JavaScript 函数。一个技能示例如下：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

由子技能组合而成。按描述和嵌入进行键控存储。作为程序被检索，而不是提示文本。

这就是 2026 年 Claude Agent SDK 的 skill：一个有名称、可检索的代码块，附带指令，智能体在需要时加载。

### 技能检索

新任务 “制作钻石镐”。智能体流程：

1. 将任务描述进行嵌入（embedding）。
2. 在技能库中查询 top-k 相似技能。
3. 检索到 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 从检索到的原语和新逻辑组合出新的技能。

这就是 MCP 资源（Phase 13）和 Agent SDK skills 所实现的模式：在知识/代码表面上进行检索，并限定到当前任务上下文。

### 迭代精炼

Voyager 的反馈循环：

1. 智能体编写一个技能。
2. 技能在环境中运行。
3. 返回三种信号之一：`success`、`error`（带堆栈跟踪）、`self-verification failure`。
4. 智能体使用该信号作为上下文重写技能。
5. 循环直到成功或达到最大轮次。

这是将 Self-Refine（Lesson 05）应用于代码生成并结合环境验证。CRITIC（Lesson 05）是类似模式，但验证器为外部工具。

### 课程与探索

Voyager 的课程模块会基于智能体已有的物品和尚未完成的任务提出诸如“在湖边建个避难所”之类的任务。提议器使用环境状态 + 技能库存来挑选恰好高出当前能力的任务 —— 即探索的甜 spot。

对于生产环境的智能体，这可转化为一个“缺失项”运算：给定当前技能库和一个领域，我们还缺乏哪些技能？团队通常以手动课程审查来实现这一点。

### 该模式的不足之处

- **技能库腐烂（Skill library rot）。** 同一技能以略有不同的描述被添加了 10 次。在写入时做去重；检索时只返回一个。
- **组合技能漂移（Composed-skill drift）。** 父技能依赖的子技能被精炼后会发生变化。为技能版本化；父技能被固定到 v1 时不会自动拾取到 v3。
- **检索质量。** 随着库超过几百条，基于描述的向量检索性能下降。用标签过滤和硬约束（“仅 category=tooling 的技能”）进行补充。

## 实现（Build It）

`code/main.py` 实现了一个 stdlib 技能库：

- `Skill` — name、description、code（字符串）、version、tags、dependencies。
- `SkillLibrary` — 注册、搜索（基于 token 重叠）、组合（依赖的拓扑排序）、以及精炼（更新时版本递增）。
- 一个脚本化的智能体会注册三个原语技能，组合出第四个技能，触发一次失败，并进行精炼。

运行它：

```
python3 code/main.py
```

运行跟踪将展示库写入、检索、组合、一次失败执行，以及 v2 的精炼 —— Voyager 的端到端循环。

## 使用场景（Use It）

- **Claude Agent SDK skills**（Anthropic）—— 2026 年的参考实现：每个 skill 有描述、代码和指令；在智能体会话期间按需加载。
- **skillkit**（npm: skillkit）—— 跨智能体的技能管理，支持 32+ 种 AI 编码智能体。
- **定制技能库**—— 面向特定领域（数据智能体的 SQL 技能、基础架构智能体的 Terraform 技能）。Voyager 模式可向下缩放。
- **OpenAI Agents SDK 的 `tools`**—— 在低层面上；每个 tool 是轻量级技能。

## 部署（Ship It）

`outputs/skill-skill-library.md` 生成一个符合 Voyager 模式的技能库，包含注册、检索、版本控制和精炼流程，适配任意目标运行时。

## 练习

1. 在 `compose()` 中添加依赖循环检测。若技能 A 依赖 B，而 B 又依赖 A，会发生什么？报错还是警告？
2. 实现每个技能的版本固定（per-skill version pinning）。当父技能以 `crafting@1` 组合子技能时，对 `crafting@2` 的精炼不得无感知地升级父技能。
3. 将基于 token 重叠的检索替换为 sentence-transformers 的嵌入（或实现一个 BM25 的 stdlib）。在 50 条技能的玩具库上测量 retrieval@5。
4. 添加一个“课程”智能体：给定当前库和领域描述，提出 5 项缺失技能。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK skill 文档。将该玩具库迁移到 SDK 的 skill schema。可发现性会发生哪些变化？

## 关键术语

| 术语 | 常说法 | 实际含义 |
|------|--------|----------|
| Skill | “可复用能力” | 有名称的代码块 + 描述，可按相似性检索 |
| Skill library | “智能体的 how-to 记忆” | 持久化存储的技能，可搜索并可组合 |
| Curriculum | “任务提议器” | 由当前能力差驱动的自下而上的目标生成器 |
| Composition | “技能 DAG” | 技能调用技能；按执行顺序做拓扑排序 |
| Iterative refinement | “自我修正循环” | 环境反馈 + 错误 + 自我验证折回到下一版本 |
| Action-space-as-code | “以代码作为动作空间” | 发出函数，而非原语命令，以实现时间扩展的行为 |
| Dedup on write | “写入去重” | 近似重复的描述在写入时合并为一个规范技能 |

## 深入阅读

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — 原始的技能库论文
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — 将技能产品化的 2026 年参考
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 实践中的技能与子智能体
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — Voyager 底层的精炼循环