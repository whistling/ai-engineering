# Benchmarks: WebArena and OSWorld

> WebArena 测试基于浏览器的智能体在四个自托管应用上的能力。OSWorld 测试桌面智能体在 Ubuntu、Windows、macOS 上的能力。在发布时（2023–2024），两者都显示出顶尖智能体与人类之间存在巨大差距。差距在缩小；失败模式并未改变。

**Type:** 学习  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 14 · 19（SWE-bench，GAIA）  
**Time:** ~60 分钟

## Learning Objectives

- 描述 WebArena 的四个自托管应用以及为何基于执行的评估很重要。  
- 解释为何 OSWorld 使用真实操作系统截图而非无障碍 API。  
- 能说出 OSWorld 的两个主要失败模式：GUI grounding（GUI 定位）和 operational knowledge（操作知识）。  
- 总结 OSWorld-G 和 OSWorld-Human 在基础基准之上添加了哪些内容。

## The Problem

通用智能体可以调用工具。它们能否通过 20 次点击来驱动浏览器完成购物结账？仅通过键盘和鼠标能否配置一台 Linux 机器？WebArena 和 OSWorld 就是为回答这些问题而设计的。

## The Concept

### WebArena (Zhou et al., ICLR 2024)

- 包含 812 个长时程任务，覆盖四个自托管的 Web 应用：一个购物站点、一个论坛、一个类似 GitLab 的开发工具、一个企业 CMS（内容管理系统）。  
- 附带工具：地图、计算器、草稿板。  
- 评估通过 gym 风格的 API 基于执行来进行——订单是否下达、问题是否关闭、CMS 页面是否更新？  
- 发布时：最好的 GPT-4 智能体成功率为 14.41%，而人类为 78.24%。

自托管的设置很重要——基准不会因为目标应用变化而不稳定，目标应用是固定且可重现的。

### Extensions

- **VisualWebArena** — 视觉定向任务，成功依赖于对图像的解释（截图作为一等观测）。  
- **TheAgentCompany**（2024 年 12 月）— 添加终端和编码；更接近真实的远程工作环境。

### OSWorld (Xie et al., NeurIPS 2024)

- 覆盖 Ubuntu、Windows、macOS 的 369 个真实计算机任务。  
- 对真实应用进行自由形式的键盘和鼠标控制。  
- 将 1920×1080 的截图作为观测。  
- 发布时：最佳模型 12.24%，人类 72.36%。

### Primary failure modes

1. GUI grounding。像素到元素的映射问题。模型难以在 1920×1080 的截图中可靠定位 UI 元素。  
2. Operational knowledge。哪个菜单有设置、哪个键盘快捷键、哪个偏好面板——这是人类多年积累的知识长尾。

### Follow-ups

- **OSWorld-G** — 包含 564 个样本的 grounding（定位）套件 + Jedi 训练集。将定位问题与规划问题分解开来，以便单独度量。  
- **OSWorld-Human** — 手工策划的金标准操作轨迹。显示出顶级智能体所用步骤比必要步骤多 1.4–2.7 倍（轨迹效率差距）。

### Why this matters

Claude 的计算机使用、OpenAI 的 CUA、Gemini 2.5 Computer Use（Lesson 21）都在训练时使用了受 WebArena 和 OSWorld 形塑的工作负载。这些基准是目标；生产模型是给出的答案。

### Where benchmarking goes wrong

- 截图驱动的评估。OSWorld 以截图为驱动；在 OSWorld 上评估使用 DOM 或无障碍 API 的智能体会忽略定位（grounding）挑战。  
- 忽视轨迹长度。仅评分成功率会错过 OSWorld-Human 揭示的 1.4–2.7 倍步骤低效问题。  
- 自托管应用版本过时。WebArena 的应用固定了特定版本；如果在未重新策划的情况下更新，会破坏可比性。

## Build It

`code/main.py` 实现了一个简单的 web-agent 验证框架：

- 一个最小的“购物应用”状态机：list_items、add_to_cart、checkout。  
- 3 个任务的金标准轨迹。  
- 一个尝试完成每个任务的脚本化智能体。  
- 基于执行的评估器（状态检查）和轨迹效率度量（步骤数与金标准比较）。

运行方法：

```
python3 code/main.py
```

输出：每个任务的成功率和轨迹效率，模拟 OSWorld-Human 的方法论。

## Use It

- **WebArena Verified** 在内部集群上自托管用于持续评估。  
- **OSWorld** 在 VM 机群中用于桌面智能体评估。  
- **计算机使用智能体**（Lesson 21）——Claude、OpenAI CUA、Gemini ——都在类似工作负载上训练。  
- **你自己的产品流程**——为你前 20 个任务采集金标准轨迹；每周对智能体进行测试。

## Ship It

`outputs/skill-web-desktop-harness.md` 构建了一个带有基于执行评估和轨迹效率度量的 web/桌面智能体框架。

## Exercises

1. 在玩具框架中添加第二个应用（一个论坛）。为其编写 3 个任务并给出金标准轨迹。  
2. 为每个任务添加轨迹效率报告。在你的玩具上，智能体是 1 倍、2 倍还是 3 倍于金标准？  
3. 实现一个“干扰”工具——金标准轨迹从未使用该工具。脚本化智能体会被诱惑去使用它吗？  
4. 阅读 OSWorld-G。你将如何在自己的评估中把定位失败和规划失败分开衡量？  
5. 阅读 WebArena 的应用 README。升级某个被固定的应用版本会导致什么问题？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| WebArena | "Web agent benchmark" | 在 4 个自托管应用上的 812 个任务；gym 风格评估 |
| VisualWebArena | "Visual WebArena" | 基于视觉的 WebArena；截图作为观测 |
| OSWorld | "Desktop agent benchmark" | 在真实 Ubuntu/Windows/macOS 上的 369 个任务 |
| GUI grounding | "Pixel-to-element mapping" | 模型在 1920×1080 截图中定位 UI 元素 |
| Operational knowledge | "OS know-how" | 哪个菜单、哪个快捷键、哪个偏好面板 |
| OSWorld-G | "Grounding suite" | 564 个仅定位的样本 + 训练集 |
| OSWorld-Human | "Gold trajectories" | 手工专家动作序列，用于衡量效率 |
| Trajectory efficiency | "Steps over gold" | 智能体步数除以人类最少步数 |

## Further Reading

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — 四个应用的 Web 基准  
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — 跨操作系统的桌面基准  
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 的基准驱动能力介绍  
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 和 WebArena 的相关数据