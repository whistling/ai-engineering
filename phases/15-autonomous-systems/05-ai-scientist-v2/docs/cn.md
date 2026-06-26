# AI Scientist v2 — 研讨会级自主研究

> Sakana 的 AI Scientist v2（Yamada 等人，arXiv:2504.08066）运行完整的研究闭环：假设、代码、实验、图表、撰写、投稿。它是首个由生成的论文通过 ICLR 2025 研讨会同行评审的系统（已披露来源）。独立评估（Beel 等人）发现 42% 的实验因编码错误而失败，文献综述常常将已建立的概念误标为新颖。Sakana 自身的文档也警告该代码库会执行由 LLM 生成的代码并建议使用 Docker 隔离。这两个方面共同构成了该系统的关键议题。

**Type:** 学习  
**Languages:** Python（标准库，research-loop 状态机玩具）  
**Prerequisites:** Phase 15 · 03（AlphaEvolve），Phase 15 · 04（DGM）  
**Time:** ~60 分钟

## 问题

研究是一个开放式任务。与 AlphaEvolve 的算法搜索或 DGM 的基准限定的自我修改不同，研究成果没有机器可校验的正确性标准。论文由审稿人评判，而不是单元测试。这使得闭环更难——但如果能闭合，其价值更高，因为研究是累积进步发生的地方。

AI Scientist v1（Sakana，2024）通过使用人类编写的模板来闭合循环。LLM 在固定脚手架内填充实验。AI Scientist v2（Yamada 等人，2025）通过使用具有视觉-语言模型（VLM）批评循环的 agentic 树搜索，消除了模板需求。该系统生成想法、实现实验、制作图表、撰写论文，并根据审稿反馈迭代。

同行评审事实：一篇 v2 生成的论文在 ICLR 2025 的一个研讨会上被接受（已披露）。独立评估结论：该系统远未可靠。两者都属实。

## 概念

### 架构

1. **想法生成。** LLM 在给定主题和既有文献的条件下提出研究想法。v1 使用模板；v2 使用在假设空间上的 agentic 搜索。
2. **新颖性检查。** 文献检索步骤检查该想法是否已被发表。这一步是 Beel 等人评估发现误标的环节——已建立的方法常被判为新颖。
3. **实验计划。** agent 起草实验方案并编写代码。
4. **执行。** 代码在沙箱中运行。失败会被送回重试循环。在 Beel 等人的测量中，42% 的实验在此阶段因编码错误失败。
5. **图表生成。** 视觉-语言模型读取生成的图表并为清晰性重写它们。这是 v2 的关键技术新增点。
6. **撰写。** LLM 起草论文，并与内部审稿人迭代。
7. **可选：投稿。** 将论文提交到会议或期刊。

### 研讨会接收结果意味什么

一篇 v2 生成的论文通过了 ICLR 2025 研讨会的同行评审。作者向程序委员会披露了论文来源。该接受是一个证明点；并不等同于声称系统“会做研究”。

重要背景：研讨会论文的门槛低于主会议论文。同行评审存在噪声；在任何给定时间被接受的提交占比通常很小。一例成功是概念验证，而非可靠性声明。Nature 2026 的论文记录了端到端闭环，该论文本身由人类研究者共同署名；它并不是“系统写出了一篇 Nature 论文”。

### 独立评估发现了什么

Beel 等人（arXiv:2502.14297）进行了外部评估。主要发现：

- **实验失败。** 42% 的实验因编码错误失败（坏的导入、张量形状不匹配、未定义变量）。重试循环捕获了一部分但并非全部错误。
- **新颖性误标。** 文献检索步骤常把已建立的概念标为新颖。这相当于研究领域的幻觉。
- **展示质量差距。** 视觉-语言图表批评能产生出版级别的可视化，从而掩盖底层实验弱点。

最后一点对于本阶段尤为重要。一个能生成令人信服的输出但并未做出令人信服研究的系统，比显而易见失败的系统更危险也更不可控。评估必须验证底层论断，而不能止步于图表。

### 沙箱逃逸的担忧

Sakana 自身代码库的 README 警告：

> 由于本软件会执行由 LLM 生成的代码，我们无法保证安全性。存在危险包、无控制的网络访问和产生非预期进程的风险。自行承担风险并考虑使用 Docker 隔离。

这是在未经验证领域内自治操作的典型形态。LLM 编写代码；代码运行；代码可以执行进程允许的任何操作。没有对文件系统、网络和进程行为进行硬限制的沙箱，任何自驱动研究 agent 都能外泄数据、消耗大量算力或自我改写。

AlphaEvolve 的沙箱故事较容易处理，因为其评估器很严格。AI Scientist v2 的循环运行开放式代码、实现开放式目标。这就是为什么它需要更强的隔离（至少 Docker；优选 seccomp / gVisor）以及在成果离开系统前对每篇投稿进行人工审查。

### v2 在前沿技术栈中的位置

| System | Target | Output kind | Evaluator | Known failure |
|---|---|---|---|---|
| AlphaEvolve | algorithms | code | unit + benchmark | 受评估器严格性约束 |
| DGM | agent scaffolding | code | SWE-bench | 奖励投机（reward hacking） |
| AI Scientist v2 | research papers | text + code + figures | 同行评审（弱） | 实验失败、误标、新打磨掩盖弱点 |

v2 在三者中拥有最弱的自动评估器、最广的输出面向，以及最快到达公开产物的路径。操作控制（沙箱、审查、披露）承担了大部分安全工作。

## 使用说明

`code/main.py` 将 v2 循环模拟为一个状态机：idea → novelty check → experiment → figure → writeup → review → accept-or-iterate。每个状态都有一个可配置的失败概率，取自 Beel 等人的发现。运行模拟 N 次并统计：

- 有多少想法到达了提交阶段。
- 有多少提交的论文包含被图表批评抛光过但存在严重实验缺陷的情况。
- 重试预算如何在质量和产出率间权衡。

## 部署说明

`outputs/skill-ai-scientist-sandbox-review.md` 是一份两门控的审查清单，用于在任何研究循环 agent 的产物离开沙箱前进行审核。

## 练习

1. 使用默认参数运行 `code/main.py`。有多少比例的循环运行产生了“干净”的论文？有多少比例产生了一个被图表批评抛光过但实验失败的论文？

2. 默认参数已使用 Beel 等人的 42% / 25%。重新运行：`--experiment-failure 0.20 --novelty-mislabel 0.10`，然后再运行 `--experiment-failure 0.60 --novelty-mislabel 0.40`。在两次运行之间，被抛光但有缺陷的份额如何变化？

3. 阅读 Sakana 的 AI Scientist v2 仓库 README 中关于沙箱要求的部分。为一个多天自动运行，列出两个你会额外施加的限制（除 Docker 之外）。

4. 阅读 Beel 等人第 4 节关于展示质量差距的讨论。设计一种额外的评估器，能够捕捉看起来抛光但实验上有缺陷的论文。

5. 提出一套可扩展于“不是每篇论文都由一名博士阅读”的研究 agent 输出人工审查方案。识别瓶颈并提出相应的设计。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|---|---|---|
| AI Scientist v1 | “Sakana 的模板化研究 agent” | 在固定脚手架中填充实验 |
| AI Scientist v2 | “无模板研究 agent” | 具有 VLM 图表批评的 agentic 树搜索 |
| Agentic tree search | “分支研究 agent” | 并行扩展多个实验方案；由内部评论器进行剪枝 |
| Vision-language critique | “VLM 对图表的润色” | 多模态模型读取图表并为清晰性重写它们 |
| Literature retrieval | “新颖性检查” | 检索既有工作以确认想法新颖性 —— 文档记录存在误标 |
| Polish masking | “漂亮的论文，坏的研究” | 展示质量优于实验质量；掩盖实验弱点 |
| Sandbox escape | “LLM 代码越狱” | Agent 执行的代码做了循环设计者未预期的事情 |

## 延伸阅读

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — 论文。  
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) — 厂商对同行评审背景的总结。  
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — 外部评估数据。  
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) — 模板化的前作。  
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 更广泛的开放式研究 agent 框架。