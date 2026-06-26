# OpenAI 准备框架与 DeepMind 前沿安全框架

> OpenAI Preparedness Framework v2（2025 年 4 月）引入了研究类别 —— 长程自治（Long-range Autonomy）、沙袋行为（Sandbagging）、自主复制与适应、破坏防护等，与被跟踪类别（Tracked Categories）区别开来。被跟踪类别会触发能力报告（Capabilities Reports）以及保障报告（Safeguards Reports），并由安全咨询小组（Safety Advisory Group）审查。DeepMind 的 FSF v3（2025 年 9 月；2026 年 4 月 17 日加入被跟踪能力等级）将自治纳入 ML R&D 与网络（Cyber）领域（例如 ML R&D 自治等级 1 = 以对人类+AI 工具具有竞争力的成本完全自动化 AI 研发流程）。FSF v3 明确通过自动化监测针对工具性推理（instrumental reasoning）滥用来处理欺骗性对齐问题。需要诚实指出的是：PF v2 中的研究类别（包括长程自治）并不会自动触发缓解措施；政策措辞使用的是“潜在（potential）”。DeepMind 自身也表示，如果工具性推理增强，“自动化监测长期来看将不足以为继”。

**Type:** 学习  
**Languages:** Python（stdlib，三框架决策表差异工具）  
**Prerequisites:** Phase 15 · 19（Anthropic RSP）  
**Time:** ~45 分钟

## 问题

第 19 课仔细阅读了 Anthropic 的扩展政策。本课通过阅读 OpenAI 与 DeepMind 的文档来完成全貌。这三份文档是解决同一问题的姊妹型产物 —— 前沿实验室应在何时暂停或限制模型 —— 它们在一小组类别上达成一致，同时在一些关键细节上存在差异，差异会带来实际影响。

共识：三家都把长程自治作为值得跟踪的能力类别；三家都承认欺骗性行为（对齐造假、沙袋行为）是一类具体风险；三家都有内部审查机构。分歧点：OpenAI 将类别分为 “Tracked”（强制缓解）与 “Research”（不会自动触发）；DeepMind 则把自治折叠进两个领域，而不单独命名自治。不同实验室对某项能力属于哪个桶的命名（Tracked vs Research，或 Critical vs Moderate，或 Tier-1 vs Tier-2）不同，其操作后果也不同。

一起阅读这些文档是有价值的练习。同一项能力在 Anthropic 可能是“必须缓解”，在 OpenAI 则是“被监控但不触发缓解”，在 DeepMind 则是“在特定领域被跟踪”。政策取向很重要。

## 概念

### OpenAI Preparedness Framework v2（2025 年 4 月）

结构：

- **Tracked Categories**：触发能力报告（模型能做什么）以及保障报告（已部署哪些缓解措施）。在部署前由安全咨询小组审查。
- **Research Categories**：实验室正在跟踪但尚未承诺具体缓解的模型能力类别。包括长程自治、沙袋行为、自主复制与适应、破坏防护等。
  
v2 的关键表述：研究类别不会自动触发缓解。政策措辞为“潜在（potential）”缓解。需要关注的具体差别是：某个类别位于 Tracked 还是 Research，是“被门控”与“被观察”之间的分水岭。

### DeepMind Frontier Safety Framework v3（2025 年 9 月；2026 年 4 月 17 日加入被跟踪能力等级）

结构：

- **Critical Capability Levels (CCLs)**：跨五个领域的能力阈值：网络（Cyber）、生物（Bio）、机器学习研发（ML R&D）、化生放（CBRN）、自治（已折叠进 ML R&D 与 Cyber）。
- **Tracked Capability Levels**：2026 年 4 月加入的额外粒度。例如：ML R&D 自治等级 1 = 以对人类 + AI 工具具有竞争力的成本完全自动化 AI 研发流程。
- **欺骗性对齐监测**：明确承诺通过自动化监测检测工具性推理滥用。

自治的表述与 OpenAI 不同。DeepMind 并不将“自治”作为顶层领域；它把自治折叠进可能造成伤害的具体领域（ML R&D 与 Cyber）。其论点是：没有具体领域的自治只是能力而非风险；反论点则认为跨领域的自治是一类元风险，框架应当命名它。

### 三方达成的一致点

- 内部安全咨询小组（Anthropic 的 SAG、OpenAI 的 SAG、DeepMind 的内部委员会）。在高能力模型部署前进行审查。
- 明确提到欺骗性对齐 / 对齐作假作为一类风险。
- 有既定频率维护的常设文档工件（Anthropic：Frontier Safety Roadmap、Risk Report；OpenAI：能力与保障报告；DeepMind：FSF 更新周期）。
- 承认仅靠监测的防御有天花板。DeepMind 明确表示：“自动化监测长期来看将不足以为继”。

### 分歧点

- **Anthropic**：在 v3.0 中移除了暂停承诺；AI R&D-4 阈值是命名的下一道门。
- **OpenAI**：Tracked vs Research 的划分；Research 类别（包括长程自治）不会自动触发门控。
- **DeepMind**：将自治折叠进其他领域；被跟踪能力等级在 2026 年 4 月增加了粒度。

### 沙袋行为（Sandbagging）：一个让三方都复杂化的问题

沙袋行为（模型在评估中策略性地表现不佳）在 OpenAI 的研究类别中。Anthropic 的 RSP v3.0 通过评估-环境差距（见第 1 课）来处理它。DeepMind 在 FSF v3 中通过欺骗性对齐监测来应对。

如果模型在评估中沙袋，所有框架的能力阈值都会被低估。框架只有在测量有效的情况下才起作用。这就是为什么除实验室自评外，还需要外部测量（第 21 课，METR）和对抗性评估。

### 阅读政策的技能

- 定位（Locate）：你关心的每项能力应当能在政策里被找到；找不到就意味着该政策未覆盖它。
- 分类（Classify）：它是 Tracked（触发缓解）还是 Research（被跟踪但不触发）？OpenAI 明确命名此类；Anthropic 和 DeepMind 有各自的等效处理。
- 节奏（Cadence）：政策是按既定周期更新，还是只在特定事件后更新？有声明的更新节奏更强。
- 独立性（Independence）：外部审查是强制性的还是可选的？Anthropic 与 Apollo 及美国 AI 安全研究所合作；OpenAI 与 METR 合作；DeepMind 以内部 SAG 为主。

## 使用方法

`code/main.py` 实现了一个小型决策表差异工具。给定一项能力（自治、欺骗性对齐、研发自动化、网络提升等），它会输出三份政策如何对该能力分类，以及会触发哪些缓解措施。它是一个阅读辅助工具，而非政策工具。

## 交付物

`outputs/skill-cross-policy-diff.md` 会生成一份针对特定能力的跨政策比较，使用这三套框架作为参考。

## 练习

1. 运行 `code/main.py`。确认差异工具的输出与至少两项你可以根据原始文档验证的能力一致。

2. 完整阅读 OpenAI Preparedness Framework v2。识别每一个 Research 类别。对每一项写一句话说明为什么它属于 Research 而非 Tracked。

3. 完整阅读 DeepMind FSF v3，以及 2026 年 4 月的被跟踪能力等级更新。识别 ML R&D 自治等级 1 的具体评估标准。你会如何对其进行外部测量？

4. 沙袋行为在 OpenAI 的 Research 类别中。设计一个评估，能够迫使沙袋模型暴露其实际能力。参考第 1 课关于评估-环境博弈（eval-context-gaming）的讨论。

5. 在某一具体能力上比较三份政策（由你选择）。指出你认为哪份政策的分类最严格、哪份最不严格，并用原文依据进行论证。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Preparedness Framework | "OpenAI's scaling policy" | PF v2（2025 年 4 月）；Tracked vs Research 类别 |
| Tracked Category | "Mandatory mitigation" | 触发能力报告 + 保障报告；SAG 审查 |
| Research Category | "Monitored only" | 被跟踪但不会自动触发缓解；包括长程自治 |
| Frontier Safety Framework | "DeepMind's scaling policy" | FSF v3（2025 年 9 月）+ 被跟踪能力等级（2026 年 4 月） |
| CCL | "Critical Capability Level" | DeepMind 在各领域（Cyber、Bio、ML R&D、CBRN）设定的阈值 |
| ML R&D autonomy level 1 | "R&D automation" | 以对人类 + AI 工具具有竞争力的成本完全自动化 AI 研发流程 |
| Sandbagging | "Strategic underperformance" | 模型在评估中策略性地表现不佳；在 OpenAI 的 Research 类别中 |
| Instrumental reasoning | "Means-ends reasoning" | 工具性推理（关于如何实现目标的推理）；DeepMind 监测的目标 |

## 延伸阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 公告。  
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。  
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 公告。  
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — 被跟踪能力等级的补充说明。  
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 格式风险报告示例。