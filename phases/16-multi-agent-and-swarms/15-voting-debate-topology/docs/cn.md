# 投票、自洽性与辩论拓扑

> 最便宜的聚合：采样 N 个独立代理，多数投票。Wang 等人 2022 年的自洽性就是用同一模型采样 N 次来做的。多代理扩展通过引入**异质**代理来摆脱单一文化——不同模型、不同提示词、不同温度、不同上下文。除了多数投票，辩论的拓扑也很重要：MultiAgentBench（arXiv:2503.01935，ACL 2025）评估了星型 / 链式 / 树型 / 图型的协调，并发现**图型在研究任务上最优**，但在 ~4 个代理之后有“协调税”。AgentVerse（ICLR 2024）记录了两种涌现行为——志愿者行为和从众行为——而从众既是一个特性（发现共识），也是一个风险（群体思维，Lesson 24）。本课绘制拓扑空间，构建每种变体，并测量协调税。

**Type:** Learn + Build  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 14 (Consensus and BFT)  
**Time:** ~75 分钟

## 问题

辩论可以提高准确性（Du 等人，arXiv:2305.14325），也可能降低准确性。辩论是否有益取决于四个结构性选择：

1. 谁与谁交流（拓扑）。
2. 多少轮（Du 2023：轮数和代理数独立影响结果）。
3. 代理是否异质（不同基础模型可以打破单一文化）。
4. 是否存在对抗性声音（强力辩护 vs. 簡化攻击）。

把“跑 5 个代理并投票”直接加到任务上的团队常常比单一代理退步。失败不是随机的：它们与拓扑和异质性有关。本课就是拓扑地图。

## 概念

### 自洽性，单模型基线

Wang 等人 2022 年的论文（"Self-Consistency Improves Chain of Thought Reasoning"）通过在温度 > 0 下对同一模型采样 N 次，并对推理路径的答案做多数投票。GSM8K 上的结果：在 N=40 时相较于单次贪心解码有显著提升。自洽性是多代理投票的单代理前身。

局限：自洽性使用一个基础模型。错误按构造是相关的。如果模型有系统偏差，所有 N 个样本都会共享该偏差。

### 多代理投票，异质扩展

用 N 个*不同*的代理替代 N 次采样。不同的基础模型（Claude、GPT、Llama）、不同提示词、不同工具访问。好处：误差不再高度相关。代价：不同代理成本不同；协调它们增加开销。

到 2026 年的规范名称为 **A-HMAD** — Adversarial Heterogeneous Multi-Agent Debate。尚未广泛被所有人采纳，但论文用该术语描述“不同模型之间的辩论，可以减少因单一文化崩溃带来的相关误差”。

### 四种拓扑

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (全连通)
```

星型（star）：一个枢纽，其他节点只与枢纽通信。等价于没有回通道的监督者-工人结构。  
链式（chain）：线性，每个代理看到前一个代理的输出。类似流水线。  
树型（tree）：分层，用于分层代理系统（Lesson 06）。  
图型（graph）：任意对任意。包括完全连通的团以及任意有向无环图（DAG）。

### 协调税（MultiAgentBench）

MultiAgentBench（MARBLE，ACL 2025，arXiv:2503.01935）在包括研究、编码和规划的任务集上对星型、链式、树型、图型进行了基准测试。关键测量结果：

- **图型** 在研究任务上胜出。信息任意流动；代理可以相互批评。
- **星型** 在快速回答的事实任务上胜出。枢纽进行过滤与整合。
- **链式** 在逐步流水线（分阶段细化）上胜出。
- **协调税** 在图型中在 ~4 个代理之后出现。墙钟时间和 token 成本增长速度超过质量提升。

这个 4 个代理上限是经验性的，不是根本性的。它反映了 2026 年 LLM 的上下文容量：每个代理的上下文被同伴输出填满，一旦每个人都能看到所有人，新增代理的边际价值会下降。

### 多代理辩论策略（“我们应该走 MAD 路线吗？”）

arXiv:2311.17371 是 2023 年的 MAD 策略综述。关键发现（被后来工作重复验证）：在相同预算下，结构上类似于自洽性的 MAD 变体（独立采样 + 聚合）通常不如自洽性。MAD 在代理真正异质且辩论具有对抗结构（某代理专门反对）时收益最大。

### AgentVerse 的涌现模式

AgentVerse（ICLR 2024，https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf）记录了即使在没有显式设计的情况下，多代理辩论也会出现两种行为：

- **志愿（Volunteer）**。代理主动提供帮助（“我可以执行下一步”）。有用处：它将子任务分配给最有能力的代理。  
- **从众（Conformity）**。代理将自己的立场调整为匹配批评者，即使批评者是错误的。这是辩论版的拍马屁（Lesson 14）。

从众是为什么“辩论直到一致”会奖励欺凌者的原因。受限轮次并且有单独裁判可以缓解这种现象。

### 异质性：真正提升准确率的旋钮

2024–2026 年实践文献的一个模式：将你的 N 个代理中的一个替换为不同的基础模型，带来的准确率提升通常比将 N 增加 1 更大。直觉为单一文化——每个新的独立误差源比一个额外的相关样本更有价值。

在极限情况下，异质性胜过数量。三个不同模型通常胜过五个同一模型副本，在大多数有明确标准答案的任务上如此。

### 陪审团方法（Jury methods）

Sibyl 框架（在 Minsky-LLM 文献中被引用）形式化了“陪审团”——一小组专门化代理在每个阶段通过投票来精炼答案。不同于纯多数投票，陪审团有角色分工：一个代理交叉询问，一个提供上下文，一个评分合理性。陪审团方法在纯投票（廉价但易单一文化）和完整 MAD（昂贵且易从众）之间取得折中。

### 何时投票加辩论占优

- 问题有明确的标准答案（事实、数学、代码行为）。投票收敛有意义。  
- 代理可以访问不同的数据源或工具（异质性可用）。  
- 轮次受限（典型 2–3 轮），并有单独的裁判或验证器。  
- 预算允许 3–5 个代理。在图型拓扑下，超过 5–7 个代理时协调税占主导。

### 何时投票加辩论适得其反

- 问题偏向意见型。代理会趋向于最有自信的答案，而非最正确的答案。  
- 所有代理共享相同的基础模型。单一文化使共识变得无意义。  
- 轮次不受限。从众总是获胜。  
- 任务很简单。单个代理使用 N=5 的自洽性更便宜且同样准确。

## 构建它

`code/main.py` 实现了：

- `run_star(agents, hub, question)` — 枢纽轮询每个工人并聚合。  
- `run_chain(agents, question)` — 顺序细化。  
- `run_tree(root, children, question)` — 深度 2 的分层聚合。  
- `run_graph(agents, question, rounds)` — 全互联辩论，有限轮次。  
- 一个脚本化的异质性旋钮：每个代理有一个 `error_bias` 表示其系统性错误倾向。  
- 一个测量工具，分别在 N=3, 5, 7 下运行每种拓扑并报告（accuracy、total_tokens、wallclock_simulated）。

运行：

```
python3 code/main.py
```

期望输出：一个拓扑 × N →（准确率、tokens、延迟）的表格。图型在研究类任务的 N=3-5 时胜出；星型在快速事实任务胜出；图型在 N=7 时显示协调税（延迟增长速度超过准确率）。

## 使用它

`outputs/skill-topology-picker.md` 是一个技能（skill），它读取任务描述并推荐一种拓扑（star / chain / tree / graph）、一个 N（代理数）、一个异质性配置（建议使用的基础模型）以及轮次上限。

## 交付部署建议

对于任何集成：

- 从**自洽性 N=5**开始，使用一个强力基础模型。这是廉价的基线。  
- 如果准确率关键，则升级到**异质投票 N=3**。测量差异。  
- 只有在任务具有结构性（研究、多步）且可实现轮次上限时，才升级到**辩论拓扑**。  
- 始终记录少数派簇。当少数派持续正确时，你有一个多样性信号。  
- 在衡量准确率的同时基准墙钟时间和 token 数。“更好但成本 10 倍”是商业决策。

## 练习

1. 运行 `code/main.py`。绘制图型拓扑的协调税曲线：准确率 vs N，tokens vs N。曲线在何处拐点？  
2. 实现 A-HMAD：三个具有刻意不同偏差的代理。在 Lesson 14 的单一文化攻击（monoculture attack）下，全同偏差基线与 A-HMAD 的比较如何？  
3. 在图型拓扑中添加一个不投票、仅对最终共识评分的“裁判”角色。这是否改变了涌现的从众行为？  
4. 阅读 AgentVerse 论文（ICLR 2024）。识别你的实现最强烈涌现出的行为是哪一种。你能否通过改变提示词来诱发相反的行为？  
5. 阅读 MultiAgentBench（arXiv:2503.01935）第 4 节（拓扑实验）。使用你的工具复现论文中“图型在研究任务上胜出”的结果（选择论文中的一个任务来复现）。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Self-consistency | "Sample N times, vote" | Wang 2022。单模型，N 个温度>0 的采样，对思维链的推理路径做多数投票。 |
| Heterogeneity | "Different models" | 不同基础模型或提示词家族的集合。打破单一文化。 |
| MAD | "Multi-agent debate" | 代理在若干轮中互相交换批评的通用术语。见 Du 2023。 |
| A-HMAD | "Adversarial Heterogeneous MAD" | 强调不同模型 + 对抗性结构的 MAD 变体。 |
| Topology | "Who talks to whom" | 星型、链式、树型、图型。决定信息流动方式。 |
| Coordination tax | "Diminishing returns" | 在图型中大约超过 4 个代理时，成本增长快于质量提升（协调税）。 |
| Volunteer behavior | "Unprompted help" | AgentVerse 的涌现模式：代理主动提出承担某一步。 |
| Conformity behavior | "Agreement under pressure" | AgentVerse 的涌现模式：代理在压力下与批评者观点一致。 |
| Jury | "Small specialized panel" | Sibyl 风格的集合，具有角色分工（询问者、上下文提供者、评分者）。 |

## 延伸阅读

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — 单模型基线  
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — 代理数 AND 轮数各自独立重要  
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — 拓扑基准，展示图型在研究类任务上优，链式适用于流水线  
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — MAD 策略综述；发现相同预算下 MAD 常输给自洽性  
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — 志愿者和从众的涌现模式  
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — 参考基准实现