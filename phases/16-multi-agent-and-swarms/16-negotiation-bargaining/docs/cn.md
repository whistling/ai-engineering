# Negotiation and Bargaining

> Agents negotiate resources, prices, task allocations, and terms. The 2026 benchmark set is clear: NegotiationArena (arXiv:2402.05863) shows LLMs can improve payoffs ~20% via persona manipulation ("desperation"); "Measuring Bargaining Abilities" (arXiv:2402.15813) shows buyer is harder than seller and scale does not help — their **OG-Narrator** (deterministic offer generator + LLM narrator) pushed deal rate from 26.67% to 88.88%; the Large-Scale Autonomous Negotiation Competition (arXiv:2503.06416) ran ~180k negotiations and found that **chain-of-thought-concealing** agents win by hiding reasoning from counterparts; Bhattacharya et al. 2025 on Harvard Negotiation Project metrics ranked Llama-3 most-effective, Claude-3 aggressive, GPT-4 fairest. This lesson implements Contract Net Protocol (the FIPA ancestor, Lesson 02), wires an LLM-style buyer/seller, runs an OG-Narrator-style decomposition, and measures how deal rate changes with each structural choice.

**Type:** 学习 + 构建  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 16 · 02（FIPA-ACL 传承）、Phase 16 · 09（并行群体网络）  
**Time:** ~75 分钟

## 问题

两个代理需要就一个价格达成一致。仅靠语言提示，2024–2026 年的 LLM 在实战议价中达成交易的比率出人意料地低（在 arXiv:2402.15813 的严格参数化议价场景中约为 27%）。规模并不能解决这个问题：GPT-4 在议价结构上并不优于 GPT-3.5；它只是更擅长议价的语言表达。

根本问题在于 LLM 混淆了两项任务——决定出价和叙述出价。OG-Narrator 把这两件事分开：一个确定性的报价生成器计算数值动作；LLM 只负责叙述。交易达成率跃升至约 89%。

这与经典多智能体发现相吻合：将机制与通信层解耦会带来优势。Contract Net Protocol（FIPA，1996；Smith，1980）是参考的任务-市场机制。把 LLM 插入到“叙述”插槽，你就得到了一个现代的 LLM 驱动任务市场。

## 概念

### Contract Net，一段话概述

Smith（1980 年）的 Contract Net Protocol：一个**管理者**广播一个**征求提案（cfp）**；**投标者**以包含报价的**propose** 消息回应；管理者选择胜出者并向胜者发送 **accept-proposal**，向其他投标者发送 **reject-proposal**。胜者完成工作。可选消息：**refuse**（投标者拒绝出价）。FIPA 将此归一为 `fipa-contract-net` 交互协议。

### 为什么 OG-Narrator 占优

"Measuring Bargaining Abilities of Language Models"（arXiv:2402.15813）观察到：

- LLM 常常违反议价规则（以荒谬价格出价，忽视对方的 ZOPA）。
- 它们锚定不好（接受糟糕的首轮出价；反报价往往是象征性的而非策略性的金额）。
- 单靠规模无法修正这些问题。更大的模型会产生更合情合理的语言，但策略性错误相似。

OG-Narrator 的分解如下：

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

报价生成器是经典的议价策略：Rubinstein 议价模型、Zeuthen 策略，或简单的以价格为单位的以牙还牙策略。LLM 负责叙述。消息包含确定性的价格和自然语言的表述。

交易率提升的原因包括：
- 价格保持在可议价区间内（ZOPA）。
- 锚点是策略性的，而非情绪化的。
- LLM 做它擅长的事：写作。

### NegotiationArena 的发现

arXiv:2402.05863 提供了规范基准。要点包括：

- 通过采用人物设定（“我迫切需要在周五前卖出”），LLM 可将收益提升约 20% —— 人设操纵是一种真实策略。
- 公平/合作的代理会被对抗性代理剥削；防御需要明确的反制姿态。
- 对称配对在约 40% 的基准场景下会收敛到不公平的结果。

这并不是说“LLM 是糟糕的谈判者”。而是“LLM 的谈判方式过于像人类，包含可被利用的部分”。

### 思维链（chain-of-thought）隐匿

Large-Scale Autonomous Negotiation Competition（arXiv:2503.06416）运行了约 18 万次谈判，涵盖多种 LLM 策略。获胜者通过对对手隐藏推理来赢得优势：

- 如果一个代理在公开可见的草稿区写下“我只会接受到 75 美元；我的保留价是 70 美元”，对手就会读取到这些信息。
- 获胜者在私下计算策略；输出通道只包含出价和最少量的叙述。

这呼应了经典博弈论（Aumann 1976 关于理性与信息）：泄露你的私有估值会损失收益。LLM 并不会自发领会这一点，且往往在可见的推理痕迹中输入保留价等信息，从而被对手利用。

工程启示：将私有草稿上下文和公开消息上下文分离。这不是可选项。

### Bhattacharya 等人 2025 的模型排名

在 Harvard Negotiation Project 指标（原则性谈判、尊重 BATNA、利益互惠）上：

- **Llama-3** 在达成交易率与收益上最为高效。
- **Claude-3** 是最具侵略性的谈判者（高锚点，迟缓让步）。
- **GPT-4** 是最公平的（在不同配对间收益方差最小）。

这是 2025 年的快照。要点不是哪款模型在 2026 年 4 月赢，而是不同基础模型具有持续的谈判风格。异构集成（Lesson 15）将其作为多样性的来源之一。

### 通过 Contract Net + LLM 实现任务分配

将 Contract Net 现代化应用于 LLM 多智能体场景的流程：

1. 管理者将任务分解为若干单元。
2. 向工作者广播 `cfp`，包含任务描述。
3. 每位工作者返回一个报价：`(price, eta, confidence)`，其中 price 可以是代币数、计算单元或美元等。
4. 管理者选择胜出者（单个或多个，视任务而定）并授予任务。
5. 被拒绝的工作者可以自由对其他任务投标。

因为协调采用广播-响应而非同步聊天，这一机制在超百个工作者时仍能很好扩展。已在生产中采用：Microsoft Agent Framework 的编排模式，以及若干 LangGraph 实现。

### LLM-利益相关者交互式谈判

NeurIPS 2024（https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf）提出了带有**秘密分数**和**最低接受阈值**的多方可评分博弈。每个利益相关方有私有效用；LLM 必须从消息中推断这些效用。这是二方议价向 N 方联盟形成的推广，对于具有异质工作能力的生产任务市非常相关。

### 叙述与机制的规则

在 2024–2026 年所有谈判基准中，一条一致的工程规则是：

> 让 LLM 负责叙述。不要让 LLM 计算出价。

如果出价需要是一个数值（价格、ETA、数量），应从谈判状态确定性地生成它，并让 LLM 产生 framing（表述）。如果出价需要是一个提案结构（任务分解、角色分配），可让 LLM 起草，但在发送之前必须按 schema 和约束进行校验。

## 构建实现

`code/main.py` 实现了：

- `ContractNetManager`、`ContractNetTask`、`Bid` —— 管理者 + 投标者，广播 cfp、收集提案、授予。
- `og_narrator_bargain(state, rng)` —— OG-Narrator 买方：确定性的 Zeuthen 风格向中点让步。
- `seller_response(state, rng)` —— 确定性的卖方反报价策略（两种风格的结构性真实基础）。
- `naive_llm_bargain(state, rng)` —— 模拟全由 LLM 执行的议价者：选择高方差的价格，常常落在 ZOPA 之外。
- 测度：在 1000 次试验中测算交易率，每次试验对保留价进行独立采样。

运行：

```
python3 code/main.py
```

预期输出：naive-LLM 的交易率约为 65–75%；OG-Narrator 的交易率约为 85–95%；15–25 个百分点的差距反映了将报价生成与叙述分离所带来的结构性优势。此外，还有一个包含 3 个投标者和 1 个任务的 Contract Net 任务市场分配示例。

## 使用方式

`outputs/skill-bargainer-designer.md` 设计了一个议价协议：谁生成出价（确定性或 LLM）、谁负责叙述、如何将私有草稿与公开消息分离，以及如何监控交易率。

## 上线清单

生产级议价检查表：

- **Separate scratchpad.** 私有状态绝不能到达对手的上下文。这是不可妥协的。
- **Deterministic offer generation.** 价格、数量、ETA：计算得出，而不是通过提示词让 LLM 生成。
- **Validate all incoming offers** 对所有传入报价按 schema 校验。在协议边界拒绝超出 ZOPA 的报价。
- **Bound rounds.** 回合限制为 3–5 回；陷入僵局时升级到调解者。
- **Measure deal rate and payoff variance** 持续监控交易率和收益方差。交易率下降通常是征兆——常见原因是提示词漂移或对手攻击。
- **Log all rejected proposals** 记录所有被拒绝的提案及确定性的理由。对于 Contract Net 管理者，落败的投标者需要理解原因。

## 练习

1. 运行 `code/main.py`。确认 OG-Narrator 在交易率上优于 naive-LLM。差距是多少？
2. 实现**基于人物的人设收益提升**（arXiv:2402.05863）——买方在叙述中采用“本周急需购买”的人设，但不改变报价生成器。交易率或收益是否发生变化？
3. 实现思维链（chain-of-thought）**隐匿**：维护一个不会传递给对手的私有草稿字符串。如果你不小心泄露它（通过交换通道模拟），会发生什么？
4. 将 Contract Net 扩展到带有保留价的 N 投标者拍卖。若所有出价都超过保留价，管理者如何在最低价与最高质量之间做抉择？你会选择哪种授予规则，为什么？
5. 阅读 Bhattacharya 等人 2025 年关于 Harvard Negotiation Project 的指标。实现两种不同风格（侵略型 vs 公平型）的议价者。在对称与非对称配对下衡量收益方差。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Contract Net | "Task market" | Smith 1980, FIPA 1996。cfp + propose + accept/reject。典型的任务市场。 |
| ZOPA | "Zone of possible agreement" | 买方最高价与卖方最低价的重叠区间。处于其外的出价无法成交。 |
| BATNA | "Best alternative to a negotiated agreement" | 如果这笔交易失败你的替代方案。设定你的保留价。 |
| OG-Narrator | "Offer generator + narrator" | 分解：确定性出价 + LLM 叙述。 |
| Zeuthen strategy | "Risk-minimizing concession" | 基于风险限制进行让步的经典出价生成器。 |
| Rubinstein bargaining | "Alternating-offer equilibrium" | 带折现的无限时域交替出价博弈的博弈论模型。 |
| CoT concealment | "Hide your reasoning" | arXiv:2503.06416 的获胜者保留私有草稿；公开通道仅显示出价。 |
| Persona manipulation | "Emotional posturing" | arXiv:2402.05863：通过表现出急迫/紧迫等人设可带来约 20% 的收益提升。 |

## 深入阅读

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — 基准；关于人设操纵与剥削的发现  
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator 与“买方比卖方难”结论  
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — 约 18 万次谈判；思维链隐匿胜出  
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — 带秘密效用的多方可评分博弈  
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — 经典机制，IEEE Transactions on Computers