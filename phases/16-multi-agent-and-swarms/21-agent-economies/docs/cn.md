# Agent Economies, Token Incentives, Reputation

> 长期（METR 的 1 小时到 8 小时工作曲线）自治代理需要经济主体性。新兴的 **五层栈** 为：**DePIN**（物理算力）→ **Identity**（W3C DIDs + 声誉资本）→ **Cognition**（RAG + MCP）→ **Settlement**（账户抽象）→ **Governance**（Agentic DAO）。生产级的代理激励网络包括 **Bittensor**（TAO 子网根据任务奖励特定模型）、**Fetch.ai / ASI Alliance**（ASI-1 Mini LLM + FET 代币）、以及 **Gonka**（基于 transformer 的 PoW，将算力重新分配到有生产力的 AI 任务）。学术工作：AAMAS 2025 的去中心化 LaMAS 使用 **Shapley-value 信用归因** 来公平奖励贡献代理；Google Research 的 “Mechanism design for large language models” 提出在单调聚合下采用第二价格代币拍卖。该课时构建一个最小代理市场，对多代理流水线应用 Shapley 值归因，并运行第二价格代币拍卖，使博弈论机制具体落地。

**Type:** 学习  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 16 · 16（谈判与讨价还价）、Phase 16 · 09（并行群体网络）  
**Time:** ~75 分钟

## 问题

当多个代理联合产出价值但需要单独被奖励时，多代理系统会变得复杂。传统机制——平均分配、最后贡献者通吃——不公平或可被操纵。基于联盟的奖励通过 Shapley 值从构造上是公平的，但计算代价高昂。2025–2026 年的文献推动了实用近似方法：Shapley 采样、单调聚合拍卖，以及通过已确认贡献累积的链上声誉。

除了信用归因，领域也转向真正的经济代理：Bittensor 的 TAO 奖励挖矿算力用于对子网特定模型进行微调，Fetch.ai/ASI 用 FET 代币奖励 ASI-1 Mini LLM 的使用，Gonka 将 transformer 的工作量证明重新指向可生产的推理任务。当前已有能够自主交易的代理；问题在于如何对齐激励。

本课将代理经济视为一类具体问题——信用归因、机制设计与声誉——并以最小数学量构建每一部分，使思想易于理解。

## 概念

### 五层代理经济栈

1. **DePIN（物理算力）。** 将 GPU、存储、带宽出租的去中心化基础设施。Bittensor 子网、Render Network、Akash。不是代理专有；代理使用它。
2. **Identity。** W3C Decentralized Identifiers（DIDs）为每个代理提供独立于平台的持久 ID。声誉绑定到 DID。Agent Network Protocol（ANP）使用 DID 作为发现层。
3. **Cognition。** 代理的推理循环：LLM + RAG + MCP。其他阶段就是为此打基础。
4. **Settlement。** 账户抽象（ERC-4337）允许代理从自己的余额支付 gas 而不持有 ETH。代理可以为服务、互相或算力付费。
5. **Governance。** Agentic DAO：人类和代理共同对协议变更投票的治理结构，投票权与声誉挂钩。

并非每个生产系统都使用全部五层。Bittensor 使用第 1、2 层，部分使用第 3、4 层，不使用第 5 层。OpenAI 的代理仅使用第 3 层。该栈是参考地图，而非强制要求。

### Bittensor、Fetch.ai、Gonka —— 谁在运行什么

**Bittensor（TAO）。** 子网针对专门任务（语言建模、图像生成、预测）。矿工提交模型输出。验证者对其排序；按 stake 加权评分分配 TAO 奖励。每个子网有自己的评估方法。经济教训：按任务特定的输出质量付费，而非消耗的算力。

**Fetch.ai / ASI Alliance。** ASI-1 Mini LLM 在 Fetch.ai 网络上运行；用户为推理支付 FET 代币。这里的代理即等同点对点的同辈：一个 Fetch 上的代理可以调用另一个完成任务并用 FET 支付。

**Gonka。** Transformer 驱动的工作量证明：“工作”是 transformer 的前向推理。矿工通过运行具有已知正确输出（来自训练数据）的推理任务来获得报酬。是一种资源生产性的 PoW，替代基于哈希的 PoW。

截至 2026 年 4 月，以上三者均已达到生产级。回报分配方式各不相同：Bittensor 根据子网验证者的质量奖励；Fetch 根据付费用户衡量效用；Gonka 奖励可验证的推理工作。

### Shapley 值信用归因

三个代理协作完成一个任务，输出得分为 0.8。谁贡献了多少？

Shapley 值：满足四条公理（效率、对称性、线性、无关性）的唯一信用分配。对代理 `i`：

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

其中 `S_i_O` 是在排列 `O` 中出现在 `i` 之前的代理集合。实际做法：枚举所有排列，记录每个排列中每个代理的边际贡献，再取平均。

对于 N=3，有 6 种排列。对于 N=10，则为 360 万——因此通常采用抽样而非完全枚举。

### 聚合的第二价格拍卖

Google Research（“Mechanism design for large language models”）提出用于聚合 LLM 输出的第二价格代币拍卖。设置：N 个代理各自提出一个补全；每个代理对被选中有一个私人价值。拍卖方选择最高价值的提案并支付第二高的价值。在单调聚合（价值取决于被选择的提案，而不是出价数量）条件下，这是诚实机制——代理会出真实出价。

对 LLM 系统的重要性：可以把补全任务外包给多个代理、根据不同定价竞标；拍卖选出最佳方案并公平支付，代理没有动机虚报。

### 声誉资本

绑定到 DID 的声誉分数由已确认贡献累积。一个简单的更新规则：

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

其中衰减因子 `alpha` 接近 1。声誉特性：

- 对路由决策而言读取成本低（“把难任务发给高声誉代理”）。
- 难以伪造（随时间累计并绑定到 DID）。
- 可被削减：未通过验证的贡献会扣减。

### AAMAS 2025 的去中心化 LaMAS

LaMAS 提案（AAMAS 2025）结合了：DID 身份、基于 Shapley 值的信用归因、以及一个简单的拍卖机制。关键主张：把信用归因去中心化使系统可审计并免受单点操纵。

### 经济学失灵的场景

- **价格预言机操纵。** 若信用函数可被操纵，代理会去操纵它。每个机制都需要经过对抗性测试。
- **Sybil 攻击。** 同一操作者创建 N 个假代理以膨胀自己的贡献。DIDs 会减缓但不能完全阻止；通过增加伪造成本（声誉成本）来缓解。
- **验证成本。** 信用归因的公平性取决于验证器。如果验证便宜（小型 LLM），容易被操纵；若昂贵（人工评审），系统难以扩展。
- **监管不确定性。** 代理经济与金融监管相交。Bittensor、Fetch、Gonka 在某些司法辖区仍处于法律灰色地带（截至 2026 年）。

### 何时采用代理经济学

- **开放网络且运营者异质。** 没有单一团队控制所有代理。
- **输出可验证。** 没有验证，信用归因只是猜测。
- **长期工作流。** 一次性任务无法从声誉累积中获益。
- **代币支付在你所在司法辖区合法可行。**

在封闭的企业系统中，经济学常被更简单的分配方法取代（经理分配工作、内部度量）。经济学文献主要适用于开放网络。

## 实现

`code/main.py` 实现了：

- `shapley(value_fn, agents)` — 对小规模 N 通过枚举求精确 Shapley。
- `second_price_auction(bids)` — 诚实机制；赢家支付第二高出价。
- `Reputation` — 绑定 DID 的声誉，带指数衰减与削减。
- Demo 1：三个代理协作，精确 Shapley 分配信用。
- Demo 2：五个代理竞标任务位置；第二价格拍卖选择赢家并显示支付。
- Demo 3：对具有异质声誉的代理进行 100 轮任务分配；基于声誉的路由在预热后优于随机路由 10–20%。

运行：

```
python3 code/main.py
```

预期输出：每个代理的 Shapley 值；表明诚实出价均衡的拍卖结果；以及预热后基于声誉路由相对于随机路由的 10–20% 质量提升。

## 使用指南

`outputs/skill-economy-designer.md` 设计了一个最小代理经济：身份层选择、信用归因机制、支付机制、声誉规则。

## 部署建议

在 2026 年运行代理经济时：

- **先从声誉入手，而不是代币。** 声誉实现成本低且已足够有价值；代币会增加法律与经济复杂性。
- **先验证再奖励。** 未经独立验证不要分配信用。自报质量容易被 Sybil 操纵。
- **Shapley 用采样而非精确。** 对排列抽样 100–1000 次；精确枚举不可扩展。
- **限制衰减因子并设声誉下限。** 无界衰减会抹去合法贡献；衰减过慢会奖励过时的高声誉代理。
- **对机制进行对抗审计。** 在开放网络前做红队场景测试。每个机制都有博弈论漏洞；你要找出漏洞而不是攻击者。

## 练习

1. 运行 `code/main.py`。确认 Shapley 值满足总和等于总价值（效率公理）。改变 value 函数；Shapley 分配是否按预期方向变化？  
2. 实现 Shapley *采样*（对 K 个排列做蒙特卡洛）。K 如何影响近似精度？对 N=4 时与精确结果比较。  
3. 在拍卖前实现一个联盟形成步骤：代理可以合并为团队并作为单元竞标。哪些联盟会形成？与单独竞标相比，结果是否帕累托更优？  
4. 阅读 Google Research 的机制设计文章。找出一个假设，如果被违反会破坏诚实性。在 LLM 场景中该失败模式看起来如何？  
5. 阅读 AAMAS 2025 的去中心化 LaMAS 论文。在合成任务上对 10 个代理实现他们的 Shapley 步骤。精确计算需要多长时间？用 100 次抽样的近似能有多接近？

## 关键信息

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| DePIN | "Decentralized physical infrastructure" | 代币激励的算力/存储/带宽。Bittensor、Akash、Render。 |
| DID | "Decentralized identifier" | W3C 的可移植 ID 规范。代理声誉绑定到 DID，而非某个平台。 |
| ERC-4337 | "Account abstraction" | 可赞助 gas 的合约账户，支持代理支付。 |
| Shapley value | "Fair credit attribution" | 满足效率、对称性、线性、无关性的唯一分配。 |
| Second-price auction | "Vickrey auction" | 诚实机制：赢家支付第二高出价。与单调聚合兼容。 |
| Reputation capital | "Accumulated quality score" | 绑定 DID 的已确认贡献分数；随时间衰减。 |
| Agentic DAO | "Agents + humans govern" | 代理作为一等公民参与的 DAO，投票权与声誉相关。 |
| TAO / FET / GPU credits | "Token denominations" | Bittensor TAO、Fetch.ai FET、各种 DePIN 代币。 |

## 延伸阅读

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 2026 年关于五层代理经济栈的综述  
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — 在单调聚合下的代币拍卖  
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — 基于 Shapley 值的信用归因  
- [Bittensor TAO documentation](https://docs.bittensor.com/) — 子网结构与奖励分配  
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM 与 FET 代币  
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — 身份基础规范