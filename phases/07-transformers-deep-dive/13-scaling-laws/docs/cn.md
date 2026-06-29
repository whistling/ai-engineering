# Scaling Laws

> 2020 年的 Kaplan 论文说：模型越大，损失越低。2022 年的 Hoffmann 论文说：你训练不足。计算资源分为两个桶 —— 参数和令牌（tokens）—— 两者的分配并不明显。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)  
**Time:** ~45 分钟

## 问题

当你有 C FLOPs 的训练计算预算并想得到最佳模型时，有两个旋钮可调：

1. **多少参数 (N)?** 模型越大，容量越高。  
2. **多少训练令牌 (D)?** 数据越多，参数的利用越好。

FLOPs 大致按 `6 × N × D` 规模增长。你可以把 N 提高并把 D 降低，或者把 D 提高并把 N 降低。哪种更好？

在 2022 年之前，答案通常是“尽量把 N 推大”。GPT-3（2020）有 175B 参数，训练约 300B 令牌。大约每个参数看到 1.7 个令牌。Kaplan 的缩放律支持这一点。

Hoffmann 等人（2022），在训练名为 Chinchilla 的一小系列模型时，发现了不同的结果：最优比率更接近 **每参数 20 个令牌**。GPT-3 被低估训练规模约 10 倍。Chinchilla（70B 参数，1.4T 令牌）在所有基准上都击败了 GPT-3（175B，300B 令牌），且推理成本仅为后者的 2.5×。

到了 2026 年，Chinchilla 的结论仍主导——但有一个重要的转折。Llama 3 8B 在 15 万亿令牌上训练，令牌/参数比为 1,875，远远超过 Chinchilla 的最优值 94 倍。对于将在大规模部署中使用的模型，推理成本比训练成本更重要，因此为得到更小的可部署模型而“过度训练”（超过 Chinchilla 最优）在 2026 年成为默认做法。

## 概念

![Chinchilla 曲线：在不同 N/D 比率下的损失 vs 计算](../assets/scaling-laws.svg)

### Hoffmann 定律

根据 Chinchilla 论文，损失遵循：

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 参数数（非嵌入部分）。  
- `D` = 训练令牌数。  
- `α ≈ 0.34`，`β ≈ 0.28`（大致对称）。  
- `E ≈ 1.69`，不可约损失上限。  
- `A ≈ 406`，`B ≈ 411`。

两个项在扩展时相互权衡。在固定计算（C = 6ND）下，对 `N` 求导并求解：

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

计算最优：每参数约 20 个令牌。

### 为什么仍然会“过度训练”

Chinchilla 最优在每个训练 FLOP 上最小化训练损失。但训练成本只付一次；推理成本则是永远的。

对于每月服务万亿令牌的聊天机器人，推理主导总成本。Llama 的策略：把模型做小，训练更久。8B 在 15T 令牌上训练显著优化推理性能：

- 能跑在消费级 GPU 上。  
- 延迟比 70B 的 Chinchilla 最优模型低很多。  
- 在大多数任务上质量已足够接近。

DeepMind 在 2024 年的一篇论文（“Over-training is the new optimal”）对其进行了形式化说明。对于推理主导的工作负载，合适的比率取决于服务体量，通常更接近每参数 100–500 个令牌。

### “涌现”与平滑性

有人声称某些能力（算术、多步推理、思维链跟随）会在某一规模突然“涌现”。

Schaeffer 等人（2023）认为这是一种测量伪像：所谓的涌现基于不连续评分（精确匹配、阈值准确率），掩盖了底层 logits 的平滑提升。用连续指标（交叉熵）观察则显示平滑曲线。

到 2026 年的共识是：通过连续损失得到的预测更可靠。基准中的跳跃往往是评分器的产物。基于连续指标来规划预算。

### 2026 年的全景

缩放律仍然适用，但：

| 因素 | 变化方式 |
|------|---------|
| 数据质量 | 筛选“好”令牌（Phi 风格）可以把曲线移 >2× 的有效计算量 |
| MoE | 总参数数与激活 FLOPs 解耦；按每激活 FLOP 的缩放律 |
| 后训练 | 某些能力（指令跟随、代码）对 SFT+RLHF 的依赖大于预训练 |
| 多模态 | 图像 + 文本令牌共同扩展；每种模态有各自曲线 |
| 合成数据 | 模型自生成训练数据；有效计算可以复合增长 |

Muon 优化器（Kimi Moonlight，2024）在同等数据下相较 AdamW 显示了 ~2× 的有效计算增益。到 2026 年，一些训练流程默认使用 Muon。这会改变缩放律中的绝对常数，但不改变其形状。

```figure
scaling-laws
```

## 实现

参见 `code/main.py`。我们实现 Chinchilla 的损失方程，并在若干计算预算下求解计算最优的 `(N, D)`。

### 第 1 步：Chinchilla 损失

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

将 `L` 在固定 `C = 6ND` 的约束下作为 `(N, D)` 的等高线绘出。找到最小值。

### 第 2 步：计算最优前沿

对于从 `1e17` 到 `1e25` FLOPs 的计算预算，寻找在 `6ND = C` 约束下使损失最小的 `(N, D)`。验证 `D/N ≈ 20`。

### 第 3 步：过度训练的成本

计算把模型做小 10×（N 为最优的 1/10，同时把 D 增为最优的 10×）你要付出的额外损失。报告以 N 成比例的推理 FLOP 节省作为交换。

### 第 4 步：与真实模型比较

代入已知的 `(N, D)` 对：GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（激活参数数），比较预测损失与报道损失。

## 如何使用

你很可能不会自己训练一台前沿模型。但缩放律能告诉你：

1. **你的微调数据是否足够。** 如果你的任务特定数据低于基模型每参数 20 个令牌，预计会在某个损失下饱和。  
2. **是否选择更大的基模型。** 如果你把预算都花在推理上，选择一个更小、训练更久的模型更划算。  
3. **收益何时递减。** 超过 Chinchilla 最优 1000× 后，对数损失的变化已接近噪声水平。

2026 年的研究轨迹：

- 数据受限的领域。经过过滤后，高质量英语令牌有限（约 5–10 万亿）。前沿预训练正接近此天花板。合成数据、多语种、多模态和基于 RLHF 放大的微调是下一批杠杆。  
- 计算乘数技巧。Muon 优化器、MoE、更好的数据整理——每一项都改变绝对常数，但不改变渐近规律。  
- RL 的缩放律。仍是悬而未决的问题。早期证据显示 RL 样本也呈幂律，但指数与预训练显著不同。

## 交付

参见 `outputs/skill-training-budget-estimator.md`。该工具根据计算预算、部署约束和目标损失为新训练任务选择 `(N, D, hours, GPU)`。

## 练习

1. 简单。运行 `code/main.py`。打印计算预算 `1e20`、`1e22`、`1e24` 的 Chinchilla 最优 `(N, D)`。与真实模型表比较。  
2. 中等。实现 Hoffmann 的“损失作为计算的函数”曲线。绘制计算最优前沿的损失 vs `log10(C)`。确定该定律预测我们需要何时 `>10^28` FLOPs 以获得额外 0.1 的交叉熵下降。  
3. 困难。在相同数据集上训练 5 个小模型（100K 到 10M 参数），拟合你自己的缩放律。估计 `α` 和 `E`。你的指数与已发表结果匹配程度如何？

## 关键词

| 术语 | 人们如何说 | 实际含义 |
|------|-----------|----------|
| Parameters (N) | "Model size" | 非嵌入的权重计数；决定模型容量。 |
| Tokens (D) | "Training data" | 看到的训练令牌数量；决定参数被利用的程度。 |
| Compute (C) | "FLOPs spent" | 对标准 transformer 近似为 `6 × N × D`。 |
| Chinchilla-optimal | "D/N ≈ 20" | 最小化每个预训练 FLOP 的损失的比率。 |
| Over-training | "Past Chinchilla" | 花额外训练 FLOP 来节省推理 FLOP；D/N >> 20。 |
| Irreducible loss | "The floor" | 缩放律中的 `E` 项；数据自身的熵。 |
| Emergent capability | "Sudden jumps at scale" | 常为评分器的产物；连续损失是平滑的。 |
| Effective compute | "Training-efficiency multiplier" | 更好的数据 / 优化器 / 架构让每个 FLOP 更“值钱”。 |

## 延伸阅读

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 第一篇缩放律论文；当时普遍训练不足。  
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla。  
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — 认为涌现是测量伪像。  
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — 解释为什么 Llama 的过度训练对其工作负载是正确的。  
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — Muon 优化器，宣称带来 ~2× 的计算乘数增益。