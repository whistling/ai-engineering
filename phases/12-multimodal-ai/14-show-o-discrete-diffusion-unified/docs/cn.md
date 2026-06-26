# Show-o 和 离散扩散统一模型

> Transfusion 将连续与离散表示混合。Show-o（Xie et al., 2024 年 8 月）采取相反的路径：文本令牌使用因果的下一个令牌预测（causal next-token prediction），图像令牌使用受 MaskGIT 启发的掩蔽离散扩散（masked discrete diffusion）。两者都在同一个 transformer 内，使用混合注意力掩码。结果是在一个主干模型、每种模态一个分词器、一个损失形式（将下一个令牌预测推广到掩蔽预测）上统一了 VQA、文本到图像、图像修补和混合模态生成。本课介绍 Show-o 的设计——为什么掩蔽离散扩散是一个并行的、少步数的图像生成器——并与 Transfusion 和 Emu3 对比。

**Type:** 学习  
**Languages:** Python（stdlib，masked-discrete-diffusion 采样器）  
**Prerequisites:** Phase 12 · 13（Transfusion）  
**Time:** ~120 分钟

## 学习目标

- 解释掩蔽离散扩散：按均匀分布掩蔽令牌的调度，然后让 transformer 恢复它们的过程。
- 比较并行图像解码（Show-o、MaskGIT）与自回归图像解码（Chameleon、Emu3）在速度和质量上的差异。
- 列出 Show-o 在一个检查点中处理的三类任务：T2I、VQA、图像修补。
- 选择一种掩蔽调度（cosine、linear、truncated）并分析其对样本质量的影响。

## 问题背景

Transfusion 的双重损失训练可行，但具有更棘手的动态——连续扩散损失的数值尺度与离散 NTP 损失不同。平衡损失权重需要超参数搜索。架构有效但复杂。

Show-o 的答案是：保持两种模态均为离散（类似 Chameleon），但通过掩蔽离散扩散而不是顺序生成图像。训练目标变成单一的掩蔽令牌预测（masked-token-prediction），它自然地推广了下一个令牌预测（next-token-prediction）。

## 概念

### 掩蔽离散扩散（MaskGIT）

原始的 Chang 等人（2022）MaskGIT 思路很优雅。先从完全掩蔽的图像开始（每个令牌都是特殊的 `<MASK>` id）。每一步并行预测所有被掩蔽的令牌，然后保留预测置信度最高的 top-K，重新掩蔽其余令牌。经过大约 8–16 次迭代，所有令牌都被填充。每步解开多少令牌的调度需要调优——cosine 调度表现良好。

训练很简单：从 [0, 1] 区间均匀采样一个掩蔽比率，将其应用到图像的 VQ 令牌上，训练 transformer 恢复被掩蔽的那些令牌。这与 BERT 对文本做的事情完全相同，只是扩展到了图像生成。

### Show-o：一个 transformer，混合掩码

Show-o 将 MaskGIT 嵌入到一个因果语言模型 transformer 中。注意力掩码如下：

- 文本令牌：因果（标准 LLM）。
- 图像令牌：在图像块内是双向的（这样被掩蔽的令牌在预测时可以看到其他所有图像令牌）。
- 文本到图像：文本可以关注先前的图像，图像可以关注先前的文本。

训练交替进行：
1. 文本序列上的标准 NTP（next-token-prediction）。
2. T2I 样本：文本 → 图像，图像令牌被掩蔽，使用掩蔽令牌预测损失。
3. VQA 样本：图像 → 文本，文本令牌被掩蔽（本质上仍是 NTP）。

统一损失是对 `<MASK>` 令牌的交叉熵，这涵盖了文本的 NTP（仅最后一个令牌被“掩蔽”）和图像的掩蔽扩散（随机子集被掩蔽）。

### 并行采样

Show-o 在约 16 步内生成一张图像，而不是 ~1000 步（按令牌自回归）或 ~20 步（连续扩散）。每一步并行预测所有被掩蔽的令牌；提交置信度最高的 top-K；重复该过程。

比较：
- Chameleon / Emu3（对令牌自回归）：每个图像需要 N_tokens 次前向传递，通常为 1024–4096 次。
- Transfusion（连续扩散）：约 20 步，每步都是完整的 transformer 前向传递。
- Show-o（掩蔽离散扩散）：约 16 步，每步都是完整的 transformer 前向传递。

在相似规模模型下，Show-o 比 Chameleon 更快；与 Transfusion 的步数大致匹配，但每步成本较低（离散词表的 logits 与连续的 MSE 损失相比）。

### 一个检查点里的任务

Show-o 在推理时支持四类任务，按提示格式选择：

- 文本生成：标准自回归文本输出。
- VQA：图像输入，文本输出。
- T2I：文本输入，通过掩蔽离散扩散生成图像。
- 修补（Inpainting）：给定部分令牌被掩蔽的图像，进行补全。

修补能力是掩蔽预测训练天然带来的。掩蔽 VQ 令牌网格的某一区域，输入其余令牌和文本提示，预测被掩蔽的令牌即可。

### 掩蔽调度

每步解开的令牌数量的调度决定质量。Show-o 推荐使用 cosine：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T（步）
```

在第 0 步时，所有令牌被掩蔽（比率 1.0）。在第 T 步时，没有令牌被掩蔽。Cosine 将质量集中在中间区间，那里的预测最具信息量。线性调度也能工作，但会更快地到达平台期。

### Show-o2

Show-o2（2025 年后续工作，arXiv:2506.15564）对 Show-o 进行了扩展：更大的 LLM 基座、更好的分词器、改进的掩蔽调度。保持相同的架构模式。

### Show-o 在谱系中的位置

在 2026 年的分类中：

- 离散令牌 + NTP：Chameleon、Emu3。简单但推理缓慢。
- 离散令牌 + 掩蔽扩散：Show-o、MaskGIT、LlamaGen、Muse。并行采样，仍受分词器限制（有损）。
- 连续 + 扩散：Transfusion、MMDiT、DiT。质量最高，训练更复杂。
- 连续 + flow matching 在 VLM 中：JanusFlow、InternVL-U。最新方向。

按任务选择：当你需要在开放权重约束下同时做 T2I + 修补 + VQA 时选择 Show-o 系列；当质量优先且能接受双重损失架构时选择 Transfusion/MMDiT。

## 使用方法

`code/main.py` 模拟了 Show-o 的采样过程：

- 一个 16 个 VQ 令牌的玩具网格。
- 一个模拟“transformer”，基于提示和当前未掩蔽的令牌预测 logits。
- 使用 cosine 调度进行 8 步的并行掩蔽采样。
- 打印中间状态（掩码模式演化）和最终令牌。

运行它，观察掩码逐步消融的过程。

## 产出（交付）

本课产出 `outputs/skill-unified-gen-model-picker.md`。对于需要理解（VQA、图像字幕）和生成（T2I、修补）且要求开源权重的产品，给出在 Show-o 系列、Transfusion/MMDiT 系列与 Emu3 / Chameleon 系列之间的具体权衡建议。

## 练习

1. 掩蔽离散扩散大约需要 ~16 步。为什么不是 1 步？如果在第 0 步就把所有令牌全部解开，会出现什么问题？

2. 修补借助掩蔽扩散是“免费”的。提出一个产品化用例（真实或假设），说明 Show-o 的修补比专门模型更有优势的场景。

3. Cosine 调度与线性调度：当 T=8 时，追踪每步解开的令牌数量。哪一种更均衡？

4. 一个 512x512 的 Show-o 图像有 1024 个令牌。在词表 K=16384 时，模型发出的信息量为 1024 * log2(16384) = 14,336 bits（约 1.75 KiB）。Stable Diffusion 输出的原始像素为 512*512*24 bits = 6,291,456 bits（约 768 KiB）。压缩率是多少？换来的质量如何？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的类条件自回归图像模型与 Show-o 的掩蔽方法有何不同？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Masked discrete diffusion | "MaskGIT-style" | 训练去预测被掩蔽的令牌；在推理时，迭代地解开置信度最高的预测 |
| Cosine schedule | "Unmask schedule" | 推理步骤中掩蔽比率的衰减；将置信度增长集中在中间区间 |
| Parallel decoding | "All tokens at once" | 每一步在一次前向传递中预测完整序列中所有被掩蔽的令牌，然后提交 top-K |
| Hybrid attention | "Causal + bidirectional" | 对文本令牌使用因果掩码，对图像块内使用双向掩码的混合注意力 |
| Inpainting | "Fill-in generation" | 在部分令牌被掩蔽的图像条件下预测缺失令牌；该能力由训练目标自然支持 |
| Commitment rate | "Top-K per step" | 每次迭代声明“已完成”的令牌数量；控制推理速度与质量的折中 |

## 延伸阅读

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)  
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)  
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)  
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)  
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)