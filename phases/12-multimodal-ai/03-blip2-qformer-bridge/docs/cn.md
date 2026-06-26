# From CLIP to BLIP-2 — Q-Former as Modality Bridge

> CLIP 将图像和文本对齐，但不能生成标题、回答问题或进行对话。BLIP-2（Salesforce，2023）通过一个小型可训练桥接模块解决了这个问题：32 个可学习的查询向量通过交叉注意力在一个冻结的 ViT 特征上进行注意，然后直接插入到一个冻结的 LLM 的输入流中。188M 参数的桥接模块将一个 11B 的 LLM 连接到 ViT-g/14。到 2026 年，所有基于适配器的 VLM —— MiniGPT-4、InstructBLIP、LLaVA 的亲属模型 —— 都是其后裔。本课阅读 Q-Former 的架构，解释其两阶段训练，并构建一个将视觉 token 提供给冻结文本解码器的玩具版本。

**Type:** 构建  
**Languages:** Python (stdlib，交叉注意力 + 可学习查询 演示)  
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 7 (Transformers)  
**Time:** ~180 分钟

## 学习目标

- 解释为什么在冻结的视觉编码器和冻结的大型语言模型（LLM）之间使用可训练的瓶颈，在成本和稳定性上优于端到端微调（Fine-tuning）。
- 实现一个交叉注意力模块，其中一组固定的可学习查询对外部图像特征进行注意。
- 讲解 BLIP-2 的两阶段预训练：表示学习阶段（ITC + ITM + ITG），然后是生成阶段（在冻结解码器上用 LM 损失进行训练）。
- 比较 Q-Former 与 LLaVA 中更简单的 MLP 投影器，并论证在何种情境下各自更优。

## 问题描述

你有一个冻结的 ViT，会为每张图像输出 256 个 patch token，每个维度 1408。你有一个冻结的 7B LLM，期望的 token 嵌入维度是 4096。显然的桥接方式 —— 一个从 1408 到 4096 的线性层 —— 是可行的，但将所有 256 个 patch token 直接输入到 LLM 的上下文会额外消耗 256 个 token。对于批量 32 张图像，这意味着视觉模态本身消耗了 8192 个 token。

BLIP-2 的问题是：能否将 256 个 token 的图像表示压缩到更少的 token（例如 32 个），同时保留足够的信息使得 LLM 能够生成标题、回答问题并进行推理？并且能否在不触碰冻结主干（backbones）的情况下仅训练这个桥接模块，从而让训练成本只在桥接参数上？

答案是：Q-Former。32 个可学习的“查询”向量通过交叉注意力从 ViT 的 patch token 中提取信息，产生 32 个 token 的视觉摘要供 LLM 使用。总计 188M 参数。在触碰 LLM 之前，先用对比、匹配和生成目标训练桥接模块。

## 概念

### 可学习查询（Learnable queries）

Q-Former 的核心技巧：不是让 LLM 的文本 token 去关注图像 patch，而是引入一组 32 个可学习的查询向量 `Q`，并让它们去关注图像 patch。查询向量是模型的参数 —— 它们在训练时被学习，并且对每张图像都使用同样的 32 个查询。

交叉注意力后，每个查询都包含图像的压缩摘要 —— “描述主要物体”、“描述背景”、“计数对象”等等。查询不一定会字面上专门对应语义标签；它们会学习任何能让下游损失下降的编码。

### 架构

Q-Former 是一个小型 Transformer（12 层，约 100M 参数），包含两条路径：

1. Query 路径：32 个查询向量先在自注意力（彼此之间）中流动，然后在冻结的 ViT 的 patch token 上进行交叉注意力，然后经过 FFN。
2. Text 路径：一个 BERT 式的文本编码器共享 Query 路径的自注意力和 FFN 权重。对于文本路径，交叉注意力被禁用。

在训练时，两条路径都会运行。查询和文本通过共享的自注意力交互，这意味着查询可以在需要时基于文本进行条件化（例如 ITM、ITG 任务）。在推理阶段用于 VLM 传递时，只运行查询路径，输出 32 个视觉 token。

### 两阶段训练

BLIP-2 的预训练分两阶段：

阶段 1：表示学习（不涉及 LLM）。三个损失：
- ITC（image-text contrastive）：CLIP 风格的对比损失，应用于池化后的查询 token 与文本 CLS token 之间。
- ITM（image-text matching）：二分类器 —— 这对图像-文本是否匹配？使用困难负样本采样。
- ITG（image-grounded text generation）：在文本上使用因果 LM 头，以查询为条件进行生成。迫使查询编码出可用于生成文本的内容。

只有 Q-Former 在训练中更新。ViT 保持冻结。不涉及 LLM。

阶段 2：生成学习。接入一个冻结的 LLM（如 OPT-2.7B 或 Flan-T5-XL 等）。将 32 个查询输出通过一个小线性层投射到 LLM 的嵌入维度。将它们预置到文本提示前。仅训练线性投影和 Q-Former，损失为拼接的提示 + 图像 + 标注序列上的 LM 损失。

阶段 2 完成后，Q-Former + 投影层即为完整的视觉适配器。推理流程：图像 → ViT → Q-Former → 线性投影 → 预置到文本 → 冻结的 LLM 输出生成。

### 参数经济学

BLIP-2 用 ViT-g/14（1.1B，冻结）+ OPT-6.7B（6.7B，冻结）+ Q-Former（188M，可训练）= 总计 8B 参数，其中 188M 是可训练的。Q-Former 约占整个堆栈参数的 ~2.4%。训练成本也反映了这一点：在几块 A100 上几天完成，而端到端微调则需要几周。

质量方面：BLIP-2 在零样本 VQA 上与 Flamingo-80B 持平或更好，同时模型体量小 50 倍。桥接设计是有效的。

### InstructBLIP 与指令感知 Q-Former

InstructBLIP（2023）在 Q-Former 中扩展了一个额外输入：指令文本本身。在交叉注意力阶段，查询现在可以同时访问图像 patch 和指令。查询可以针对具体指令进行专门化（“数车子”、“描述氛围”），而不是学习一个单一的固定摘要。在保持任务外推性能的同时对基准任务有提升。

### MiniGPT-4 与仅投影器方法

MiniGPT-4 保留了 Q-Former，但仅训练输出的线性投影而冻结其余部分。代价低，但质量受限 —— 查询是 BLIP-2 的，不是你训练得到的。适合快速迭代，但不是最佳架构。

### 为什么 LLaVA 更简单

LLaVA（2023，Lesson 12.05）用一个简单的 2 层 MLP 代替了 Q-Former，将每个 ViT patch token 投影到 LLM 空间 —— 对于 24x24 网格这是 576 token/图像，全部输入到 LLM。压缩率更差，但允许 LLM 直接关注原始 patch。在当时这是有争议的；到 2023 年末，由于大量的视觉指令数据（LLaVA-Instruct-150k），MLP 被证明可以训练出保留足够信号的模型。权衡是：LLaVA 的上下文更快被占满，但它更容易自然扩展到多图像和视频。

到 2026 年，领域分化：当 token 预算重要（长视频、多图像）时 Q-Former 存活；当每个 token 的质量更重要时 MLP 投影器占主导。

### 门控交叉注意力：Flamingo，祖先方案

Flamingo（Lesson 12.04）早于 BLIP-2，并在每个冻结的 LLM 层使用了相同的交叉注意力思想，而不是只做单层桥接。BLIP-2 展示了仅在输入层压缩也能奏效。Gemini 和 Idefics 则结合了两者：输入层的预置 token 加上可选的门控交叉注意力，用于上下文内的少样本（few-shot）。

### 2026 年的后裔

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及大多数为节省 token 预算而设计的视频-语言模型。
- Perceiver resampler：Flamingo 的变体（Lesson 12.04）；Idefics 家族、Eagle、OmniMAE。
- MLP projector：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- Attention pool：VILA、PaliGemma。

四种方案都有效。决定因素是你受限的是 token 预算还是每个 token 的质量。

## 使用方法

`code/main.py` 构建了一个 stdlib 风格的 Q-Former 式交叉注意力：

1. 模拟 256 个图像 patch token（维度 128）。
2. 实例化 32 个可学习查询（维度 128）。
3. 运行缩放点积交叉注意力（Q 来自查询，K/V 来自 patch）。
4. 通过线性层投射到 LLM 维度（512）。
5. 输出 32 个准备给 LLM 使用的视觉 token。

所有数学运算用纯 Python（对向量做嵌套循环）。是玩具实现，但形状和计算是正确的。注意力权重矩阵会被打印出来，这样你可以看到每个查询从哪些 patch 拉取信息。

## 交付物

本课生成 `outputs/skill-modality-bridge-picker.md`。在给定目标 VLM 配置（视觉编码器 token 数、LLM 上下文预算、部署约束、质量目标）时，它会推荐 Q-Former、MLP 或 Perceiver resampler，并提供简短的理由和每种桥接的参数量估计。

## 练习

1. 在 PyTorch 中实现交叉注意力模块。验证在 32 个查询和 256 个 key/value 的情况下，注意力权重矩阵是 32 x 256，并且每一行在 softmax 后和为 1。

2. 在 BLIP-2 的阶段 1 中，Q-Former 同时运行三个损失：ITC、ITM、ITG。为每个损失写出前向函数签名（伪代码）。哪一个损失需要文本编码路径处于激活状态？

3. 比较参数量：Q-Former（12 层，768 隐层）与一个 2 层 MLP 投影器（1408 → 4096，两层）。在什么规模的 LLM 下，188M 的 Q-Former 成本在训练效率上能收回成本？

4. 阅读 BLIP-2 论文的第 3.2 节（arXiv:2301.12597）关于 Q-Former 的初始化方式。解释为什么从 BERT-base 初始化（而非随机初始化）能加速收敛。

5. 对于 1 FPS 采样、10 分钟的视频（采样到 60 帧），计算每帧的 token 成本：(Q-Former → 32 token/帧) vs (MLP projector → 576 token/帧)。哪一种能适配到 128k-token 的 LLM 上下文窗口中？

## 关键词

| Term | People 常说 | 实际含义 |
|------|------------|----------|
| Q-Former | "Querying transformer" | 小型 Transformer，带有 32 个可学习查询向量，对冻结的 ViT 特征进行交叉注意力 |
| Learnable queries | "Soft prompt for vision" | 一组固定的参数，作为交叉注意力中的 query 端；为每个模型学习，并在所有输入间共享 |
| Cross-attention | "Q from here, K/V from there" | query、key 和 value 来自不同来源的注意力；查询如何从 ViT patch 中拉取信息 |
| ITC | "Image-text contrastive" | CLIP 风格的对比损失，应用在 Q-Former 池化的查询与文本 CLS 上 |
| ITM | "Image-text matching" | 对困难负样本进行的二分类；迫使查询区分细粒度的不匹配 |
| ITG | "Image-grounded text generation" | 因果 LM 损失，文本在查询条件下生成；迫使查询编码出可被解码为文本的内容 |
| Two-stage pretraining | "Representation then generative" | 阶段 1 训练 Q-Former（ITC/ITM/ITG）；阶段 2 接入冻结的 LLM，仅训练投影 + Q-Former |
| Frozen backbone | "Do not finetune" | 视觉编码器和 LLM 权重固定；只有桥接模块被训练 |
| Projection head | "Linear to LLM dim" | 将 Q-Former 输出映射到 LLM 嵌入维度的线性层 |
| Perceiver resampler | "Flamingo's version" | 类似的可学习查询交叉注意力，但 Flamingo 在每一层使用，而不是仅作为单层桥接 |

## 延伸阅读

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — 核心论文。  
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — 带有 ITC/ITM/ITG 三合一的前身。  
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — “align before fuse” —— 阶段 1 训练的概念祖先。  
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — 指令感知的 Q-Former。  
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — 仅投影器方法。  
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 用于可学习查询交叉注意力的通用架构。