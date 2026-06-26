# LLaVA 与视觉指令微调

> LLaVA（2023 年 4 月）是史上被复制最广泛的多模态架构。它用一个 2 层 MLP 替换了 BLIP-2 的 Q-Former，用简单的 token 级拼接替换了 Flamingo 的门控交叉注意力，并在由 GPT-4 根据仅文本的字幕生成的 158k 视觉指令轮次上进行了训练。任何在 2023 到 2026 年间构建 VLM 的实践者，都在某种程度上实现了 LLaVA 的变体。LLaVA-1.5 加入了 AnyRes。LLaVA-NeXT 提高了分辨率。LLaVA-OneVision 将图像、多图像和视频统一到一个配方中。本课阅读该配方，实现 projector（投影器），并解释为何“更简单者胜”。

**Type:** 构建  
**Languages:** Python（标准库，projector + 指令模板构建器）  
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — 指令微调)  
**Time:** ~180 分钟

## 学习目标

- 构建一个 2 层 MLP projector，将 ViT patch 嵌入（维度 1024）映射到 LLM 的嵌入维度（维度 4096）。
- 理解 LLaVA 的两阶段配方：（1）在 558k 字幕对上进行 projector 对齐；（2）在 158k GPT-4 生成的视觉指令轮次上进行视觉指令微调。
- 构造 LLaVA 格式的 prompt，包含图像 token 占位符、system prompt、用户/助手回合。
- 解释为何社区从 Q-Former 转向 MLP，尽管 Q-Former 在 token 预算上占优。

## 问题陈述

BLIP-2 的 Q-Former（第 12.03 课）将图像压缩为 32 个 token。干净、高效，对基准测试很好。但它有两个问题。

第一，Q-Former 是可训练的，但它的损失并不是最终任务。Stage 1 训练 ITC+ITM+ITG，Stage 2 训练 LM 损失。queries 学到的是一种中间表征，之后 LLM 还要解码它。瓶颈中会丢失信息。

第二，Q-Former 有 1.88 亿参数，而在 LLaVA 的 2023 年规模下，你不得不与目标 LLM 一起共同设计它。换 LLM 就要重训 Q-Former；换视觉编码器也要重训。每种组合都像是一个单独的研发项目。

LLaVA 的答案简单到令人尴尬：取 ViT 的 576 个 patch token，对每个 token 通过一个 2 层 MLP（`1024 → 4096 → 4096`），然后把所有 576 个投进 LLM 的输入序列。没有瓶颈。没有在奇怪目标上做的 stage 1 预训练。直接用 LM 损失训练 MLP。

数据从哪来？LLaVA 的第二个洞见是：用 GPT-4（仅文本）生成指令数据。把 COCO 字幕和图像的边界框文本喂给 GPT-4，要求其生成对话、描述和复杂推理问题。免费得到 158k 条指令-回应轮次。无需人工标注。

结果是：一个在 8 张 A100 上训练一天就能跑通的 VLM，在 MMMU 上击败了 Flamingo，并发布了可供社区扩展的开源 checkpoint。到 2023 年末，它已衍生出 50+ 个分支。

## 概念

### 架构

LLaVA-1.5（13B）：
- 视觉编码器：CLIP ViT-L/14 @ 336（Stage 1 冻结，Stage 2 可选解冻）。
- Projector：2 层 MLP，GELU 激活，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来为 Llama-3.1-8B）。

图像 + 文本 prompt 的前向流程：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占用了 LLM 上下文的 576 个 token。在 2048 的上下文中，这还剩 1472 个 token 给文本。在 32k 的上下文中，这几乎可以忽略不计。

### Stage 1：projector 对齐

冻结 ViT。冻结 LLM。仅训练这 2 层 MLP。数据集：558k 图像-字幕对（LAION-CC-SBU）。损失：在字幕上进行的语言建模，条件是投影后的图像 tokens。

在 batch=128 的单轮 epoch 中，几小时内就能完成。projector 学会将 ViT 空间映射到 LLM 空间。无任务特定监督。

### Stage 2：视觉指令微调

解冻 projector（仍可训练）。解冻 LLM（通常完全解冻，有时用 LoRA）。在 158k 视觉指令轮次上训练。

指令数据是关键。Liu 等人通过以下流程生成：
1. 取一张 COCO 图像。
2. 提取文本描述（5 条人工字幕 + 边界框列表）。
3. 用三种 prompt 模板发送给 GPT-4：
   - 对话： “生成一段关于该图像的用户与助手之间的来回对话。”
   - 详细描述： “给出对该图像的丰富、详细描述。”
   - 复杂推理： “提出一个需要基于图像推理的问题，然后回答它。”
4. 将 GPT-4 的输出解析为（instruction, response）对。

这些步骤并不直接接触图像——只用文本描述。GPT-4 会为图像“想象”出合理的内容。有噪声，但有效：158k 个轮次足以解锁对话能力。

### 为什么社区照搬

- 无需调参的 stage-1 特定损失。全程 LM 损失。
- Projector 几小时内可训练完，而不是数天。
- 可以更换 LLM（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），通常只需重训 projector。
- 视觉指令数据流水线基于 GPT-4，便于为新域低成本再生成。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）加入了：
- 将学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调。
- 更好的 system prompt。
- 支持从 2048 到 32k 上下文。

LLaVA-NeXT（2024 年 1 月）加入了：
- AnyRes：将高分辨率图像拆成 2x2 或 1x3 网格的 336x336 裁切 + 一个全局低分辨率缩略图。每个裁切变为 576 个 tokens；每张图像总计约 2880 个视觉 token。OCR 与图表任务表现跃升。
- 更好的指令数据混合，包含 ShareGPT4V（高质量的 GPT-4V 字幕）。
- 更强的基础 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第 12.08 课深入覆盖 OneVision。简短版本：相同的 projector，但以课程化训练覆盖单图像、多图像和视频，在共享的视觉 token 预算下统一模型。

### 与 Q-Former 的比较

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| Visual tokens per image | 32 | 576 (基础) 或 2880 (AnyRes) |
| Trainable params | 188M + LM | 40M + LM |
| Stage 1 loss | ITC+ITM+ITG | 仅 LM |
| LLM drop-in | 需要重训练 | 只需最小重训练即可替换 |
| Multi-image | 不方便 | 自然（拼接） |
| Video | 不方便 | 自然（逐帧拼接） |
| Token budget | 小 | 大 |

MLP 在简单性和 token 灵活性上胜出，Q-Former 在 token 预算上占优。到 2023 年末，token 预算已不再是绑定性约束（LLM 上下文增长到 32k–128k+），因此简单性占了上风。

### Prompt 格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是占位符 token。在分词之前，它会被 576 个视觉 tokens（或 AnyRes 的 2880 个）替换。Tokenizer 会看到一个比训练时稍长的序列，但 LLM 能处理这种新输入，因为 Stage 1 教会了它。

### 参数经济学

LLaVA-1.5-7B 参数分解：
- CLIP ViT-L/14 @ 336：303M（Stage 1 冻结，Stage 2 常解冻）。
- Projector（2x 线性层）：约 22M 可训练参数。
- Llama-7B：7B。
- 总计：7.3B 参数。Stage 2 可训练量：完整的 7B + 22M projector。

Stage 2 的训练成本：约 20 小时在 8x A100 上。这是关键数字——一天、一台节点、可复现。这也是 LLaVA 得以广泛传播的原因。

## 使用示例

`code/main.py` 实现了：

1. 一个纯 Python 的 2 层 MLP projector（玩具规模为维度 16 → 32 → 32）。
2. prompt 构建流水线：system prompt + 将 `<image>` 替换为 N 个投影 tokens + 用户回合 + 助手生成占位符。
3. 一个可视化工具，展示在 LLM 上下文中 576-token 视觉块占用的百分比（相对于 2k / 32k / 128k 上下文）。

## 部署（Ship It）

本课产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 系列 checkpoint，它会运行 10 条 prompt 的 vibes-eval 套件（3 个 caption、3 个 VQA、2 个推理、2 个拒绝），并输出可读的人类评分卡。它不是基准；只是一个连通性与烟雾测试，确认 projector 与 LLM 已正确衔接。

## 练习

1. 计算 2 层 MLP projector（`1024 → 4096 → 4096`，带 GELU 和偏置）的可训练参数量。它占 LLaVA-13B 的多少比例？

2. 为一个“拒绝”案例构造 LLaVA prompt —— 图像中含有私人个体。写出预期的助手回应。为什么 LLaVA 应该在零样本情况下拒绝？需要什么训练数据来强化这种拒绝行为？

3. 阅读 LLaVA-NeXT 博客的 AnyRes 部分。计算一张 1344x672 图像在 AnyRes 下的视觉 token 数。与基础 336x336 下的 576 tokens 做比较。

4. LLaVA 的 stage-1 projector 使用字幕上的 LM 损失进行训练。如果跳过 Stage 1，直接进入 Stage 2（视觉指令微调），会发生什么？引用 Prismatic VLMs 的消融（arXiv:2402.07865）给出答案。

5. LLaVA-Instruct-150k 使用 GPT-4 + COCO 字幕生成指令。针对新域（医学 X 光、卫星影像），描述用于生成领域指令的四步数据流水线。每一步可能出现什么问题？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Projector | "MLP bridge" | 2 层 MLP，GELU，将 ViT 维度映射到 LLM 维度 |
| Image token | "<image> placeholder" | 在推理前被 N 个投影视觉 tokens 替换的 prompt 标记 |
| Visual instruction tuning | "LLaVA stage 2" | 在 GPT-4 生成的（图像描述、指令、回应）三元组上训练 |
| Stage 1 alignment | "Projector pretraining" | 冻结 ViT 与 LLM，使用字幕上的 LM 损失训练 projector |
| AnyRes | "Multi-crop tiling" | 将高分辨率图像拆成瓦片网格，并将每个瓦片的视觉 tokens 串联 |
| LLaVA-Instruct | "GPT-4-generated" | 由 COCO 字幕 + GPT-4 合成的 158k 指令-回应对 |
| Vision encoder freeze | "Backbone locked" | Stage 1 中 CLIP 权重不更新，Stage 2 有时也不更新 |
| ShareGPT4V | "Better captions" | 由 GPT-4V 生成的 1M 条密集字幕，用于更高质量的对齐 |
| VQA | "Visual question answering" | 针对图像的自由格式问题进行回答的任务 |
| Prismatic VLMs | "Design-space paper" | Karamcheti 2024 的消融研究，系统性测试了 projector 与数据选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。  
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。  
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集字幕数据集。  
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间消融。  
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一单图像、多图像、视频。