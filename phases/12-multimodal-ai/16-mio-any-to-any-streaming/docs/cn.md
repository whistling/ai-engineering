# MIO 与 任意到任意的流式多模态模型

> GPT-4o 推出了大多数开源模型无法复制的产品特性：一个能听语音、看视频并实时语音回复的代理。到 2024 年底，开源社区的答案是 MIO（Wang 等，2024 年 9 月）。MIO 对文本、图像、语音和音乐进行分词，将它们交织成序列，训练一个自回归 Transformer，并实现任意模态到任意模态的生成。AnyGPT（Zhan 等，2024 年 2 月）是概念验证；MIO 是放大实现；Unified-IO 2（Allen AI，2023 年 12 月）是带视觉与动作接地的表亲。本课阅读任意到任意模式——四个分词器，一个 Transformer，支持流式的解码。

**Type:** 学习  
**Languages:** Python (stdlib, four-modality token allocator + streaming decode loop)  
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 6 (Speech and Audio)  
**Time:** ~120 分钟

## 学习目标

- 设计一个共享词表，使文本、图像、语音和音乐令牌在不冲突的情况下共存。  
- 比较 SEED-Tokenizer（图像）与 SpeechTokenizer residual-VQ（语音）在压缩率与重构质量上的权衡。  
- 解释构建任意到任意生成的四阶段训练课程。  
- 列出三种开源的任意到任意方案及其主要权衡：MIO、AnyGPT、Unified-IO 2。

## 问题背景

声明一个统一多模态模型很容易，但在大规模上构建很难。直到 2024 年，大多数“任意到任意”系统仍然是流水线式的：视觉模型 → 文本表示 → 语音模型 → 音频。每个跳转都会丢失信息、增加延迟并使训练变得复杂。GPT-4o 的演示视频展示了单模型的替代方案并能在亚秒级响应；开源系统在这方面落后好几个月。

工程挑战包括：

- 必须为每个模态设计分词器，压缩得尽可能无损以便重构，并以 Transformer 可消费的速率输出令牌。  
- 单一词表需要为文本（32k+）、图像（16k+）、语音（4k+）、音乐（8k+）分配空间。最少需要四万多个条目。  
- 训练数据必须覆盖每一种输入-输出对（text→image、image→speech、speech→image 等），或者模型必须学会组合能力。  
- 推理必须以足够快的速度流式输出令牌以满足会话延迟（<500ms 首字节时间）。

## 概念

### 针对四种模态的四个分词器

MIO 的分词器栈：

- 文本：标准 BPE，词表约 32000。  
- 图像：SEED-Tokenizer（2023）——带离散码本的量化 VAE，4096 条目，每张图像为 32x32 令牌。  
- 语音：SpeechTokenizer residual-VQ（2023）——将 16kHz 波形编码为 8 个分层码本；第一级为粗略内容，后续层补充韵律与说话人身份。  
- 音乐：类似的 residual-VQ（Meta 的 MusicGen / Encodec 家族），4-8 个码本。

每种模态输出整数令牌。令牌在共享词表中分配不重叠的 ID 范围：

```
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

总计：~48k 词表。输入嵌入与输出投影跨越全部条目。

### 流式解码

语音生成使用 residual-VQ。Transformer 预测语音的基础层（layer 0）令牌；并行解码的残差量化器预测后续层。每个第 0 层令牌大约对应 16kHz 下 ~50ms 的音频。

流式模式：

1. 用户在麦克风讲话；实时音频分词器每 50ms 输出语音令牌。  
2. MIO 在令牌到达时消费它们（提示预填充 + 增量前向）。  
3. 生成的输出令牌被流式发送；并行语音解码器将其转换为音频样本，延迟约 50–150ms。  
4. MIO 论文中的首字节音频时间（time-to-first-audio-byte）：~300–500ms，接近 GPT-4o 的 ~250ms。

Mini-Omni（arXiv:2408.16725）、GLM-4-Voice（arXiv:2412.02612）和 Moshi（arXiv:2410.00037）是互补的流式语音-LLM 设计。特别是 Moshi 在单 GPU 上实现了 160ms 的往返延迟。

### 四阶段训练课程

MIO 的训练课程：

1. Stage 1 — alignment。大规模模态对语料：text-image、text-speech、text-music。每对使用各自的词表段。训练共享词表。  
2. Stage 2 — interleaved。多模态交织文档（带图片与视频的博客、有转录的播客等）。训练跨模态上下文。  
3. Stage 3 — speech-enhanced。额外的音频数据以提升语音质量，同时不损害文本能力。  
4. Stage 4 — SFT。跨模态的指令微调：VQA、图像字幕、叙述、语音到语音对话。

缺失某一阶段会削弱特定能力：跳过第 2 阶段模型会丧失跨模态上下文；跳过第 3 阶段语音质量差。

### 视觉思维链（Chain-of-visual-thought）

MIO 引入了视觉思维链：模型在推理步骤中发出中间的图像令牌。对于“这只猫是在爬树吗？”之类的问题，模型会：

1. 发出 `<image>` 令牌呈现场景（基于输入图像或草图）。  
2. 发出分析该草图的文本。  
3. 给出最终答案。

渲染出的中间图像作为草稿板（scratchpad）。在空间推理任务上基准有所提升。该思路类似文本推理中的思维链（chain-of-thought）。

### 任意到任意的竞品

- AnyGPT（arXiv:2402.12226）：4 模态（text、image、speech、music），设计类似。  
- Unified-IO 2（arXiv:2312.17172）：增加视觉动作输出、深度、法线。任务多样但规模较小。  
- NExT-GPT（arXiv:2309.05519）：LLM + 模态特定的扩散解码器。不是单模型方案。  
- CoDi（arXiv:2305.11846）：可组合的扩散；通过共享潜在实现任意到任意。

MIO 是最接近纯令牌式任意到任意的方法。AnyGPT 是其概念上的先驱。

### 延迟预算

对于会话产品，每个组件的延迟都很关键：

- 麦克风到音频令牌：~50ms。  
- 预填充（音频令牌 + 历史）：在 8B 模型上 ~100ms。  
- 第一个输出令牌：~50ms。  
- 并行 residual-VQ + 语音解码器：~100–150ms。

首字节音频总时间：最低 ~300ms。GPT-4o 声称 ~250ms，Moshi 声称 160ms。MIO/AnyGPT 在公开基准上通常位于 400–600ms 范围。

### 为什么任意到任意依然困难

即便到 2026 年，开源的任意到任意模型在两个方面仍落后于闭源系统：

- 语音质量。residual-VQ 分词器是有损的；对话语音听起来比 ElevenLabs 级别的声音更机械。  
- 跨模态推理。让模型“根据所见来唱歌”仍比纯视觉任务更容易失败。

这些仍是开放的研究问题。Qwen3-Omni（Lesson 12.20）是 2025 年最先进的开源尝试之一。

## 使用方法

`code/main.py`：

- 定义四模态词表分配并打印。  
- 将一系列多模态输入（文本、图像、音频片段、音乐）路由到分词器路由器。  
- 模拟文本到语音响应的流式解码并统计延迟。  
- 根据编码器、预填充与解码器延迟计算预期的首字节音频时间。

## 部署建议（Ship It）

本课生成 `outputs/skill-any-to-any-pipeline-auditor.md`。给定会话产品规范（输入模态、输出模态、延迟目标），该报告审计 MIO 系列设计选择并计算延迟预算。

## 练习

1. 你的产品接受语音输入并返回语音输出。终端到终端的延迟预算目标是多少？列出各个耗时的组件。  

2. SpeechTokenizer residual-VQ 使用 8 个码本。说明为什么需要并行解码残差层（相对于顺序解码），以及它带来多少延迟节省。  

3. 你的词表包含 32k 文本 + 4k 图像 + 4k 语音。再加 8k 音乐和 ~10 个分隔符。在隐藏维度为 4096 时，嵌入矩阵的参数开销是多少？  

4. 视觉思维链会生成中间图像。哪些类型的问题会受益？哪些类型的问题会因额外令牌而受损？  

5. 阅读 Moshi（arXiv:2410.00037）。描述其“内在独白”（inner monologue）技巧，并与 MIO 的视觉思维链进行比较。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Any-to-any | "Multimodal in/out" | 一个模型可以以任意方向接收并输出文本、图像、语音和音乐 |
| Residual-VQ | "Speech tokenizer stack" | 多码本分词，每层添加信息；基础层为内容，后续层为韵律等 |
| SEED-Tokenizer | "Image codes" | MIO 使用的具有 4096 条目的离散图像分词器 |
| Chain-of-visual-thought | "Visual scratchpad" | 模型在最终答案前生成一个中间图像作为推理步骤（视觉思维链） |
| Time-to-first-audio-byte | "TTFAB" | 从用户语音到首个音频输出的延迟；对话感受需要 <500ms |
| Four-stage curriculum | "Training recipe" | Alignment -> Interleaved -> Speech-enhanced -> SFT（按此顺序） |

## 进一步阅读

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)  
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)  
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)  
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)  
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)