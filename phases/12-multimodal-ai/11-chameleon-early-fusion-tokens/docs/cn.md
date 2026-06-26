# Chameleon 与 早期融合的仅令牌多模态模型

> 到目前为止我们见过的每个 VLM 都把图像和文本分开。视觉令牌来自视觉编码器，流入投影器，然后在 LLM 内与文本相遇。视觉和文本的词表从不重叠。Chameleon（Meta，2024 年 5 月）提出：如果它们重叠会怎样？训练一个 VQ-VAE，把图像转换为来自共享词表的离散令牌序列。每个多模态文档现在都是一个序列——文本令牌和图像令牌交错，使用单一自回归损失。副作用：模型可以生成混合模态输出——在一次推理调用中交替输出文本和图像令牌。本课阅读早期融合论点并端到端构建一个玩具版本。

**Type:** 构建  
**Languages:** Python（标准库，VQ-VAE 分词器 + 交错解码器）  
**Prerequisites:** Phase 12 · 05, Phase 8（生成式 AI）  
**Time:** ~180 分钟

## 学习目标

- 解释为何共享词表 + 单一损失会改变模型的能力。  
- 描述 VQ-VAE 如何将图像标记化为与 transformer 的下一个令牌目标兼容的离散序列。  
- 列出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 放置、LayerNorm 顺序。  
- 比较 Chameleon 与 BLIP-2 的 Q-Former 方法，并说明在何种情况下应选择哪种方法。

## 问题背景

基于适配器的 VLM（如 LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两类不同的数据。一个文本令牌经过 `embed(text_token)`；一张图像经过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，中途才融合。

三个后果：

1. LLM 只能消费图像，不能生成图像。输出仅为文本。  
2. 混合模态文档（交替段落和图像，例如文章）很尴尬——你要么在模型外部解析多模态输入，要么串联生成。  
3. 分布不匹配。视觉令牌和文本令牌位于隐藏空间的不同区域，产生微妙的对齐问题。

Chameleon 拒绝这个前提：图像只是来自共享词表的离散令牌序列。用交错文档训练模型，一个损失、一个自回归解码器，你就能免费解锁混合模态生成。

## 概念

### 作为图像分词器的 VQ-VAE

分词器是一个向量量化变分自编码器。架构：

- 编码器：CNN + ViT，将图像映射为空间特征图，例如 32x32 的 256 维特征。  
- 码本：学习得到的 K 个向量（Chameleon 使用 8192），维度也是 256。  
- 量化：对每个空间特征，按 L2 距离查找最近的码本条目。用整数索引替代连续特征。  
- 解码器：CNN，将量化后的特征还原为像素。

训练：VAE 重建损失 + 承诺损失 + 码本损失。码本索引构成图像的离散字母表。

对 Chameleon 来说：一张图像变为 32*32 = 1024 个令牌，从 8192 个词表中选取。与文本令牌（来自 LLM 的 BPE 词表，例如 32000）拼接。最终词表大小：40192。Transformer 看到的是一个序列，一个损失。

### 共享词表

Chameleon 的词表结合了文本令牌、图像令牌和模态分隔符。每个令牌都有唯一的 ID。输入嵌入层将每个 ID 映射到 D 维隐藏向量。输出投影把隐藏向量映射回词表 logits。Softmax 选择下一个令牌，不论其模态为何。

分隔符很重要：`<image>` 与 `</image>` 标签将图像令牌序列括起来。在生成时，如果模型输出 `<image>`，下游软件就知道接下来的 1024 个令牌是 VQ 索引，应发送到解码器进行像素渲染。

### 混合模态生成

推理是对共享词表的下一个令牌预测。示例提示：“Draw a cat and describe it.” Chameleon 会输出：

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自主选择顺序——它可能先生成图像然后文本、先文本后图像，或交错。相同的解码器、相同的损失。

与只生成文本的适配器 VLM 相比，Chameleon 重新打开了模型输出模态的可能性。

### 训练稳定性 —— QK-Norm、dropout、LayerNorm 顺序

早期融合在大规模下训练不稳定。Chameleon 论文记录了三项技巧：

- QK-Norm。在 attention 内部将 LayerNorm 应用于 query 和 key 投影，在点积之前进行。防止深层处 logits 幅度爆炸。被多款 2024 年后出现的大模型采用。  
- Dropout 放置。在每次残差相加（residual-add）之后应用 dropout，而不仅仅在 attention 和 MLP 之后。当来自图像令牌的梯度可能占主导时，需要更多正则化。  
- LayerNorm 顺序。残差分支使用 Pre-LN（标准做法），并且在最后一层的跳跃连接上增加一个额外的 LN。稳定化最后一层的梯度流。

没有这些技巧，34B 参数的 Chameleon 在多个检查点上训练会发散。有了它们，训练收敛。训练配方与架构一样是贡献之一。

### 分词器的重建上限

VQ-VAE 是有损的。在 8192 个码本条目和每张 512x512 图像 1024 个令牌的配置下，重建 PSNR 大约上限为 26–28 dB。这足以生成可识别的图像，但明显不如连续空间扩散方法（例如 Stable Diffusion 3 可达 32+ dB）。

分词器是瓶颈。更好的分词器（如 MAGVIT-v2、IBQ、SBER-MoVQGAN）可以提升上限。Emu3（第 12.12 课）仅通过更好的分词器就实现了与 SDXL 相当的生成质量。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（早期融合、共享词表）：
- 一个损失，一个解码器。  
- 生成混合模态输出。  
- 分词器是质量上限。  
- 昂贵：在推理路径上每生成一张图像需运行 VQ-VAE 解码器。

BLIP-2 / LLaVA（后期融合、分离塔）：
- 视觉输入进来，文本输出走出（只能输出文本）。  
- 重用预训练的 LLM。  
- 理解任务没有分词器瓶颈。  
- 便宜：只需一次前向传播。

按任务选择。如果你需要图像生成，选择 Chameleon 系列；如果只需理解，适配器 VLM 更简单且能更多复用预训练计算。

### Fuyu 与 AnyGPT

Fuyu（Adept，2023）是相关方法：完全跳过单独的视觉编码器，把原始图像 patch 作为令牌通过 LLM 的输入投影喂入，就像令牌一样，不用分词器。比 Chameleon 简洁，但无法实现共享词表的输出生成。  

AnyGPT（Zhan 等，2024）将 Chameleon 扩展到四种模态：文本、图像、语音、音乐。对每种模态都使用类似的 VQ-VAE 技术，共享 transformer。实现任意到任意的生成。在第 12.16 课有更多内容。

## 使用方法

`code/main.py` 构建了一个端到端的早期融合玩具模型：

- 一个小型 VQ-VAE 风格的量化器，将 8x8 patch 映射为码本索引（K=16）。  
- 一个共享词表：文本 id 0..31，图像 id 32..47，分隔符 48、49。  
- 一个玩具自回归解码器（bigram 表），在合成的标题 + 图像令牌序列上训练。  
- 采样循环，根据提示输出交替的文本 + 图像令牌。

代码有意将 transformer 做得很小（双字搭档表）以便你能端到端追踪信号流。

## 交付产物

本课会产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品规范（仅需理解 vs 需要理解+生成、所需图像质量、成本预算），它会在 Chameleon 系列（早期融合）和 LLaVA 系列（后期融合）之间进行选择，并用定量经验法则给出理由。

## 练习

1. Chameleon 使用 K=8192 的码本条目和每张 512x512 图像 1024 个令牌。估算与一张 24 位 RGB 图像相比的压缩比。它是有损的吗？有多大损失？  

2. 一张 4K 图像（3840x2160）以相同的 VQ-VAE 密度会产生多少图像令牌？Chameleon 风格的模型能否在一次推理调用中生成 4K 图像？首先会先出现什么问题——上下文、分词器质量，还是 KV cache？  

3. 用纯 Python 实现 QK-Norm。给定 64 维的 query 和 key，展示在 LayerNorm 前后它们的点积。为什么在深层中需要控制幅度？  

4. 阅读 Chameleon 论文第 2.3 节关于训练稳定性的内容。描述论文在没有 QK-Norm 的 34B 参数情形下观测到的确切失败模式。所谓的“范数爆炸”特征是什么？  

5. 扩展玩具解码器，使之在文本提示下发出混合模态响应。测量在训练数据分布为 60% 先文本 / 40% 先图像 时，模型选择先生成图像 vs 先生成文本的频率。

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Early fusion | "Unified tokens" | 图像从一开始就被转换为与 transformer 共享词汇表的离散令牌 |
| VQ-VAE | "Image tokenizer" | CNN + ViT + 码本，将图像映射为 transformer 可预测的整数索引 |
| Shared vocabulary | "One dictionary" | 覆盖文本 + 图像 + 模态分隔符的单一令牌 ID 空间 |
| QK-Norm | "Attention stabilizer" | 在点积之前对 query 与 key 应用 LayerNorm，防止范数爆炸 |
| Mixed-modality generation | "Text + image output" | 在一次推理中自主生成交错的文本与图像令牌 |
| Codebook size | "K entries" | VQ-VAE 可量化到的离散向量数量；在压缩与保真间权衡 |
| Tokenizer ceiling | "Reconstruction limit" | 解码 VQ 令牌所能达到的最佳 PSNR；限定模型的图像质量上限 |

## 深入阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)  
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)  
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)  
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)  
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)