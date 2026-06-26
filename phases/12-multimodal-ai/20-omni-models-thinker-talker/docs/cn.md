# Omni Models: Qwen2.5-Omni 和 Thinker-Talker 划分

> GPT-4o 在 2024 年 5 月的产品演示之所以具有颠覆性，并不是因为底层模型本身，而是因为产品形态——一个语音界面：你说话，模型看到摄像头视角，然后在不到 250ms 内回应。开源生态在 2024 和 2025 年余下时间都在竞速实现那个产品表面。Qwen2.5-Omni（2025 年 3 月）是参考开源设计：一个 Thinker（大型文本生成 transformer）加上一个 Talker（并行语音生成 transformer），通过流式语音 token 连接。Mini-Omni 做了简化，Moshi 达到相似延迟，GLM-4-Voice 将其扩展到中文。本课讲解 Thinker-Talker 架构以及使流式实时对话工作的延迟预算。

**Type:** 构建  
**Languages:** Python（stdlib，流式管道延迟模拟器 + VAD 循环）  
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)  
**Time:** ~180 分钟

## 学习目标

- 将推理管道拆分为 Thinker（文本推理）和 Talker（语音合成），并解释为什么并行流式能工作。  
- 逐组件计算对话交互的 time-to-first-audio-byte（TTFAB）预算。  
- 描述 TMRoPE 在 Thinker 中如何对视觉、音频和文本进行时间对齐的位置编码。  
- 说出三种实时对话模式：半双工、轮流发言、全双工。

## 问题概述

一个实时语音助手必须快速完成很多事情：

1. 听用户。实时语音分词、语音活动检测（VAD）以判断用户何时结束讲话。  
2. 可选地观看。摄像头输入 2–4 FPS，连同音频一起流入 Thinker。  
3. 思考。基于对话历史生成回复。  
4. 说话。合成语音 token，将其解码为波形，并流式输出到用户扬声器。

每一步都会增加延迟。要有对话感，总往返延迟需要 < 500ms —— 在此之下用户不会注意延迟。GPT-4o 声称约 ~250ms。Moshi 约 ~160ms。Qwen2.5-Omni 约 ~350–500ms。

所有组件都必须支持流式处理。不能“全部批次化后再解码”。

## 概念

### Thinker 和 Talker

Qwen2.5-Omni 的分解：

- Thinker：7B–80B 的文本生成 transformer。消耗交错的文本 + 图像 + 音频 token。输出表示要说内容的文本 token。  
- Talker：较小的语音生成 transformer（200M–1B）。消耗 Thinker 的文本输出 token 加上最近的语音上下文 token。输出离散的语音 token（residual-VQ 索引）。  
- 语音解码器：流式波形解码器（SNAC、MoVQGAN 家族），将语音 token 实时转为音频样本。

这种分离很重要。Thinker 需要大模型以获得良好的推理能力。Talker 可以小，因为其任务是局部的——把文本转换为语音 token。更大的 Talker 并不会更有表现力，反而更慢。

并行运行两者的流程：

1. Thinker 发出文本 token t_i。  
2. Talker 通过流式接收 t_i 并发出语音 token s_i, s_{i+1}, ..., s_{i+k}。  
3. 语音解码器实时消费语音 token 并输出音频样本。  
4. 当 Thinker 到达文本 token t_{i+3} 时，Talker 已经为 t_0..t_{i+2} 流式输出了音频。

### TMRoPE —— 时间对齐的多模态位置编码

Thinker 需要融合图像帧（例如 4 FPS）、音频帧（例如 50 帧/秒）和来自对话历史的文本。把所有图像先放再音频再文本的天真序列会丢失时间对齐信息。

TMRoPE 为每个 token 分配绝对时间戳。视觉 token 在 t=2.3s，音频 token 在 t=2.32s，用户的文本 token “停” 在 t=2.35s。RoPE 按时间戳旋转注意力；模型把它们看作在时间上同时发生。

这是实现“他边挥手边说你好”这种场景的基础——模型会在同一概念时间点同时看到视频帧和音频。

### 流式语音合成

语音 token 必须流式生成。Mini-Omni（Xie & Wu，2024）提出“语言模型可以在思考时听、说并流式进行”：Thinker 的输出 token 与 Talker 的输出 token 在相同序列中交错。Talker 在 Thinker 确认下一个文本 token 后立即触发。没有批次边界。

Moshi（Défossez 等，2024 年 10 月）是最快的开源实现。单 A100 实验中达到 160ms 的 TTFAB。其架构：一个 7B 的 transformer 在交替位置上发出文本和语音 token，并通过“内在独白（inner monologue）”将思考流和说话流分离。这实质上是将 Thinker + Talker 融合为一个模型，通过训练技巧实现。

### VAD 与轮流发言

语音活动检测运行在输入端。有两种模式：

- 半双工（Half-duplex）：用户说话，模型听；模型说话，用户听。通过 VAD 静音检测（约 200ms）实现清晰切换。  
- 全双工（Full-duplex）：双方可同时说话。模型可以做反馈性短语（“嗯嗯”）或打断，难度更大。Moshi 支持这一点。

Qwen2.5-Omni 默认支持半双工，通过静音阈值实现轮流发言。全双工需要应用层面的处理。

### Qwen3-Omni（2025 年 11 月）

继任者。Qwen3-80B Thinker、更大的 Talker、改进的 TMRoPE-v2。延迟接近 GPT-4o 的 250ms。开放权重。OmniBench 基准在与 Gemini 2.0 Live 的对比中具有竞争力。

### 生产延迟预算

典型流式交互的延迟拆分：

- 麦克风 -> 音频 token：40–80ms。  
- 预填（提示 + 历史）：7B 时约 100–200ms，70B 时远高于此。  
- 第一个 Thinker 文本 token：40ms。  
- Talker 处理第一个文本 token：20ms。  
- 第一个语音 token 提交：40ms。  
- Residual-VQ 解码：30ms。  
- 语音波形解码：50–80ms。

在 7B 时总 TTFAB：320–510ms；在 70B 时：600–900ms。前沿级质量通常意味着 70B+；因此存在前沿延迟差距。

### Token 速率计算

在 16kHz 的语音、以 50 Hz 为基准语音 token 的设定下，每秒输出需要 50 个语音 token。Talker 必须以 ≥50 tok/s 的速度发出 token 才能跟上。以典型 LLM 在 H100 上 30–80 tok/s 的吞吐量来看，一个小型（200–300M）的 Talker 足够快；7B 的 Talker 会跟不上。

这就是为什么出现小型专用 Talker，而不是“直接用主模型”的原因。

## 使用说明

`code/main.py`：

- 模拟一个具有伪造 token 发出速率的 Thinker-Talker 管道。  
- 计算可配置模型大小和麦克风采样率下的 TTFAB。  
- 演示带 VAD 静音阈值的半双工轮流发言。

## 部署建议

本课产生 `outputs/skill-omni-streaming-budget.md`。给定实时语音产品的目标 TTFAB 和功能集合（带视觉输入、双语、全双工），选择 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni 并确定 Thinker/Talker 的规模。

## 练习

1. 你的目标 TTFAB 是 300ms。使用 7B Thinker 和 300M Talker，写出每个组件的延迟分解。  
2. Qwen2.5-Omni 使用 TMRoPE。描述当用户在 t=1s 开始讲话而摄像头在 t=1.2s 捕捉到一个手势时，模型所“看到”的内容。  
3. 全双工支持要求模型在监听时发出音频。提出一种训练数据格式以教会模型这一点。  
4. 阅读 Moshi 论文第 4 节。描述“内在独白”分离以及它为何避免了 Thinker-Talker 的划分。  
5. 计算吞吐预算：要跟上 16kHz 语音且基层语音 token 速率为 50 tok/s，Talker 必须以多快的速率发出 token？

## 关键词

| 术语 | 人们如何描述 | 实际含义 |
|------|--------------|----------|
| Thinker | “推理大脑” | 生成要说内容的大型文本生成 transformer |
| Talker | “发声的嘴” | 从 Thinker 的文本生成离散语音 token 的小型 transformer |
| TTFAB | “延迟预算” | Time-to-first-audio-byte：从用户语音结束到首个音频样本输出的时间 |
| TMRoPE | “时间对齐的 RoPE” | 使用跨视觉、音频、文本的绝对时间戳的位置编码 |
| Half-duplex | “轮流发言” | 用户与模型交替发言；通过 VAD 静音检测判定用户结束 |
| Full-duplex | “同时发声” | 模型可在听的同时说话；支持反馈性短语 |
| Inner monologue | “Moshi 的分离” | 单模型设计，在交替位置上将思考流与说话流编排分离 |

## 延伸阅读

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)  
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)  
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)  
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)  
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)