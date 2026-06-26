# 视频-语言模型：时间令牌与定位

> 视频不是一堆照片的堆栈。一个 5 秒钟的剪辑包含因果顺序、动作动词和事件时序，这是单幅图像模型无法表示的。Video-LLaMA（Zhang 等，2023 年 6 月）推出了第一个具有视听定位的开源视频-LLM。VideoChat 和 Video-LLaVA 扩展了这一模式。到 2025 年，Qwen2.5-VL 的 TMRoPE 弥合了与前沿专有模型的差距。每个系统对时间令牌的处理不同 —— Q-former 每个剪辑、concat-pool 每帧、TMRoPE 每令牌。本文课程解读这些模式，构建一个 uniform-vs-dynamic 帧采样器，并在时间定位任务上评估。

**Type:** 构建  
**Languages:** Python（stdlib、帧采样器 + 时间定位评估器）  
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)  
**Time:** ~180 分钟

## 学习目标

- 解释为什么时间位置编码（temporal positional encoding）能在不改变视觉编码器的情况下独立影响视频 VLM 的性能。
- 比较均匀采样、动态 FPS 与事件驱动帧采样在每秒令牌数与定位准确率上的差异。
- 描述 Q-former-per-clip（Video-LLaMA）与 pooled-per-frame（Video-LLaVA）与 M-RoPE-per-token（Qwen2.5-VL）三种设计差异。
- 列出四个视频基准：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题

一分钟视频在 30 FPS 下有 1800 帧。以每帧 196 个视觉令牌（ViT-B 在 224 分辨率）计算，就是 352k 令牌 —— 超过任何 2024 年代 LLM 的上下文容量。

目前有三种降维策略：

1. 子采样帧（取 1–8 FPS，视内容而定）。
2. 对每帧的 patch 令牌进行激进池化（3x3 或 4x4 双线性池化）。
3. 通过 Q-former 压缩：对一个 16 帧剪辑输出 64 个令牌。

每种权衡不同。子采样丢失时间细节；池化丢失空间细节；Q-former 两者都略有损失但能节省令牌。

时间位置编码是另一条轴：模型如何知道第 5 帧在第 6 帧之前？选项包括简单的一维 temporal RoPE（Video-LLaMA）、学习的时间嵌入（Video-LLaVA）以及 TMRoPE（Qwen2.5-VL，全 3D）。

## 概念

### Video-LLaMA：每剪辑 Q-former + 音频分支

Video-LLaMA（2023）是第一个开源视频-LLM。架构要点：

- 16 帧剪辑，2 FPS（对应 8 秒）。
- 每帧经 ViT 提取特征 -> Video Q-former 对所有 16 帧做 cross-attention -> 32 个学习查询 -> LLM。
- 并行音频分支：波形 -> ImageBind 音频编码器 -> 音频 Q-former -> 32 查询 -> LLM。

优点：视听联合推理。缺点：剪辑长度固定，无法输出任意时间定位。

### VideoChat 与 Video-LLaVA

VideoChat 保留 Video-LLaMA 的思想，但去掉了音频并简化。Video-LLaVA（Lin 等，2023）在图像和视频帧上训练单一视觉编码器（“在投影前先对齐”），得到统一表示。两者都是冻结的 CLIP 编码器 + MLP + LLM。

两者都不能处理长视频，通常为 8–16 帧系统。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入了 TMRoPE —— Temporal-Modality Rotary Position Embedding。每个 patch 令牌携带 (t, h, w) 位置，其中 t 是实际时间戳（而非帧索引）。

与简单时间嵌入的关键区别：

- 使用绝对时间，而非索引。模型看到的是“在 4.2 秒时”，而不是“在第 15 帧”。
- 按令牌旋转，而非按剪辑旋转。每个视觉令牌由其时间戳独立旋转。
- 与动态 FPS 兼容。如果这里采样 2 FPS、那里采样 4 FPS，TMRoPE 能原生处理不均匀间隔。

TMRoPE 使得模型能回答“猫在第几秒跳跃？”这类问题，模型可以输出“在 4.2 秒”。Video-LLaMA 则只能说“在剪辑早期”。

### 帧采样策略

- 均匀（Uniform）：在时长内均匀采样 N 帧。简单，但会丢失运动峰值。
- 动态 FPS（Dynamic FPS）：基于运动强度自适应采样。光流或帧差用于在高运动段更密集采样。Qwen2.5-VL 在训练时采用该策略。
- 事件驱动（Event-driven）：先运行轻量检测器，在动作发生处采样更多帧。VideoAgent 使用此法。
- 关键帧 + 上下文：在镜头边界处采样关键帧，并取若干相邻帧。常用于电影内容。

### 每帧池化

在 1 FPS 且每帧 576 个令牌时，5 分钟剪辑有 172,800 个令牌。对于 Qwen2.5-VL-72B 的 128k 上下文仍然昂贵。

3x3 双线性池化可将每帧减少到 64 个令牌 -> 5 分钟约 19,200 令牌。对大多数任务而言这是一个折衷的甜点区间。

若在 agent 工作流中空间细节不重要，可更激进池化（6x6 -> 每帧 16 个令牌）。

### 四个视频基准

- VideoMME：全面的视频理解，包含短/中/长视频。
- TempCompass：细粒度时间推理的基准，关注“之前/之后”问题。
- EgoSchema：长时域第一视角视频评测。
- Video-MMMU：多模态多学科的视频问答。

完整的视频 VLM 评估需要覆盖这四个基准。它们侧重不同轴线 —— TempCompass 全是关于时序排序，EgoSchema 强调 3+ 分钟的长时域推理，VideoMME 跨越多种时长。

### 定位输出格式

时间定位的输出格式：

- 自由文本（Free text）："猫在大约第 4 秒跳跃。" 易于理解但不精确。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 使用此训练目标。
- 基于令牌（Token-based）：在答案中插入特定的 `<time>4.1</time>` 令牌。Qwen2.5-VL 的内部格式之一。

基于令牌的格式对下游最准确。Qwen2.5-VL 的 JSON 输出格式可直接解析。

### 2026 年最佳实践

到 2026 年的视频 VLM 推荐做法：

- 编码器：SigLIP 2 配合 M-RoPE 或 TMRoPE（Qwen2.5-VL）。
- 帧采样：动态 FPS（根据运动在 1–4 FPS 之间），并设置最大帧数上限。
- 每帧池化：3x3 双线性池化。
- 输出：包含时间和事件字段的结构化 JSON。
- 基准：VideoMME + TempCompass 用于通用评估；EgoSchema 用于长时域评估。

## 使用示例

`code/main.py` 包含：

- 均匀与动态 FPS 帧采样器。
- 一个玩具时间定位评估器：给定真值事件发生时间 T 与模型输出，使用容差计算准确度分数。
- 针对 Video-LLaMA（16 帧，Q-former）、Video-LLaVA（8 帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）进行比较。

## 交付件

本课会产出 `outputs/skill-video-vlm-frame-planner.md`。针对给定视频任务（监控、动作识别、时间定位、摘要），它会选出帧采样器、池化因子、输出格式与预期准确率等级。

## 练习

1. 对于一个 3 分钟的烹饪演示，选择均匀采样还是动态 FPS。并用令牌计数给出理由。

2. TMRoPE 相比简单的时间嵌入表具体增加了什么能力？

3. 为时间定位设计一个 JSON schema，要求 VLM 可以学习输出，包含错误情况的表示。

4. 阅读 Video-LLaVA 第 3 节“Alignment Before Projection”。为什么这比分别训练图像和视频编码器更好？

5. 给定 VideoMME 排行榜，到 2026 年开源最优模型与专有最优模型之间的差距是多少？其中有多少差距可归因于时间编码 vs 基础 LLM 规模？

## 关键词

| 术语 | 常说法 | 实际含义 |
|------|--------|----------|
| Temporal grounding | “时间定位答案” | VLM 输出事件发生的具体时间戳范围 |
| TMRoPE | “Time-Multimodal RoPE” | 带有绝对时间戳的 3D 旋转位置编码，由 Qwen2.5-VL 使用 |
| Dynamic FPS | “基于运动的采样” | 在高运动片段密集采样，在静止片段稀疏采样 |
| Frame pooling | “每帧空间压缩” | 在送入 LLM 前用双线性插值减少每帧 patch 数量 |
| Video Q-former | “剪辑压缩器” | 用 cross-attention 将 N 帧映射到 K 个学习查询的瓶颈模块 |
| VideoMME | “视频基准” | 全面的短/中/长视频基准，含 2500+ 样本 |

## 延伸阅读

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)