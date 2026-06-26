# Qwen-VL 家族与 Dynamic-FPS 视频

> Qwen-VL 家族 — Qwen-VL (2023)、Qwen2-VL (2024)、Qwen2.5-VL (2025)、Qwen3-VL (2025) — 在 2026 年成为最具影响力的开源视觉-语言模型谱系。每一代都做出一个决定性的架构押注，并在十二个月内被开源生态复制：原生动态分辨率通过 M-RoPE、带绝对时间对齐的 dynamic-FPS 采样、ViT 中的 window attention，以及结构化的 agent 输出格式。到 Qwen3-VL 时，配方已稳定：2D-RoPE-ViT 编码器支持原生宽高比输入、通过 MLP 投影到大型 Qwen3 语言基座，并且训练阶段把 OCR、定位与 agent 行为作为一等目标。本课按时间顺序解读该家族，帮助你理解每个旋钮为何在此处。

**Type:** 学习  
**Languages:** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)  
**Prerequisites:** Phase 12 · 06 (patch-n'-pack)  
**Time:** ~120 分钟

## 学习目标

- 计算 M-RoPE 的三轴旋转（时间、高度、宽度）并解释为何需要这三轴。
- 为一个视频挑选 dynamic-FPS 采样策略，并就 tokens-per-second 与事件检测准确率进行权衡推理。
- 按顺序列出四个 Qwen-VL 世代升级及每项升级的功能赋能。
- 设计一个 Qwen2.5-VL 风格的 JSON agent 输出格式，并从 VLM 响应中解析结构化的工具调用。

## 问题背景

Qwen-VL 于 2023 年 8 月发布，作为对 LLaVA-1.5 和 BLIP-2 的直接回应。Qwen 团队要解决的差距有三点：分辨率、视频和结构化输出。

分辨率：LLaVA-1.5 运行在 336x336。对于照片还行，但对于中文发票或密集的电子表格截图则无用。Qwen-VL 的首个创新是 448x448 和带定位边界框输出，让模型能指向具体对象。

视频：Video-LLaMA 对每帧分别编码并输入到 LLM。它适用于短片，但不适合以时间轴为信号的多分钟视频。Qwen 团队希望一个能理解时间的单一编码器。

结构化输出：LLaVA 输出自由文本。agent 需要 JSON。Qwen-VL 在训练中使用了显式的 JSON 输出格式，包括把边界框坐标作为文本输出。

每一代 Qwen-VL 都在这三条轴线中的一条或多条上进行扩展。

## 概念解析

### Qwen-VL（2023 年 8 月）

第一代：使用 OpenCLIP ViT-bigG/14 作为编码器（2.5B 参数），LLama 兼容的 Q-Former（1 步 256 个 query），Qwen-7B 基座。贡献点：

- 448x448 分辨率（当时开源 VLM 的 SOTA）。
- Grounding：在图文对上训练，输出显式坐标 token。例如 "The cat is at <box>(112, 204), (280, 344)</box>"。
- 从一开始就进行中文+英文的多语种训练。

当时基准：在英语上能和 GPT-4V 竞争，在中文上占主导。定位监督是亮点。

### Qwen2-VL（2024 年 9 月）— M-RoPE 与原生分辨率

Qwen2-VL 用原生动态分辨率的 ViT 编码器替代了固定分辨率 + Q-Former 堆栈。关键变化：

- 原生动态分辨率。ViT 接受任意可被 28 整除的 HxW（patch 14 并做 2x 空间合并）。例如 1120x672（合并后 40x24 patch）会生成 960 个视觉 token。无需重缩放、切片或缩略图。
- M-RoPE（Multimodal RoPE）。每个 token 带有一个 3D 位置 (t, h, w)，而非 1D。图像的 t=0，视频的 t=帧索引。RoPE 对 query/key 向量按每个轴的频率做旋转。无需位置嵌入表。
- MLP projector。舍弃 Q-Former；对合并的 patch tokens 使用两层 MLP。
- 支持带动态 FPS 的视频。默认按 1-2 FPS 采样，但模型接受任意帧数。

结果：Qwen2-VL-7B 在若干多模态基准上匹配 GPT-4o，并在 DocVQA 上胜出（94.5 vs 88.4）。架构改变是决定性举措。

### Qwen2.5-VL（2025 年 2 月）— dynamic FPS + 绝对时间

Qwen2.5-VL 的重大转变在于视频。dynamic FPS 不只是“在需要时多采样”。论文形式化了：

- 绝对时间 tokens。不是使用位置索引（帧 0、1、2…），而是使用实际时间戳。比如 “At 0:04, the cat jumps.” 模型看到在帧 token 间插入的 `<time>0.04</time>` token。
- Dynamic FPS。对慢动作素材以 1 FPS 采样，对动作片段以 4+ FPS。用户或训练者可选择；M-RoPE 自适应处理。
- ViT 中的 window attention。空间注意力在局部窗口内，提升吞吐；每隔若干层加入全局注意力。
- 明确的 JSON 输出格式。在工具调用数据上训练："{\"tool\": \"click\", \"coords\": [380, 220]}". 开箱即用的 agent-ready 输出。
- MRoPE-v2 缩放。位置随最大输入尺寸缩放，从而避免 10 分钟视频耗尽频率范围。

基准：Qwen2.5-VL-72B 在大多数视频基准上击败 GPT-4o，在文档任务上与 Gemini 2.0 持平，并为 GUI 定位（ScreenSpot）设定了开源 SOTA（84% 准确率 vs GPT-4o 的 38%）。

### Qwen3-VL（2025 年 11 月）

Qwen3-VL 是一次增量升级，侧重整合而非重构：更大的 LLM 主干（Qwen3-72B）、更广的训练数据、改进的 OCR、更强的推理通过 Qwen3 的“思考模式”。ViT 与 M-RoPE 保持不变。论文重点在于数据与训练改进而非架构创新。

谱系结论：到 2025 年 Qwen-VL 架构已稳定，后续世代主要扩展算力与数据，而不是原语。

### M-RoPE 的数学原理

经典 RoPE 对维度为 d 的 query `q`，按位置 `m` 使用配对坐标旋转：

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE 将隐藏维度拆成三段。假设 `d = 96`。将 32 维分配给时间轴、32 维分配给高度、32 维分配给宽度。每段按其轴位置进行旋转。位于 (t=5, h=10, w=20) 的 patch 在其三段上分别应用 `R_t(5)`、`R_h(10)`、`R_w(20)`。

文本 token 使用 `t = text_index, h = 0, w = 0`（或归一化的选择），以保持兼容性。视频帧使用 `t = frame_time, h = row, w = col`。单张图像使用 `t = 0`。

好处：一个位置编码能同时处理文本、图像与视频，不需要分支代码或不同的位置表。

### Dynamic-FPS 采样逻辑

给定时长为 `T` 秒且目标 token 预算为 `B` 的视频：

1. 计算你能承受的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 中选择满足 `fps <= fps_max` 的目标 FPS。
3. 若运动强（基于光流启发或用户显式请求），选择更高 FPS；若运动弱，则选择更低。
4. 以选定 FPS 均匀采样；在帧之间插入 `<time>t</time>` tokens。

Qwen2.5-VL 在训练时隐式学会了这套逻辑；推理时用户通过 `fps` 参数控制。一个 60 秒的动作序列按 4 FPS 且每帧 81 token = 19440 token，在 32k 上下文中是可控的。

### 结构化 agent 输出

Qwen2.5-VL 的 agent 训练显式针对结构化工具调用，例如：

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：对模型输出执行 JSON.parse。相比之下，自由文本 "click at (1024, 512)" 需要正则与模糊处理。正是这次变更，将 Qwen2-VL 的 ScreenSpot 分数从 55% 提升到 84%。

## 使用方法

`code/main.py` 实现了：

- 对混合文本、图像 patch 与视频帧的打包序列计算 M-RoPE 位置。
- Dynamic-FPS 采样器：给定（duration, budget, motion_level），选择 FPS 并输出帧时间戳。
- 一个玩具级的 Qwen2.5-VL JSON 输出解析器，处理带坐标字段的工具调用响应。

运行它，然后在 5 分钟视频上将固定 FPS 替换为 dynamic-FPS，感受差异。

## 部署指南

本课输出 `outputs/skill-qwen-vl-pipeline-designer.md`。给定一个视频任务（监控、agent、动作识别、无障碍），它会生成 Qwen2.5-VL 的配置（帧预算、FPS 策略、window-attention 标志、agent 输出模式）和延迟估计。每次为视频产品部署 Qwen-VL 家族模型时请使用此工具。

## 练习

1. 计算隐藏维 48（每段 16）时，位于 (t=3, h=5, w=7) 的 patch 的 M-RoPE 旋转。展示每段前面三个 pair 的旋转角度。
2. 一段 10 分钟的监控录像以 1 FPS 产生多少帧？在 384 分辨率并做 3x 池化时总共多少 tokens？Qwen2.5-VL 默认的 32k 上下文能处理吗？
3. 为 30 秒网球来回、30 秒菜谱演示、30 秒 UI-agent 录制选择 FPS。根据 dynamic-FPS 逻辑为每种情况说明理由。
4. Qwen2.5-VL 完全移除了 Q-Former。为什么在 2025 年简单的 MLP 可以奏效而在 2023 年不行？（提示：数据规模与编码器质量。）
5. 将三段 Qwen2.5-VL 的 JSON 工具调用输出解析成 Python dict。对于格式错误的 JSON 会失败什么、Qwen 食谱建议采用什么恢复策略？

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| M-RoPE | "Multimodal RoPE" | 在隐藏维中按时间、高度和宽度段划分的 3D 旋转位置嵌入 |
| Dynamic FPS | "Smart sampling" | 根据运动、时长和 token 预算为每个视频选择的帧采样率 |
| Absolute time token | "Timestamp token" | 在序列中交错的 `<time>t</time>`，让模型看到实际秒数而非帧索引 |
| Window attention | "Local attention" | 将空间自注意力限制在小窗口内以加速；并周期性加入全局注意力 |
| Structured agent output | "JSON mode" | 训练监督教会 VLM 输出可解析的 JSON，包含坐标和工具名 |
| min_pixels / max_pixels | "Resolution bounds" | 每次请求的 Qwen2.5-VL 控制总像素数从而控制 token 数量 |
| Grounding | "Point-at-it" | 以文本 token 输出边界框坐标；自 Qwen-VL v1 起已采用 |

## 延伸阅读

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)  
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)  
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)  
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)  
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)