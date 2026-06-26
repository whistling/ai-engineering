# Capstone 12 — Video Understanding Pipeline (Scene, QA, Search)

> Twelve Labs 将 Marengo + Pegasus 产品化。VideoDB 发布了面向视频的 CRUD API。AI2 的 Molmo 2 发布了开源 VLM 检查点。Gemini 的长上下文原生支持数小时视频。TimeLens-100K 在大规模上定义了时序定位。到 2026 年，管道形态已基本确定：场景分割、按场景生成字幕 + 嵌入、转录对齐、多向量索引，以及返回带有（start, end）时间戳和帧预览的查询结果。本结业项目的目标是摄取 100 小时视频、跑通公开基准，并评估计数与动作类问题的幻觉率。

**Type:** 结业项目  
**Languages:** Python (pipeline), TypeScript (UI)  
**Prerequisites:** 阶段 4 (CV), 阶段 6 (speech), 阶段 7 (transformers), 阶段 11 (LLM engineering), 阶段 12 (multimodal), 阶段 17 (infrastructure)  
**Phases exercised:** P4 · P6 · P7 · P11 · P12 · P17  
**Time:** 30 小时

## 问题

长视频问答是截至 2026 年最吃带宽的多模态问题。Gemini 2.5 Pro 能原生读取 2 小时视频，但要把 100 小时的视频摄入为可查询语料库，仍然需要场景级索引。生产形态通常包括场景分割（TransNetV2 或 PySceneDetect）、按场景用 VLM 生成字幕（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、转录对齐（Whisper-v3-turbo，带词级时间戳），以及一个并排存储字幕、帧嵌入和转录的多向量索引。查询管线返回带有（start, end）时间戳和帧预览的答案。

基准是公开的（ActivityNet-QA、NeXT-GQA）以及你自己的 100 条自定义问题集。计数问题和动作类问题是已知的难点；本结业项目要对这些类的幻觉进行明确度量。

## 概念

在摄入阶段同时并行运行三条流水线。场景分割将视频切成场景。VLM 生成每个场景的字幕并从关键帧提取帧嵌入。ASR 对音频做词级时间戳对齐。三条流通过 (scene_id, time range) 进行关联。每个场景在多向量索引（Qdrant）中有三种向量类型：字幕嵌入、关键帧嵌入、转录嵌入。

在查询时，自然语言问题同时在三类向量上进行检索；结果用 RRF 合并；一个时序定位适配器（TimeLens 风格）在顶级场景内精细化（start, end）窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收查询 + 顶级场景 + 裁剪帧，并返回带有引用时间戳和帧预览的答案。

对幻觉的测量很关键。计数（“多少人进入房间？”）和动作类（“厨师是在搅拌前倒入吗？”）的问题尤其不可靠。把这些类别的准确率与描述性问题分开报告。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割：TransNetV2（2024–26 年的 SOTA）或 PySceneDetect  
- ASR：使用 faster-whisper 调用 Whisper-v3-turbo 并导出词级时间戳  
- VLM 字幕器与答案器：Gemini 2.5 Pro、Qwen3-VL-Max 或 Molmo 2  
- 时序定位：基于 TimeLens-100K 训练的适配器或 VideoITG  
- 索引：支持多向量的 Qdrant（caption / frame / transcript）  
- UI：Next.js 15，HTML5 视频播放器与场景缩略图  
- 评估：ActivityNet-QA、NeXT-GQA、以及自定义 100 问手工标注集  
- 幻觉基准：对计数类与动作类子集进行手工标注评估

## 构建步骤

1. **摄入入口（Ingest walker）。** 接受 YouTube 链接或本地 MP4。必要时降采样到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect，输出 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k–8k 个场景。

3. **ASR 过程。** 在音频上运行 Whisper-v3-turbo；导出词级时间戳；按场景切分成片段转录。

4. **VLM 字幕化。** 对每个场景，用关键帧和简短的字幕模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max），生成字幕文本 + 帧嵌入。

5. **多向量索引。** 在 Qdrant 中建集合并命名 3 个向量。Payload 包含：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题分别在三类向量上做密集检索；用互惠秩融合（RRF）合并；top-k=5 场景。

7. **时序定位。** 在 top 场景上运行 TimeLens 风格的适配器，在场景内部精细化（start, end）窗口。

8. **VLM 合成。** 用查询 + top-3 场景剪辑（作为图像或短视频片段）+ 转录调用 Gemini 2.5 Pro。要求返回带有 `(video_id, start_ms, end_ms)` 的引用。

9. **评估。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100 问自定义集。报告总体准确率 + 各类别分解（计数、动作、描述性）。

## 使用示例

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付

`outputs/skill-video-qa.md` 是交付物。给定 YouTube URL 或上传的视频，管道将索引场景并以带时间戳引用的形式回答问题。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Temporal grounding IoU | 在保留的时序定位集合上计算交并比（IoU） |
| 20 | QA accuracy | NeXT-GQA 与自定义 100 问的准确率 |
| 20 | Ingest throughput | 每花费美元能摄取的视频小时数 |
| 20 | UI and citation UX | 时间戳链接、缩略图带、跳转到指定帧的体验 |
| 15 | Hallucination rate | 计数与动作类准确率分别报告 |
| **100** | | |

## 练习

1. 在字幕生成阶段用 Qwen3-VL-Max 替换 Gemini 2.5 Pro。对 50 个场景的人工评分样本报告字幕质量差异。

2. 将每场景的帧嵌入由多向量简化为一个 pooled 向量。衡量检索性能的回退（regression）。

3. 构建一个“计数严格”模式：合成器为每个被计数的实例提取时间戳，且用户需点击验证。测量用户验证是否降低幻觉率。

4. 基准摄取成本：在三种 VLM 选择下对比每美元可摄取的视频小时数，找出性价比最优解。

5. 添加说话人区分的转录：在音频上运行 pyannote 说话人分离，并对每位说话人的转录做嵌入。演示 “Alice 在 X 上说了什么？” 类问题的解答能力。

## 关键术语

| 术语 | 常说法 | 实际含义 |
|------|-------|---------|
| Scene segmentation | “Shot detection” | 在镜头边界处分割视频为场景 |
| Multi-vector index | “Caption + frame + transcript” | 在 Qdrant 集合中为不同表征命名向量 |
| Temporal grounding | “When exactly did it happen” | 为查询答案精细化（start, end）时间窗口（时间定位） |
| Frame embedding | “Visual representation” | 关键帧的向量嵌入；用于场景视觉相似性检索 |
| RRF fusion | “Reciprocal rank fusion” | 多个有序列表的合并策略；一种经典的混合检索技巧 |
| Counting hallucination | “Miscount” | VLM 在“多少 X”问题上的已知失败模式（计数幻觉） |
| ActivityNet-QA | “Video-QA benchmark” | 长视频问答准确率基准 |

## 进一步阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开源 VLM 检查点  
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 大规模时序定位实现  
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — 托管参考实现  
- [VideoDB](https://videodb.io) — 面向视频的 CRUD API 参考  
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业参考  
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型  
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源替代方案  
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评估基准