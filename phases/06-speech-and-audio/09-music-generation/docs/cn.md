# 音乐生成 — MusicGen、Stable Audio、Suno 与许可地震

> 2026 年的音乐生成：Suno v5 和 Udio v4 在商业领域占据主导；MusicGen、Stable Audio Open 和 ACE-Step 引领开源阵营。技术问题基本解决。法律问题（Warner Music 与 Suno 的 5 亿美元和解、UMG 的和解）在 2025–2026 年重塑了该领域。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 6 · 02 (频谱图), Phase 4 · 10 (扩散模型)  
**Time:** ~75 分钟

## 问题定义

文本 → 30 秒到 4 分钟的音乐片段，包含歌词、人声和结构。有三个子问题：

1. **伴奏生成。** 文本如 “lo-fi hip-hop drums with warm keys” → 音频。代表模型：MusicGen、Stable Audio、AudioLDM。
2. **完整歌曲生成（含人声 + 歌词）。** “关于德州雨夜的乡村歌曲” → 完整歌曲。代表模型：Suno、Udio、YuE、ACE-Step。
3. **条件/可控生成。** 扩展现有片段、重生桥段、换流派、提取 stem、或做修补（inpaint）。Udio 的修补 + stem 分离是 2026 年重要的商用特性。

## 概念

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### 基于神经编解码器令牌的 Token LM

Meta 的 **MusicGen**（2023，MIT）及其众多衍生：以文本/旋律嵌入为条件，自回归地预测 EnCodec 令牌（32 kHz，4 个 codebook），然后用 EnCodec 解码。参数规模 300M - 3.3B。是强基线；超过 30 秒后表现欠佳。

**ACE-Step**（开源，4B XL，2026 年 4 月发布）将其扩展到整首歌的歌词条件生成。是开源社区中距离 Suno 最近的方案。

### 基于谱图或潜在空间的扩散模型

**Stable Audio (2023)** 与 **Stable Audio Open (2024)**：在压缩音频潜在空间上的扩散模型。擅长循环片段、音效设计、环境质感。不太适合有明确结构的整首歌曲。

**AudioLDM / AudioLDM2**：借鉴图像 T2I 的潜在扩散做文本到音频的生成，推广到音乐、音效、语音。

### 混合（生产级）— Suno、Udio、Lyria

权重闭源。很可能是 AR 编解码器 LM + 基于扩散的声码器，并带有针对人声 / 鼓 / 旋律的专门头。Suno v5（2026）是质量领先者（ELO 1293）。Udio v4 增加了修补与 stem 分离（低音、鼓、人声可分别下载）。

### 评估指标

- **FAD（Fréchet Audio Distance）。** 使用 VGGish 或 PANNs 特征计算生成音频与真实音频分布的嵌入级距离。越低越好。MusicGen small 在 MusicCaps 上约 4.5 FAD；SOTA 约 3.0。
- **音乐性（主观）。** 人类偏好评估。Suno v5（ELO 1293）领先。
- **文本-音频对齐。** 使用 CLAP 评估提示词与输出之间的一致性。
- **音乐性缺陷。** 节拍异常、声乐片段漂移、30 秒后结构丢失等。

## 2026 模型图谱

| 模型 | 参数量 | 时长 | 人声 | 许可 |
|------|--------|------|------|------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | yes | Apache-2.0 |
| YuE | 7B | &gt; 2 min | yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | yes, ELO 1293 | commercial |
| Udio v4 (closed) | ? | 4 min | yes + stems | commercial |
| Google Lyria 3 (closed) | ? | real-time | yes | commercial |
| MiniMax Music 2.5 | ? | 4 min | yes | commercial API |

## 法律环境（2025–2026）

- **Warner Music 与 Suno 的和解。** 5 亿美元。WMG 现在对 AI 模拟声线、音乐权利和 Suno 上的用户生成曲目具有监管权。Udio 与 UMG 的和解采取了类似安排。
- **欧盟 AI 法案** + **加州 SB 942**：要求对 AI 生成的音乐进行标注披露。
- **Riffusion / MusicGen** 在 MIT 许可下没有合规负担，但也没有商业人声能力。

可安全上线的模式：

1. 只生成伴奏（MusicGen、Stable Audio Open、MIT/CC0 输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music）并按次获取许可。
3. 在自有或已授权的曲库上训练（多数企业最终选择此路）。
4. 为生成内容打标签并嵌入水印与元数据。

## 开始构建

### 步骤 1：使用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种规模：`small`（300M，速度快）、`medium`（1.5B）、`large`（3.3B）。small 足以验证“想法是否成立”。

### 步骤 2：旋律条件生成

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen 的 melody 输入是色度图（chromagram），它会保留旋律走向同时替换音色。适用于 “把这段旋律做成弦乐四重奏” 的场景。

### 步骤 3：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish 嵌入距离。对流派级别的回归测试很有用；但不能替代人工听审。

### 步骤 4：将其加入 LLM-音乐工作流

结合 Lesson 7–8 的思路：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使用建议

| 目标 | 技术栈 |
|------|--------|
| 伴奏音色设计 | Stable Audio Open |
| 游戏/自适应音乐 | Google Lyria RealTime（闭源） |
| 含人声的整首歌（商业） | Suno v5 或 Udio v4，需明确许可 |
| 含人声的整首歌（开源） | ACE-Step XL 或 YuE |
| 短广告 Jingle | MusicGen 使用哼唱作为旋律条件 |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍会发生但需注意的问题

- **版权清洗的提示词。** “以 Taylor Swift 风格的歌曲”——Suno/Udio 的商业接口现在会过滤这类请求，开源模型通常不会。你应当添加自有的过滤名单。
- **30 秒后重复/漂移。** 自回归模型容易循环。可以交叉淡化多次生成的结果，或使用 ACE-Step 来获得更好的结构连贯性。
- **节拍漂移。** 模型会偏离 BPM。提示中加上 BPM 标签，并用 librosa 的 `beat_track` 做后处理过滤。
- **人声可懂度。** Suno 表现优秀；开源模型的歌词往往含糊。如果歌词非常重要，使用商业 API 或微调（Fine-tuning）。
- **单声道输出。** 开源模型常生成单声道或伪立体声。用专门的立体声重构工具（如 ezst、Cartesia 的立体声扩散）升级。

## 上线要点

保存为 `outputs/skill-music-designer.md`。确定模型、许可策略、时长/结构计划，以及面向音乐生成部署的披露元数据。

## 练习

1. **简单。** 运行 `code/main.py`。它会生成一个“生成式”的和弦进行 + 鼓点的 ASCII 表示 —— 如果愿意可以用任意 MIDI 渲染器播放。
2. **中等。** 安装 `audiocraft`，用 MusicGen-small 对 4 个风格提示生成 10 秒片段，针对某个参考流派集合测量 FAD。
3. **困难。** 使用 ACE-Step（或 MusicGen-melody），对同一旋律用不同音色提示生成三个变体。计算 CLAP 相似度以验证与提示的一致性。

## 关键术语

| Term | 大家怎么说 | 实际含义 |
|------|------------|---------|
| FAD | Audio FID | 真实音频与生成音频嵌入分布之间的 Fréchet 距离。 |
| Chromagram | Melody as pitches | 每帧 12 维的色度向量；作为旋律条件输入。 |
| Stems | Instrument tracks | 分离后的低音 / 鼓 / 人声 / 主旋律等轨道，通常为 WAV 文件。 |
| Inpainting | Regen a section | 在时间窗内掩码并仅由模型重生该片段。 |
| CLAP | Text-audio CLIP | 对比式音频-文本嵌入；用于评估文本与音频的一致性。 |
| EnCodec | Music codec | Meta 的神经编解码器，用于 MusicGen；32 kHz，4 个 codebook。 |

## 进一步阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开源自回归基准。  
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 音效设计的默认选择。  
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 开源 4B 整首歌曲生成器，2026 年 4 月发布。  
- [Suno v5 platform docs](https://suno.com) — 商业质量领先者。  
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 用于音乐与音效的潜在扩散方法。  
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025 年 11 月的先例报道。