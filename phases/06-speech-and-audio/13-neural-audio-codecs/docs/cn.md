# Neural Audio Codecs — EnCodec, SNAC, Mimi, DAC and the Semantic-Acoustic Split

> 2026 年的音频生成几乎全部基于令牌。EnCodec、SNAC、Mimi 和 DAC 将连续波形转换为变换器可以预测的离散序列。语义与声学令牌的拆分——第一个码本为语义，其余为声学——是自 Transformer 以来对音频最重要的架构性变革。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 6 · 02（频谱图）, Phase 10 · 11（量化）, Phase 5 · 19（子词分词）  
**Time:** ~60 分钟

## 问题

语言模型作用于离散令牌。音频是连续的。如果你想做一个面向语音/音乐的类 LLM 模型——比如 MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——你首先需要一个**神经音频编解码器**：一个学习型编码器，把音频离散化为小词汇量的令牌，以及与之匹配的解码器用于重构波形。

目前出现了两大类：

1. **重构优先的编解码器** — EnCodec、DAC。优化感知音质。令牌是“声学”的——它们捕捉包括说话者身份、音色、背景噪声在内的所有信息。
2. **语义优先的编解码器** — Mimi（Kyutai）、SpeechTokenizer。通过从 WavLM 蒸馏等方法，强制第一个码本编码语言/语音学内容。后续码本为声学细节。

2024–2026 年的洞见：**纯重构型编解码器在从文本生成时会产生模糊的语音。** LLM 对编解码器令牌的建模必须同时学习语言结构和声学结构，这在同一码本里不可扩展。将两者分开——语义码本 0、声学码本 1–N——正是让 Moshi 和 Sesame CSM 起作用的关键。

## 概念

![四种编解码器格局：EnCodec、DAC、SNAC（多尺度）、Mimi（语义+声学）](../assets/codec-comparison.svg)

### 核心技巧：残差矢量量化（Residual Vector Quantization，RVQ）

与使用一个巨大码本（为达到高质量需要数百万码）不同，所有现代音频编解码器都使用**RVQ**：一系列小码本的级联。第一个码本对编码器输出进行量化；第二个码本对残差量化；以此类推。每个码本通常有 1024 个码。8 个码本 = 有效词汇量 1024^8 ≈ 10^24。

在推理时，解码器对每帧选择的所有码向量求和以重建波形。

### 2026 年重要的四个编解码器

**EnCodec（Meta，2022）。** 基线。基于波形的编码器-解码器，RVQ 瓶颈。24 kHz，最多 32 个码本，默认 4 个码本 @ 1.5 kbps。使用 `1D conv + transformer + 1D conv` 架构。被 MusicGen 使用。

**DAC（Descript，2023）。** 使用 L2 归一化码本、周期性激活函数和改进损失的 RVQ。是开源编解码器中重构保真度最高的——在 12 个码本时有时与原始语音无法区分。支持 44.1 kHz 全频带。

**SNAC（Hubert Siuzdak，2024）。** 多尺度 RVQ——粗糙的码本在比精细码本更低的帧率下工作。有效地以分层方式建模音频：在 ~12 Hz 的粗略“草图”加上 50 Hz 的细节。Orpheus-3B 使用它，因为分层结构很适合基于 LM 的生成。

**Mimi（Kyutai，2024）。** 2026 年的变革者。帧率 12.5 Hz（非常低），8 个码本 @ 4.4 kbps。第 0 号码本通过 **从 WavLM 蒸馏** 得到——训练目标是预测 WavLM 的语音内容特征。码本 1–7 为声学残差。这个拆分驱动了 Moshi（课时 15）和 Sesame CSM。

### 帧率对语言建模的重要性

较低的帧率 = 更短的序列 = 更快的 LM。

| Codec | 帧率 | 1 秒 = N 帧 | 适用场景 |
|-------|------:|------------:|---------|
| EnCodec-24k | 75 Hz | 75 | 音乐、通用音频 |
| DAC-44.1k | 86 Hz | 86 | 高保真音乐 |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM 高效建模 |
| Mimi | 12.5 Hz | 12.5 | 流式语音 |

在 12.5 Hz 下，10 秒语音只有 125 帧 —— 变换器可以轻松预测这些帧。

### 语义与声学令牌

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **语义令牌（Mimi 的第 0 号码本）。** 编码“说了什么”——音素、词、内容。通过辅助预测损失从 WavLM 蒸馏得到对齐。
- **声学令牌（码本 1–7）。** 编码音色、说话者身份、韵律、背景噪声、精细细节。

一个自回归 LM 首先预测语义令牌（在文本条件下），然后预测声学令牌（在语义 + 说话者参考条件下）。这种因式分解是现代 TTS 能进行零样本克隆说话者的原因：语义模型处理内容；声学模型处理音色。

### 2026 年的重构质量（比特率越低越好）

| Codec | 比特率 | PESQ | ViSQOL |
|-------|-------:|-----:|-------:|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

传统编解码器如 Opus 在感知质量上按比特率仍然占优。神经编解码器的优势在于**产生离散令牌**（Opus 无法生成）以及**生成模型友好性**（LLM 能用这些令牌做的事）。

## 搭建

### 步骤 1：使用 EnCodec 编码

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps（千比特每秒）

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames)，dtype=int64
```

`n_codebooks=8` 在 6 kbps 下。每个码的取值范围是 0–1023（10 位）。

### 步骤 2：解码并测量重构

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 步骤 3：语义-声学拆分（Mimi 风格）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # 形状 (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

第 0 号语义码本与 WavLM 对齐。你可以训练一个文本到语义（text-to-semantic）的变换器——词汇量远小于直接到音频的情形。然后一个单独的声学到波形的解码器在说话者参考条件下合成音频。

### 步骤 4：为什么在编解码器令牌上用自回归 LM 可行

对于 Mimi 的 12.5 Hz × 8 码本，一个 10 秒的语音片段：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 个令牌对变换器来说是一个很小的上下文。一个 2.56 亿参数的变换器在现代 GPU 上可以在毫秒级生成 10 秒语音。

## 使用建议

将问题映射到编解码器：

| 任务 | Codec |
|------|-------|
| 通用音乐生成 | EnCodec-24k |
| 最高保真重构 | DAC-44.1k |
| 基于自回归 LM 的语音（TTS） | SNAC 或 Mimi |
| 流式全双工语音 | Mimi（12.5 Hz） |
| 带文本的音效库 | EnCodec + T5 条件 |
| 精细音频编辑 | DAC + 修补（inpainting） |

经验法则：**如果你在构建生成模型，优先考虑 Mimi 或 SNAC。如果你在构建压缩管线，使用 Opus。**

## 陷阱

- **码本太多。** 增加码本线性提升保真，但也线性增加 LM 序列长度。一般在 8–12 个码本就够了。  
- **帧率不匹配。** 在 12.5 Hz 的 Mimi 上训练的 LM，然后在 50 Hz 的 EnCodec 上做微调会默默失败。  
- **假设所有码本等价。** 在 Mimi 中，码本 0 承载内容；丢失它会毁掉可懂度。丢失码本 7 几乎察觉不到。  
- **仅用重构质量作为唯一指标。** 一个编解码器可能在重构上表现优秀，但如果语义结构糟糕，基于 LM 的生成将毫无用处。

## 部署

将文件保存为 `outputs/skill-codec-picker.md`。为特定的生成或压缩任务选择合适的编解码器。

## 练习

1. 简单。运行 `code/main.py`。它实现了一个玩具的标量 + 残差量化器，并测量随着码本数增加的重构误差。  
2. 中等。安装 `encodec`，在一个保留的语音片段上比较 1、4、8、32 个码本的效果。绘制 PESQ 或 MSE 与比特率的关系图。  
3. 困难。加载 Mimi。编码一个片段。将码本 0 用随机整数替换；解码。然后同样替换码本 7。比较两种破坏——破坏码本 0 会毁掉可懂度；破坏码本 7 几乎不改变音频。

## 术语表

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| RVQ | Residual quantization | 一系列小码本的级联；每个码本对前一阶段的残差进行量化。 |
| 帧率 | Codec speed | 每秒多少个令牌帧。越低 = 对 LM 越快。 |
| 语义码本 | Codebook 0 (Mimi) | 从自监督学习特征蒸馏得到的码本；编码内容。 |
| 声学码本 | 其余码本 | 音色、韵律、噪声、精细细节。 |
| PESQ / ViSQOL | 感知质量 | 与 MOS 相关的客观评估指标。 |
| EnCodec | Meta codec | 基于 RVQ 的基线；被 MusicGen 使用。 |
| Mimi | Kyutai codec | 12.5 Hz 帧率；语义-声学拆分；驱动 Moshi。 |

## 相关阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — RVQ 基线。  
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — 开源中保真最高的方案。  
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — 多尺度 RVQ。  
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — 语义-声学拆分、WavLM 蒸馏。  
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — 语义/声学两阶段范式。  
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 最初的可流式 RVQ 编解码器。