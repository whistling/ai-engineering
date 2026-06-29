# Text-to-Speech (TTS) — From Tacotron to F5 and Kokoro

> ASR 将语音转为文本；TTS 将文本转为语音。到 2026 年的栈分为三部分：文本 → tokens、tokens → mel、mel → 波形。每一部分都有一个适合在笔记本上运行的默认模型。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** 阶段 6 · 02 (频谱图 & Mel), 阶段 5 · 09 (Seq2Seq), 阶段 7 · 05 (完整 Transformer)  
**Time:** ~75 分钟

## 问题

你有一段字符串："Please remind me to water the plants at 6 pm." 你需要生成一个 3 秒的音频片段，听起来自然、韵律（停顿、重音）正确，“plants”的元音发音准确，并且在 CPU 上用于实时语音助手时推理时间低于 300 ms。你还要能够切换音色、处理混合语言输入（例如 "remind me at 6 pm, daijoubu?"），并且在读人名时不要出丑。

现代 TTS 管道通常如下：

1. **文本前端。** 归一化文本（日期、数字、邮箱），转换为音素或子词 token，预测韵律特征。
2. **声学模型。** 文本 → 梅尔谱。代表工作：Tacotron 2 (2017)、FastSpeech 2 (2020)、VITS (2021)、F5-TTS (2024)、Kokoro (2024)。
3. **声码器。** 梅尔 → 波形。WaveNet (2016)、WaveRNN、HiFi-GAN (2020)、BigVGAN (2022)、2024 年以后的神经编解码器声码器。

到 2026 年，声学模型与声码器的分界在端到端扩散和流匹配模型中变得模糊。但用于调试的心理模型仍然是这三部分。

## 概念

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017)。** 序列到序列：字符嵌入 → BiLSTM 编码器 → 位置敏感注意力 → 自回归 LSTM 解码器产生梅尔帧。速度慢（AR），长文本时表现不稳。仍被作为基线引用。

**FastSpeech 2 (2020)。** 非自回归。时长预测器输出每个音素对应的梅尔帧数量。一遍式推理，速度比 Tacotron 快约 10×。牺牲了一些自然度（对齐更单调），但广泛部署。

**VITS (2021)。** 联合训练编码器 + 基于流的时长模型 + HiFi-GAN 声码器，端到端使用变分推断。高质量、单模型。2022–2024 年开源 TTS 的主导方案。变体：YourTTS（多说话人零样本）、XTTS v2 (2024, Coqui)。

**F5-TTS (2024)。** 基于流匹配的扩散 Transformer。韵律自然，使用 5 秒参考音频即可零样本语音克隆。2026 年开源 TTS 排行榜首位。335M 参数。

**Kokoro (2024)。** 小型（82M），可在 CPU 上运行，适用于实时的最佳英文 TTS。闭词汇、仅限英语，Apache-2.0 许可。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商用最先进方案。ElevenLabs v2.5 的情感标签（"[whispered]"、"[laughing]"）和角色音色在 2026 年主导有声书制作。

### 声码器演进

| Era | Vocoder | Latency | Quality |
|-----|---------|---------|---------|
| 2016 | WaveNet | offline only | SOTA at release |
| 2018 | WaveRNN | ~realtime | good |
| 2020 | HiFi-GAN | 100× realtime | near-human |
| 2022 | BigVGAN | 50× realtime | generalizes across speakers/langs |
| 2024 | SNAC, DAC (neural codecs) | integrated with AR models | discrete tokens, bit-efficient |

到 2026 年，大多数 “TTS” 模型已经实现从文本到波形的端到端；梅尔谱成为内部表示。

### 评估

- **MOS (Mean Opinion Score)。** 1–5 量表，众包评分。仍是金标准；速度极慢。
- **CMOS (Comparative MOS)。** A 对比 B 的喜好。每次标注的置信区间更窄。
- **UTMOS、DNSMOS。** 无参考的神经 MOS 预测器。用于排行榜评估。
- **CER (Character Error Rate) via ASR。** 将 TTS 输出通过 Whisper，计算输出文本与输入文本的 CER。作为可懂度的代理指标。
- **SECS (Speaker Embedding Cosine Similarity)。** 语音克隆质量指标。

2026 年 LibriTTS test-clean 的一些数值：

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 构建它

### 第 1 步：音素化输入

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 返回: 'həloʊ wɜːld'
```

音素是通用的桥梁。避免将原始文本直接喂给低于 VITS 级别质量的模型。

### 第 2 步：运行 Kokoro（2026 年 CPU 默认）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单文件，82M 参数。

### 第 3 步：用 F5-TTS 做语音克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入 5 秒参考音频及其转录；F5 会克隆韵律与音色。

### 第 4 步：从头实现 HiFi-GAN 声码器

太大，不适合教程脚本，但结构如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 个上采样模块，总共 256x，从梅尔率到音频率
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> 波形
```

训练：对抗训练（判别器在短窗口上）+ 梅尔谱重建损失 + 特征匹配损失。已商品化 —— 使用 `hifi-gan` 仓库或 nvidia-NeMo 的预训练检查点。

### 第 5 步：完整管道（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 使用它

2026 年的技术选取建议：

| Situation | Pick |
|-----------|------|
| Real-time English voice assistant | Kokoro (CPU) or XTTS v2 (GPU) |
| Voice cloning from 5 s reference | F5-TTS |
| Commercial character voices | ElevenLabs v2.5 |
| Audiobook narration | ElevenLabs v2.5 or XTTS v2 + fine-tune |
| Low-resource language | Train VITS on 5–20 h target-lang data |
| Expressive / emotion tags | ElevenLabs v2.5 or StyleTTS 2 fine-tune |

开源领导者（截至 2026 年）：**F5-TTS 代表质量，Kokoro 代表效率**。除非你是历史研究者，否则别再去用 Tacotron。

## 陷阱

- **没有文本归一化。** "Dr. Smith" 可能会读成 "Doctor" 还是 "Drive"？"2026" 是读作 "twenty twenty six" 还是 "two zero two six"？在音素化之前先归一化。
- **OOV 专有名词。** "Ghumare" → "ghyu-mair"？为未知 token 提供回退的字形到音素（G2P）模型。
- **削波。** 声码器输出很少裁剪，但推理时梅尔缩放不匹配可能会超出 ±1.0。始终使用 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24 kHz；下游管道期望 16 kHz → 需要重采样，否则会产生混叠。

## 上线部署

保存为 `outputs/skill-tts-designer.md`。为指定的音色、延迟和语言目标设计一个 TTS 管道。

## 练习

1. **简单。** 运行 `code/main.py`。从玩具词表构建音素字典，估计每个音素的时长，并打印伪造的“梅尔”时间表。
2. **中等。** 安装 Kokoro，用 `af_bella` 和 `am_adam` 两个音色合成相同句子。比较音频时长与主观质量差异。
3. **困难。** 录制一段 5 秒的参考音频。使用 F5-TTS 克隆它。报告参考与克隆输出之间的 SECS。

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Phoneme | Sound unit | 抽象的音类；英语约有 39 个（ARPABet）。 |
| Duration predictor | How long each phoneme lasts | 非自回归模型输出；每个音素的整数帧数。 |
| Vocoder | Mel → waveform | 将梅尔谱映射到原始采样点的神经网络。 |
| HiFi-GAN | Standard vocoder | 基于 GAN；2020–2024 年主流。 |
| MOS | Subjective quality | 来自人工评审员的 1–5 主观得分。 |
| SECS | Voice-clone metric | 目标与输出说话人嵌入的余弦相似度。 |
| F5-TTS | 2024 open-source SOTA | 流匹配扩散；零样本克隆。 |
| Kokoro | CPU English leader | 82M 参数模型，Apache 2.0 许可。 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — 序列到序列基线。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 端到端基于流的模型。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前开源 SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 截至 2026 年仍在使用的声码器。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 年的 CPU 友好英文 TTS。