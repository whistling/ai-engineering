# Audio Transformers — Whisper Architecture

> 音频是随时间变化的频率图像。Whisper 是一个以梅尔谱图为输入并生成语音文本的 ViT（视觉变换器）。

**Type:** 学习
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 分钟

## 问题背景

在 Whisper（OpenAI, Radford 等，2022）出现之前，最先进的自动语音识别（ASR）通常是 wav2vec 2.0 和 HuBERT —— 自监督特征提取器加上微调的头。高质量但昂贵的数据流水线，对域敏感。多语言语音识别通常需要为每个语言族训练独立模型。

Whisper 做了三个关键押注：

1. **在所有数据上训练。** 从互联网上抓取了 68 万小时带弱标签的音频，覆盖 97 种语言。没有干净的学术语料库，也没有音素标签。
2. **单模型多任务。** 一个解码器通过任务令牌（task tokens）联合训练转录、翻译、语音活动检测、语言识别和时间戳标注。
3. **标准的编码器-解码器 Transformer。** 编码器消费对数梅尔谱图（log-mel spectrograms）。解码器自回归地产生文本标记。没有声码器（vocoder）、没有 CTC、没有 HMM。

结果：Whisper large-v3 在口音、噪声和零标注数据语言上都表现稳健。到 2026 年，它成为每个开源语音助手和大多数商用语音应用的默认语音前端。

## 概念解析

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### 第 1 步 — 重采样 + 分帧

音频采样率为 16 kHz。剪切/填充到 30 秒。计算对数梅尔谱图：80 个梅尔频带，步幅为 10 ms → 约 3,000 帧 × 80 个特征。这就是 Whisper 所看到的“输入图像”。

### 第 2 步 — 卷积干（convolutional stem）

两个 Conv1D 层，kernel=3，stride=2，将 ~3,000 帧降采样到 ~1,500 帧。在不增加太多参数的情况下将序列长度减半。

### 第 3 步 — 编码器

对于 large 版本，编码器是一个 24 层的 Transformer，运行在 1,500 个时间步上。使用正弦位置编码（sinusoidal positional encoding）、自注意力（self-attention）、GELU 前馈网络（FFN）。输出维度为 1,500 × 1,280 隐状态。

### 第 4 步 — 解码器

一个 24 层的 Transformer 解码器。它以自回归方式从 BPE 词汇表生成标记，该词汇表是 GPT-2 的超集，并包含一些音频专用的特殊标记。

### 第 5 步 — 任务令牌

解码器的 prompt 以控制令牌开头，告诉模型要执行的任务：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型在这种约定上训练。你通过前缀来控制任务。这相当于 2026 年在语音上的指令微调（instruction-tuning）。

### 第 6 步 — 输出

采用 beam search（宽度 5）并使用对数概率阈值。若未包含 `<|notimestamps|>` 标记，则模型每 0.02 秒预测一次时间戳。

### Whisper 各尺寸

| 模型 | 参数量 | 层数 | d_model | 注意力头数 | 显存 (fp16) |
|------|--------:|-----:|--------:|-----------:|------------:|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo（2024）将解码器从 32 层砍到 4 层。解码速度提升 8×，WER 回退不足 1 个百分点。这种解码速度的提升是 Whisper-turbo 在 2026 年成为实时语音代理默认选择的原因。

### Whisper 不做的事

- 不做说话人分割（谁在说话）。可与 pyannote 配合使用。
- 原生不支持实时流式 —— 固定 30 秒窗口。现代封装（如 `faster-whisper`, `WhisperX`）通过 VAD + 重叠实现流式。
- 不支持超过 30 秒的长上下文，除非外部分块。实际上效果不错，因为人类语音转录通常不需要很长远程上下文。

### 2026 年生态

| 任务 | 模型 | 备注 |
|------|------|------|
| 英语 ASR | Whisper-turbo, Moonshine | Moonshine 在边缘设备上速度快 4× |
| 多语言 ASR | Whisper-large-v3 | 覆盖 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可实现 ~150 ms 的延迟目标 |
| TTS（文本转语音） | Piper, XTTS-v2, Kokoro | 仍是编码器-解码器模式，但受 Whisper 形态影响 |
| 音频 + 语言模型 | AudioLM, SeamlessM4T | 在一个 Transformer 中同时使用文本标记与音频标记 |

## 构建流程

参见 `code/main.py`。我们不训练 Whisper —— 我们构建对数梅尔谱图流水线和任务令牌 prompt 格式化器。这些是在生产中你真正会接触到的部分。

### 第 1 步：合成音频

生成一个 1 秒、频率为 440 Hz 的正弦波，采样率 16 kHz。共 16,000 个采样点。

### 第 2 步：对数梅尔谱图（简化）

完整的梅尔谱图需要 FFT。这里用一个简化的分帧 + 每帧能量方法来展示流水线，而不依赖 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame = 25 ms，hop = 10 ms。与 Whisper 的窗口设置一致。这里用每帧能量来代替梅尔频带以作教学用途。

### 第 3 步：填充到 30 秒

Whisper 始终处理 30 秒的块。将谱图填充（或剪切）到 3,000 帧。

### 第 4 步：构建 prompt 令牌

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是完整的任务控制界面。一个 4 标记的前缀。

## 使用示例

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快、兼容 OpenAI 的用法：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

何时在 2026 年选择 Whisper：

- 需要单模型覆盖多语言 ASR。
- 在嘈杂、多样化音频下需要稳健转录。
- 研究或原型开发 ASR —— 最快的入门点。

何时选择其他方案：

- 在边缘设备上追求超低延迟流式 —— Moonshine 在相同性能下速度更快。
- 实时对话式 AI 需要 <200 ms 的端到端延迟 —— 选专用流式 ASR。
- 说话人分割 —— Whisper 不做，需要额外接入 pyannote。

## 部署（Ship It）

参见 `outputs/skill-asr-configurator.md`。该 skill 为新的语音应用选定 ASR 模型、解码参数和预处理流水线。

## 练习

1. 简单：运行 `code/main.py`。确认在 16 kHz、10 ms hop 下，1 秒信号的帧数约为 100；30 秒约为 3,000 帧。
2. 中等：使用 `numpy.fft` 构建完整的对数梅尔谱图。验证 80 个梅尔频带在数值误差范围内与 `librosa.feature.melspectrogram(n_mels=80)` 匹配。
3. 困难：实现流式推理：将音频切分为 10 秒窗口、2 秒重叠，分别对每块运行 Whisper，然后合并转录结果。在一段 5 分钟的播客样本上测量相对于单次完整传入的词错误率（WER）。

## 术语速览

| 术语 | 常说的说法 | 实际含义 |
|------|-----------|---------|
| Mel spectrogram | "Audio image" | 二维表示：一轴为频率箱（梅尔频带），另一轴为时间帧；每个单元为对数能量。 |
| Log-mel | "What Whisper sees" | 经过对数变换的梅尔谱图；近似人类对响度的感知。 |
| Frame | "One time slice" | 一个 25 ms 的样本窗口；以 10 ms 步幅重叠。 |
| Task token | "Prompt prefix for speech" | 解码器 prompt 中的特殊令牌，例如 `<\|transcribe\|>` / `<\|translate\|>`。 |
| Voice activity detection (VAD) | "Find the speech" | 去除静音的门控器；能大幅降低成本。 |
| CTC | "Connectionist Temporal Classification" | 经典的对齐无关训练损失；Whisper 不使用它。 |
| Whisper-turbo | "Small decoder, full encoder" | large-v3 的编码器 + 4 层解码器；解码更快。 |
| Faster-whisper | "The production wrapper" | 基于 CTranslate2 的重实现；int8 量化；比 OpenAI 的参考实现快 4×。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 参考代码与模型权重。阅读 `whisper/model.py` 可在 ~400 行内看到 Conv1D 干 + 编码器 + 解码器的从上到下实现。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 步骤 5–6 中描述的 beam search 与任务令牌逻辑；约 500 行，可读性强。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 先驱工作；在某些场景下仍是 SOTA 的特征提取方法。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产级封装，比参考实现快 4×。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年的边缘友好 ASR，受 Whisper 形态启发但更小巧。
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 标准的微调流程，包括梅尔谱图预处理器和标记-时间戳处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（编码器、解码器、交叉注意力、生成），与本课的架构图相对应。