# 音频基础 — 波形、采样、傅里叶变换

> 波形是原始信号。声谱图是其表示形式。梅尔特征是对机器学习友好的形式。每个现代的 ASR 和 TTS 流水线都沿着这三级阶梯运行，而第一级是理解采样和傅里叶变换。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 1 · 06 (向量与矩阵), Phase 1 · 14 (概率分布)  
**Time:** ~45 分钟

## 问题

麦克风产生的是随时间变化的气压信号。你的神经网络消费的是张量。在它们之间存在一套约定，违反这些约定会产生难以察觉的错误：模型训练看起来没问题但 WER 翻倍，或者 TTS 输出嘶嘶声，或语音克隆系统记住了麦克风而不是说话者。

语音系统的每个错误都可归结为以下三个问题之一：

1. 数据是以什么采样率录制的，模型期望的采样率是多少？
2. 信号是否发生了混叠（aliasing）？
3. 你是在操作原始采样还是在频率表示上工作？

把这些问题弄对了，Phase 6 的其余内容就是可解的。弄错了，即使是 Whisper-Large-v4 也会产生垃圾输出。

## 概念

![波形、采样、DFT 与频率桶可视化](../assets/audio-fundamentals.svg)

**Waveform（波形）。** 一个一维的浮点数组，取值在 `[-1.0, 1.0]`。按采样编号索引。要转换为秒数，用采样率除：`t = n / sr`。在 16 kHz 下的 10 秒片段是一个包含 160,000 个浮点数的数组。

**Sampling rate (sr)。** 每秒采样次数。到 2026 年常见采样率：

| Rate | Use |
|------|-----|
| 8 kHz | 电话、旧式 VOIP。奈奎斯特在 4 kHz，损失辅音。ASR 不推荐。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都使用 16 kHz。 |
| 22.05 kHz | 较旧模型的 TTS 声码器训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**Nyquist-Shannon（奈奎斯特-香农采样定理）。** 采样率为 `sr` 时，能无歧义表示的最高频率为 `sr/2`。`sr/2` 边界称为*奈奎斯特频率*。高于奈奎斯特的能量会被*混叠*（折叠到较低频率），并破坏信号。下采样前务必先做低通滤波。

**Bit depth（位深）。** 16 位 PCM（带符号 int16，范围 ±32,767）是通用交换格式。音乐常用 24 位，内部 DSP 常用 32 位浮点。像 `soundfile` 这样的库会读取 int16，但返回的数组通常是 `[-1, 1]` 范围内的 float32。

**Fourier Transform（傅里叶变换）。** 任意有限信号都可以看作不同频率正弦波的叠加。离散傅里叶变换（DFT）对 `N` 个样本计算出 `N` 个复数系数——每个频率桶对应一个系数。`bin k` 对应频率 `k · sr / N` Hz。幅值是该频率的振幅，角度是相位。

**FFT。** 快速傅里叶变换：当 `N` 为 2 的幂时，DFT 的一种 `O(N log N)` 算法。每个音频库底层都用 FFT。以 16 kHz 采样、1024 点 FFT，会得到 512 个可用频率桶，覆盖 0–8 kHz，分辨率约为 15.6 Hz。

**Framing + window（分帧与窗函数）。** 我们不会对整个音频片段执行单次 FFT。通常将其切成重叠的*帧*（典型为 25 ms，步长 10 ms），将每帧乘以窗函数（Hann、Hamming）以消除边缘不连续，然后对每帧执行 FFT。这就是短时傅里叶变换（STFT）。Lesson 02 从这里继续。

```figure
mel-scale
```

## 动手实现

### 步骤 1：读取音频并绘制波形

`code/main.py` 仅使用标准库的 `wave` 模块以保持示例无依赖。生产环境中可使用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # 形状 (T,), sr=int
```

### 步骤 2：从头合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

在 16 kHz、时长 1 秒下合成 440 Hz 的正弦（音乐标准 A）会得到 16,000 个浮点数。用 `wave.open(..., "wb")` 以 16 位 PCM 编码写入文件。

### 步骤 3：手工计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

复杂度为 `O(N²)` —— 对 `N=256` 用来验证正确性没问题，但对真实音频则无用。实际代码会调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 步骤 4：找到主频率

幅度峰值索引 `k_star` 对应频率为 `k_star * sr / N`。在 440 Hz 正弦上运行此方法应该返回位于 `440 * N / sr` 的峰。

### 步骤 5：演示混叠

在 10 kHz 采样（奈奎斯特 = 5 kHz）下对 7 kHz 正弦采样。7 kHz 超出奈奎斯特，会折叠到 `10 − 7 = 3 kHz`。FFT 峰会出现在 3 kHz。这是经典的混叠演示，也是每个 DAC/ADC 都带陡峭低通滤波器的原因。

## 实际使用（2026 年你会部署的栈）

| Task | Library | Why |
|------|---------|-----|
| Read/write WAV/FLAC/OGG | `soundfile` (libsndfile wrapper) | 速度快、稳定，返回 float32。 |
| Resample | `torchaudio.transforms.Resample` or `librosa.resample` | 内建正确的抗混叠处理。 |
| STFT / Mel | `torchaudio` or `librosa` | 对 GPU 友好；属于 PyTorch 生态。 |
| Real-time streaming | `sounddevice` or `pyaudio` | 跨平台的 PortAudio 绑定。 |
| Inspect a file | `ffprobe` or `soxi` | 命令行工具，快速，报告 sr/声道/编解码信息。 |

决策规则：**在匹配任何其他内容之前先匹配采样率**。Whisper 期望 16 kHz 单声道 float32。给它 44.1 kHz 立体声，你会得到看起来像模型错误的垃圾输出。

## 交付

保存为 `outputs/skill-audio-loader.md`。该技能帮助你检查音频输入是否与下游模型的期望一致，并在不一致时正确重采样。

## 练习

1. **简单。** 在 16 kHz 下合成 1 秒的 220 Hz + 440 Hz + 880 Hz 混合信号。运行 DFT。确认在预期的频点看到三个峰值。
2. **中等。** 以 48 kHz 录制一段 3 秒的语音 WAV。用 `torchaudio.transforms.Resample`（带抗混叠）下采样到 16 kHz，再用简单的抽取（每隔三个样本取一个）下采样到 16 kHz。对两者做 FFT。混叠出现在何处？
3. **困难。** 仅使用 `math` 和步骤 3 的 DFT，从头构建 STFT。帧长 400，步长 160，Hann 窗。用 `matplotlib.pyplot.imshow` 绘制幅值。这就是 Lesson 02 的声谱图。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sample rate | How many samples per second | ADC 测量信号的频率，以 Hz 表示。 |
| Nyquist | The max frequency you can represent | `sr/2`；高于它的能量会被混叠回较低频段。 |
| Bit depth | Resolution of each sample | `int16` = 65,536 个等级；`float32` = 在 `[-1, 1]` 范围内约等于 24 位精度。 |
| DFT | The Fourier transform for sequences | `N` 个样本 → `N` 个复数频率系数。 |
| FFT | The fast DFT | `O(N log N)` 算法，通常要求 `N` 为 2 的幂。 |
| Bin | Frequency column | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | Spectrogram under the hood | 分帧 + 加窗后随时间变化的 FFT。 |
| Aliasing | Weird frequency ghosts | 高于奈奎斯特的能量镜像到较低的频率桶。 |

## 进一步阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 支持采样定理的论文。  
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费、经典的 DSP 教科书。  
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 含代码的实用入门教程。  
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 关于为什么真实世界音频不是干净正弦的参考书。  
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10 分钟内澄清频率桶直觉。