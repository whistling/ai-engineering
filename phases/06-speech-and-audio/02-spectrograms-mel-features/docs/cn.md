# 频谱图、梅尔刻度与音频特征

> 神经网络并不善于直接消费原始波形。它们消费频谱图。它们对梅尔频谱图的表现更好。到 2026 年，所有的 ASR、TTS 和音频分类器的成败都取决于这一项预处理选择。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 6 · 01（音频基础）  
**Time:** ~45 分钟

## 问题

取一个 10 秒、16 kHz 的音频片段。那是 160,000 个浮点数，值在 `[-1, 1]` 之间，几乎与标签“狗叫”或“单词 cat”不相关。原始波形中包含信息，但模型难以直接提取。两个相同的音素在相隔 100 ms 的位置上，其原始采样点完全不同。

频谱图解决了这个问题。它在人的感知忽略的时域细节处（微秒级抖动）进行压缩，并保留感知关注的结构（在大约 10–25 ms 的时间窗口内哪些频段有能量）。

梅尔谱图更进一步。人类对音高的感知是对数性的：100 Hz 与 200 Hz 的距离在感知上类似于 1000 Hz 与 2000 Hz。梅尔刻度对频率轴进行扭曲以匹配这种感知。梅尔刻度的谱图是从 2010 年到 2026 年语音机器学习中最重要的单一特征。

## 概念

![波形 到 STFT 到 梅尔谱图 到 MFCC 梯度](../assets/mel-features.svg)

**STFT（短时傅里叶变换）。** 将波形切成重叠的帧（典型：25 ms 窗口，10 ms 跳步 = 在 16 kHz 下为 400 样本 / 160 样本）。对每一帧乘以窗函数（默认是 Hann；Hamming 有稍微不同的权衡）。对每帧做 FFT。将幅度谱按时间堆叠成形状为 `(n_frames, n_freq_bins)` 的矩阵。这就是频谱图。

**对数幅度。** 原始幅度跨越 5–6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围。每个生产级流水线都使用对数幅度，而不是原始幅度。

**梅尔刻度。** 频率 `f`（Hz）到梅尔 `m` 的映射为 `m = 2595 * log10(1 + f / 700)`。在 1 kHz 以下近似线性，以上近似对数。覆盖 0–8 kHz 的 80 个梅尔箱是标准的 ASR 输入。

**梅尔滤波器组。** 一组在梅尔刻度上等距的三角形滤波器。每个滤波器是相邻 FFT 频点的加权和。将 STFT 幅度乘以滤波器组矩阵即可通过一次矩阵乘法得到梅尔谱图。

**对数梅尔谱图。** `log(mel_spec + 1e-10)`。Whisper 的输入、Parakeet 的输入、SeamlessM4T 的输入。2026 年通用的音频前端。

**MFCCs。** 对对数梅尔谱图应用 DCT（type II），保留前 13 个系数。特征去相关并进一步压缩。直到 ~2015 年这仍是主流特征，之后基于原始对数梅尔的 CNN/Transformer 赶上了它。仍在说话人识别（x-vectors、ECAPA）中使用。

**分辨率权衡。** 更大的 FFT 提供更好的频率分辨率但降低时间分辨率。25 ms / 10 ms 是音频 ML 的默认；音乐常用 50 ms / 12.5 ms；瞬态检测（鼓击、爆破音）用 5 ms / 2 ms。

```figure
spectrogram-window
```

## 实现它

### 第 1 步：对波形分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一个 10 秒、16 kHz 的片段，使用 `frame_len=400, hop=160` 会得到 998 帧。

### 第 2 步：Hann 窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在做 FFT 之前逐元素相乘。可以去除截断在非零端点时产生的谱泄漏。

### 第 3 步：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境会使用 `torch.stft` 或 `librosa.stft`（基于 FFT、向量化）。这里的循环是教学用途；在 `code/main.py` 中对短片段运行良好。

### 第 4 步：梅尔滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

在 `n_fft=400` 条件下，覆盖 0–8 kHz 的 80 个梅尔滤波器会得到形状为 `(80, 201)` 的矩阵。将形状为 `(n_frames, 201)` 的 STFT 幅度乘以其转置即可得到 `(n_frames, 80)` 的梅尔谱图。

### 第 5 步：对数梅尔

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代：`librosa.power_to_db`（参考归一化的 dB），或 `10 * log10(power + eps)`。Whisper 使用了更复杂的裁剪 + 归一化流程（参见 Whisper 的 `log_mel_spectrogram`）。

### 第 6 步：MFCCs

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每一帧的对数梅尔谱做 DCT，保留前 13 个系数。那就是你的 MFCC 矩阵。通常会丢弃第一个系数（它编码整体能量）。

## 使用

2026 年栈：

| 任务 | 特征 |
|------|------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 个对数梅尔，10 ms hop，25 ms 窗口 |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 个梅尔，5–12 ms hop 以实现精细的时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 个对数梅尔，10 ms hop |
| 说话人嵌入（ECAPA-TDNN、WavLM） | 80 个对数梅尔或原始波形的自监督表示 |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散 tokens（不是梅尔谱） |
| 关键词检测 | 用于微型设备的 40 维 MFCCs |

经验法则：**如果你不是做音乐，从 80 个对数梅尔开始。**任何偏离的证明责任在于提出偏离的人。

## 到 2026 年仍会上线的问题

- **梅尔数量不匹配。** 训练用 80 个梅尔，推理用 128 个梅尔。静默失败。记录两端的特征形状。
- **上游采样率不匹配。** 在 22.05 kHz 上计算的梅尔与 16 kHz 的梅尔看起来不同。在特征化之前修正采样率。
- **dB 与 log 的混淆。** Whisper 期望的是对数梅尔，而不是 dB 梅尔。有些 Hugging Face 流水线会自动检测；你自己的代码不会。
- **归一化漂移。** 训练时使用逐句归一化，推理时用全局归一化。生产 bug 可能会把 WER 翻倍。
- **填充泄漏。** 在片段末尾零填充会在尾部帧产生平坦谱。对称填充或复制填充。

## 部署

保存为 `outputs/skill-feature-extractor.md`。该技能为给定模型目标选择特征类型、梅尔箱数、帧/跳步和归一化方案。

## 练习

1. 简单：运行 `code/main.py`。它会合成一个啁啾信号（频率从 200 → 4000 Hz 扫频）并打印每帧的 argmax 梅尔箱。可选绘图，确认其匹配扫频。
2. 中等：对 `n_mels` 在 `{40, 80, 128}` 和 `frame_len` 在 `{200, 400, 800}` 的组合重新运行。测量时间轴上尖峰带宽。哪个组合对啁啾的分辨率最好？
3. 困难：实现 `power_to_db` 并比较在 AudioMNIST 上用小型 CNN 分类器的 ASR 准确率，使用 (a) 原始对数梅尔、(b) 以 `ref=max` 的 dB 梅尔、(c) MFCC-13 + delta + delta-delta。报告 top-1 准确率。

## 术语表

| 术语 | 人们怎么说 | 它真正的意思 |
|------|-----------|-------------|
| Frame | 切片 | 25 ms 的波形块，作为一次 FFT 的输入。 |
| Hop | 步幅 | 连续帧之间的样本数；10 ms 是 ASR 的默认值。 |
| Window | Hann/Hamming 的东西 | 点乘的乘子，用于将帧边缘渐零。 |
| STFT | 频谱图生成器 | 分帧 + 加窗后的 FFT；产生时间 × 频率矩阵。 |
| Mel | 扭曲的频率 | 符合感知的对数刻度；`m = 2595·log10(1 + f/700)`。 |
| Filterbank | 那个矩阵 | 将 STFT 投影到梅尔箱的三角形滤波器组。 |
| Log-mel | Whisper 的输入 | `log(mel_spec + eps)`；在 2026 年已标准化。 |
| MFCC | 传统特征 | 对数梅尔的 DCT；13 个系数，去相关。 |

## 深入阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文。  
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 梅尔刻度的原始论文。  
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现。  
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 与 hop/window 的参考。  
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 模型的生产级音频预处理流水线。