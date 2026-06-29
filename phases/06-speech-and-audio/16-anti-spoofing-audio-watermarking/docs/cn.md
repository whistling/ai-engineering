# Voice Anti-Spoofing & Audio Watermarking — ASVspoof 5, AudioSeal, WaveVerify

> Voice cloning shipped faster than defenses. 2026 production voice systems need two things: a detector (AASIST, RawNet2) that classifies real vs fake speech, and a watermark (AudioSeal) that survives compression and editing. Ship both or do not ship voice cloning.

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 6 · 06 (说话人识别), Phase 6 · 08 (语音克隆)  
**Time:** ~75 分钟

## 问题概述

三个相关的防御层：

1. **反欺骗 / 深度伪造检测。** 给定一段音频，判断是合成的还是真实的？ASVspoof 基准（ASVspoof 2019 → 2021 → 5）是这一领域的权威。
2. **音频水印。** 在生成的音频中嵌入无法察觉的信号，供之后检测器提取。AudioSeal（Meta）和 WavMark 是开源选项。
3. **可验证的溯源。** 对音频文件及其元数据进行加密签名。C2PA / Content Authenticity Initiative。

检测面向不配合的对手。水印面向合规性——AI 生成的音频应可被识别。到 2026 年，两者都是必须的。

## 概念

![反欺骗、音频水印与溯源 — 三层防御](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024–2025 的基准

与以前版本相比的最大变化：

- **众包数据**（非录音棚干净音频）——更贴近真实场景。
- **~2000 名说话人**（之前约 ~100）。
- **32 种攻击算法。** 包括 TTS、语音转换和对抗扰动。
- **两条赛道。** Countermeasure (CM) 独立检测；Spoofing-robust ASV (SASV) 针对生物特征系统。

ASVspoof 5 上的 SOTA：约 7.23% EER。旧版 ASVspoof 2019 LA：0.42% EER。实际部署中：对野外采集的片段预计 5–10% EER。

### AASIST 与 RawNet2 — 检测模型家族

**AASIST**（2021，持续更新至 2026）。对频谱特征使用图注意力（graph-attention）。当前在 ASVspoof 5 countermeasure 任务上为 SOTA。

**RawNet2。** 基于原始波形的卷积前端 + TDNN 骨干。作为更简单的基线；通过微调仍具有竞争力。

**NeXt-TDNN + SSL 特征。** 2025 年的变体：ECAPA 风格 + WavLM 特征 + focal loss。在 ASVspoof 2019 LA 上达到了 0.42% EER。

### AudioSeal — 2024 年的默认水印

Meta 的 **AudioSeal**（2024 年 1 月，v0.2 于 2024 年 12 月）。关键设计点：

- **局部化检测。** 在 16 kHz 采样率下按帧检测水印（时间分辨率 1/16000 s）。
- **生成器 + 检测器联合训练。** 生成器学习嵌入不可闻信号；检测器通过数据增强学习鲁棒检测。
- **鲁棒。** 能抵抗 MP3 / AAC 压缩、均衡、±10% 速度变化、混噪（+10 dB SNR）。
- **快速。** 检测器运行速度为实时的 485×；比 WavMark 快 1000×。
- **容量。** 每条话语可嵌入 16 位有效载荷（可编码模型 ID、生成时间戳、用户 ID）。

### WavMark

AudioSeal 之前的开源基线。可逆神经网络，32 bits/sec。问题：

- 同步（synchronization）暴力破解慢。
- 可被高斯噪声或 MP3 压缩移除。
- 不适合实时场景。

### WaveVerify（2025 年 7 月）

解决了 AudioSeal 在时序操作（反转、速度变换）上的弱点。使用基于 FiLM 的生成器 + Mixture-of-Experts 检测器。在标准攻击下与 AudioSeal 竞争并可以处理时序编辑。

### 对手利用的缺口

来自 AudioMarkBench 的结果："在移调（pitch shift）条件下，所有水印的比特恢复准确率均低于 0.6，表明几乎被完全移除。" **移调是通用攻击。** 到 2026 年，没有一个水印能够完全鲁棒地对抗激进的音高修改。这就是为什么需要同时部署检测（如 AASIST）和水印。

### C2PA / Content Authenticity Initiative

这不是 ML 技术——而是一种清单（manifest）格式。音频携带关于创建工具、作者、日期的加密签名元数据。Audobox / Seamless 采用该规范。适用于溯源；但当坏人重新编码并剥离元数据时，C2PA 无能为力。

## 实作

### 第 1 步：一个简单的谱特征检测器（玩具）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音通常在高频能量上异常平坦。生产级检测会使用 AASIST，而不是这个玩具实现。但直觉是成立的。

### 第 2 步：AudioSeal 嵌入 + 检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: 在 [0, 1] 之间的浮点数 — 表示水印存在的概率
# decoded_payload: 16 位；将其与嵌入的 payload 比对
```

### 第 3 步：评估 — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 第 4 步：生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成都应输出： (1) 水印，(2) 签名清单，(3) 符合保留策略的审计日志。

## 使用场景

| 用例 | 防御措施 |
|------|---------|
| 发布 TTS / 语音克隆 | 对每个输出嵌入 AudioSeal（水印为必选） |
| 生物特征语音解锁 | AASIST + ECAPA 集成；加入活体检测挑战（liveness challenge） |
| 呼叫中心欺诈检测 | 对 20% 的来电样本运行 AASIST |
| 播客真实性 | 上传时使用 C2PA 签名；若为 AI 生成则同时嵌入 AudioSeal |
| 研究 / 训练检测器 | 使用 ASVspoof 5 的训练/开发/评估集 |

## 常见陷阱

- **只嵌水印但从未运行检测器。** 毫无意义。把检测器纳入 CI。
- **检测器未做校准。** AASIST 在 ASVspoof LA 上训练容易过拟合；真实世界准确率会下降。请根据你的领域做校准。
- **移调缺口（Pitch-shift gap）。** 激进的移调可以移除大部分水印。准备检测回退机制。
- **剥离元数据并重新托管。** C2PA 可以通过重新编码轻易绕过。始终将加密签名与感知性防御（水印）结合使用。
- **将活体作为唯一检测手段。** 要求用户说随机短语可以防止重放攻击，但不能防止实时克隆。

## 部署

保存为 `outputs/skill-spoof-defender.md`。为语音生成部署选择检测模型、水印方案、溯源清单和可执行的运维手册。

## 练习

1. **简单。** 运行 `code/main.py`。在合成音频上运行玩具检测器 + 玩具水印嵌入/检测。
2. **中等。** 安装 `audioseal`，在 TTS 输出中嵌入 16 位有效载荷并重新解码。对音频加入噪声并测量比特恢复准确率（Bit Recovery Accuracy）。
3. **困难。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量 EER。对一组 F5-TTS 生成的保留测试集进行测试 —— 观察 OOD 检测如何退化。

## 术语

| 术语 | 常说的含义 | 实际含义 |
|------|-----------|---------|
| ASVspoof | 基准 | 双年挑战；2024 = ASVspoof 5。 |
| CM (countermeasure) | 检测器 | 分类器：真实语音 vs 合成/转换语音。 |
| SASV | 说话人验证 + CM | 集成的生物识别 + 伪造检测。 |
| AudioSeal | Meta 的水印方案 | 局部化，16 位有效载荷，比 WavMark 快 485×。 |
| Bit Recovery Accuracy | 水印存活率 | 攻击后恢复的有效载荷比特的比例。 |
| C2PA | 溯源清单 | 关于创建/署名的加密元数据格式。 |
| AASIST | 检测器家族 | 基于图注意力的反欺骗 SOTA。 |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前的基准。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 默认的水印方法。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 针对时序攻击的 MoE 检测器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA 的检测骨干。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 鲁棒性评估。
- [C2PA specification](https://c2pa.org/specifications/specifications/) — 溯源清单格式。