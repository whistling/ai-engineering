# 音频评估 — WER、MOS、UTMOS、MMAU、MMAU-Pro、FAD 与开放排行榜

> 你无法交付无法衡量的东西。本课命名了 2026 年各类音频任务的度量：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、WER-on-ASR-round-trip）、音频-语言（MMAU、LongAudioBench）、音乐（FAD、CLAP）和说话人（EER）。以及用于比较的排行榜。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (模型评估)  
**Time:** ~60 分钟

## 问题

每个音频任务都有多个度量，每个度量衡量一条不同的轴线。使用错误的度量会导致你发布一个在仪表盘上看起来很棒但在生产中糟透的模型。到 2026 年的规范列表：

| 任务 | 主要 | 次要 |
|------|------|------|
| ASR（自动语音识别） | WER | CER · RTFx · 首令牌延迟 |
| TTS（文本到语音） | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| 语音克隆 | SECS（ECAPA 余弦） | MOS · CER |
| 说话人验证 | EER | minDCF · 在操作点的 FAR / FRR |
| 说话人分割（Diarization） | DER | JER · 说话人混淆 |
| 音频分类 | top-1 · mAP | macro F1 · 按类召回率 |
| 音乐生成 | FAD | CLAP · 听评面板 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式 S2S | 延迟 P50/P95 | WER · MOS |

## 概念

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR 度量

**WER（词错误率）。** `(S + D + I) / N`。在评分前做小写、去标点、数字规范化。可使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。小于 5% = 读稿类语音的人类同等水平。

**CER（字符错误率）。** 同样公式，按字符级别计算。用于语调语言（普通话、粤语）或词边界难以判定的场景。

**RTFx（逆实时因子）。** 每秒处理的音频秒数 / 墙钟秒数。数值越高越好。Parakeet-TDT 达到 3380×。Whisper-large-v3 约为 ~30×。

**首令牌延迟（first-token latency）。** 从音频输入到第一个转录令牌的墙钟时间。对流式场景至关重要。Deepgram Nova-3: ~150 ms。

### TTS 度量

**MOS（平均意见分）。** 1-5 的人工评分。黄金标准但耗时。每个样本收集 20+ 听众，模型评估用 100+ 样本。

**UTMOS（2022–2026）。** 学习型 MOS 预测器。在标准基准上与人工 MOS 的相关约为 0.9。F5-TTS: UTMOS 3.95；真实语音（ground truth）：4.08。

**SECS（说话人编码器余弦相似度）。** 用于语音克隆。将参考与克隆输出的 ECAPA 嵌入做余弦相似度。> 0.75 = 可识别的克隆。

**WER-on-ASR-round-trip。** 对 TTS 输出运行 Whisper，计算与输入文本的 WER。能捕捉可懂度的回归。2026 年 SOTA：CER < 2%。

**TTFA（time-to-first-audio）。** 墙钟延迟。Kokoro-82M: ~100 ms；F5-TTS: ~1 s。

### 语音克隆专用

将 **SECS + MOS + CER** 作为三元组来评估。SECS 高但 MOS 低说明音色对但不自然；反之则说明自然但说话人错误。

### 说话人验证

**EER（等误率）。** 假接受率（FAR）等于假拒绝率（FRR）时的阈值。ECAPA 在 VoxCeleb1-O 上：0.87%。

**minDCF（最小检测成本）。** 在选定操作点（常以 FAR=0.01）下的加权成本。比 EER 更贴近生产环境。

### 说话人分割（Diarization）

**DER（分割错误率）。** `(FA + Miss + Confusion) / total_speaker_time`。漏检语音 + 误报语音 + 说话人混淆，各自按占比计算。AMI 会议：DER 约 10–20% 比较现实。pyannote 3.1 + Precision-2 商用：在良好录音上 <10% DER。

**JER（Jaccard 错误率）。** DER 的替代，能更好抵抗短段偏差。

### 音频分类

多标签：**mAP（平均精度均值）** 在所有类上求均值。AudioSet：BEATs-iter3 达到 0.548 mAP。

多类互斥：**top-1、top-5 准确率**。Speech Commands v2：Audio-MAE 达到 99.0% top-1。

不平衡任务：**macro F1** + **按类召回率**。务必报告按类结果——整体准确率会掩盖哪些类失败。

### 音乐生成

**FAD（Fréchet 音频距离）。** 在 VGGish 嵌入空间上计算真实与生成音频分布间的距离。MusicGen-small 在 MusicCaps 上为 4.5，MusicLM 为 4.0。值越低越好。

**CLAP 分数。** 使用 CLAP 嵌入衡量文本与音频的对齐度。> 0.3 = 对齐合理。

**听评面板 MOS。** 对消费者级音乐仍然是最终裁决。Suno v5 在 TTS Arena 的 ELO 为 1293（基于配对人工偏好）。

### 音频-语言基准

**MMAU（Massive Multi-Audio Understanding）。** 10k 条音频问答对。

**MMAU-Pro。** 1800 条困难题，分为四类：speech / sound / music / multi-audio。4 选一的随机猜测为 25%。Gemini 2.5 Pro 总体约 ~60%；multi-audio 在所有模型上约 ~22%。

**LongAudioBench。** 包含数分钟级别的剪辑与语义查询。Audio Flamingo Next 击败了 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 标注生成基准。使用 SPICE、CIDEr、FENSE 等指标。

### 流式语音到语音（Streaming S2S）

**延迟 P50 / P95 / P99。** 从用户说话结束到首个可听响应的墙钟时间。Moshi: 200 ms；GPT-4o Realtime: 300 ms。

**输出的 WER / MOS。**

**打断响应性（barge-in responsiveness）。** 从用户中断到助手静音的时间。目标 < 150 ms。

### 2026 年排行榜

| 排行榜 | 赛道 | URL |
|--------|------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，基于配对投票的 ELO 排名 | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU 音乐子集 | 音乐 LALM | （在 MMAU 内） |
| HEAR benchmark | 自监督音频 | `hearbenchmark.com` |

## 实现步骤

### 步骤 1：带规范化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# 约 0.17
```

### 步骤 2：TTS 回环 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 步骤 3：语音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 步骤 4：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 步骤 5：说话人验证的 EER（与第 6 课相同代码）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使用指南

为每次发布配备一个固定的评估 harness，在每次模型更新时运行。三条基本规则：

1. 规范化后再评分。小写、去标点、数字展开。报告所用的规范化规则。
2. 报告分布，而不是平均值。延迟用 P50/P95/P99。分类报告按类召回率。MMAU 按类别报告。
3. 运行一个规范的公开基准。即便你的生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告能让评审进行同台比较。

## 陷阱

- **UTMOS 外推问题。** 在 VCTK 风格的干净语音上训练；对嘈杂 / 克隆 / 情绪化音频打分不足。
- **MOS 面板偏差。** 20 名 Amazon Mechanical Turk 工人 ≠ 20 名目标用户。若风险较高，应为领域面板付费。
- **FAD 依赖参考集。** 在不同模型间比较时要使用相同的参考分布。
- **聚合 WER 的误导。** 整体 5% WER 可能掩盖口音语音上的 30% WER。按人口学切片报告。
- **公开基准饱和。** 前沿模型在标准基准上多数已接近天花板。构建一个反映你流量的内部 held-out 集。

## 发布（Ship It）

保存为 `outputs/skill-audio-evaluator.md`。为任意音频模型发布选定指标、基准和报告格式。

## 练习

1. 简单：运行 `code/main.py`。在玩具输入上计算 WER / CER / EER / SECS / FAD-ish / MMAU-ish。
2. 中等：构建一个 TTS 回环 WER harness。将你的 Kokoro 或 F5-TTS 输出通过 Whisper。对 50 个提示计算 WER。标记 WER > 10% 的提示。
3. 困难：对你在第 10 课中选择的 LALM，在 MMAU-Pro 的 speech + multi-audio 子集（各 50 条）上打分。报告按类别准确率并与公开数值比较。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|---------|
| WER | ASR 分数 | 规范化后按词级计算的 `(S+D+I)/N`。 |
| CER | 字符级 WER | 用于声调语言或字符级系统。 |
| MOS | 人工意见分 | 1-5 评分；20+ 听众 × 100 个样本。 |
| UTMOS | ML MOS 预测器 | 学习模型；与人工 MOS 的相关约为 0.9。 |
| SECS | 语音克隆相似度 | 参考与克隆之间的 ECAPA 余弦相似度。 |
| EER | 说话人验证分数 | FAR = FRR 的阈值。 |
| DER | 分割错误率 | (FA + Miss + Confusion) / 总时间。 |
| FAD | 音乐生成质量 | 在 VGGish 嵌入上的 Fréchet 距离。 |
| RTFx | 吞吐量 | 每一墙钟秒处理的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带规范化工具的 WER/CER 库。  
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 学习型 MOS 预测器。  
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 音乐生成的标准度量。  
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 年实时排名。  
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 基于人工投票的 TTS 排行榜。  
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM 推理排行榜。  
- [HEAR benchmark](https://hearbenchmark.com/) — 音频自监督学习（SSL）基准。