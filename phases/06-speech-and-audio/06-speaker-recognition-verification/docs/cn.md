# 说话人识别与验证

> ASR 问 “他们说了什么？”，说话人识别问 “是谁说的？”。数学上看起来相同 —— 嵌入加余弦 —— 但每一个生产决策都取决于一个单一的 EER 数值。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 6 · 02（频谱图 与 梅尔）, Phase 5 · 22（嵌入模型）  
**Time:** ~45 分钟

## 问题

用户说出一个口令。你想知道：这是不是他们声称的人（*验证*，1:1），或者这是你的注册库中的第一个人（*识别*，1:N）？还是两者都不是 —— 这是一个未知说话人（*开放集*）？

2018 年之前：GMM-UBM + i-vectors。EER 尚可，但对通道变化（电话 vs 笔记本）和情绪敏感。2018–2022：x-vectors（基于 TDNN，使用角度间隔损失训练）。2022 年起：ECAPA-TDNN 与 WavLM-large 嵌入。到 2026 年，该领域被三种模型和一个指标主导。

该指标是 **EER** —— Equal Error Rate。设置决策阈值使得 False Accept Rate = False Reject Rate。交叉点即为 EER。在每篇论文、每个排行榜、每次采购讨论中都使用。

## 概念

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**流水线。** 注册（Enrollment）：录制目标说话人 5–30 秒；计算固定维度嵌入（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证：获取测试话语的嵌入；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020，2026 年仍占主导）。** 强调通道注意（Channel Attention）、传播（Propagation）与聚合（Aggregation）的 Time-Delay Neural Network。1D 卷积块带有 squeeze-excitation，多头注意力池化，随后线性层输出 192 维。使用 Additive Angular Margin 损失（AAM-softmax）在 VoxCeleb 1+2（2,700 名说话人，1.1M 语句）上训练。

**WavLM-SV（2022+）。** 在预训练的 WavLM-large SSL 骨干上用 AAM 损失微调。质量更高但更慢 —— 体积 300+ MB vs 15 MB。

**x-vector（基线）。** TDNN + 统计池化。经典；在 CPU / 边缘设备上仍有用。

**AAM-softmax。** 标准 softmax 在角度空间加入 margin m：对正确类别使用 `cos(θ + m)`。强制类间角度分离。典型 `m=0.2`，缩放 `s=30`。

### 评分

- **余弦相似度（Cosine）**：在注册与测试嵌入之间计算余弦。基于阈值的决策。  
- **PLDA（Probabilistic LDA）。** 将嵌入投影到一个潜在空间，在该空间内同一说话人与不同说话人的似然比有闭式解。加在余弦之上可带来大约 +10–20% 的 EER 降低。2020 年前常用；现在仅在封闭集设置中使用。  
- **得分归一化。** `S-norm` 或 `AS-norm`：将每个得分相对于一组假冒者（cohort）的均值和标准差进行归一化。对跨域评估至关重要。

### 你应该知道的数字（2026）

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### 说话人分割（Diarization）

在多说话人片段中回答 “谁在什么时候说话”。流水线：VAD → 分段 → 对每段做嵌入 → 聚类（凝聚聚类或谱聚类）→ 平滑边界。现代栈：`pyannote.audio` 3.1，将说话人分割 + 嵌入 + 聚类封装为一次调用。2026 年 AMI 数据集上的 SOTA DER 约为 ~15%（相较 2022 年的 23% 有所下降）。

## 实现步骤

### 第 1 步：从 MFCC 统计量构建玩具嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

远非 SOTA —— 仅用于教学。`code/main.py` 将其用作合成说话人数据上的概念验证。

### 第 2 步：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 第 3 步：从相似度对计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 (eer, threshold_at_eer)。请同时报告两者。

### 第 4 步：用 SpeechBrain 生产化

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
# 注册：对 3-5 个干净样本的嵌入取平均
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
# 验证
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA 典型阈值；在你的数据上调优
```

### 第 5 步：用 pyannote 进行说话人分割

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 使用建议

2026 年栈选择：

| Situation | Pick |
|-----------|------|
| Closed-set 1:1 verification, edge | ECAPA-TDNN + 余弦阈值 |
| Open-set verification, cloud | WavLM-SV + AS-norm |
| Diarization (meetings, podcasts) | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing (replay / deepfake detection) | AASIST 或 RawNet2 |
| Tiny embedded (KWS + enrollment) | Titanet-Small（NeMo） |

## 陷阱

- **通道不匹配。** 在 VoxCeleb（网络视频）上训练的模型 ≠ 电话通话音频。务必在目标通道上评估。  
- **短语音时长。** 测试音频低于 3 秒时 EER 会急剧恶化。  
- **有噪音的注册样本。** 一个有噪音的注册样本会污染锚点。使用 ≥3 个干净样本并取平均。  
- **跨条件使用固定阈值。** 始终在来自目标域的留出开发集上调优阈值。  
- **对未归一化嵌入使用余弦。** 先做 L2 归一化；否则幅值将主导相似度。

## 上线要点

保存为 `outputs/skill-speaker-verifier.md`。选择模型、注册协议、阈值调优计划和防欺诈措施。

## 练习

1. 简单：运行 `code/main.py`。构建合成“说话人”（不同的音色轮廓），注册，计算 100 对试验列表的 EER。  
2. 中级：使用 SpeechBrain 的 ECAPA 对 30 个 VoxCeleb1 语句（5 个说话人 × 每人 6 条）进行实验。比较余弦与 PLDA 的 EER。  
3. 困难：用 `pyannote.audio` 构建完整的注册 → 分割 → 验证流水线。在 AMI 开发集上评估 DER。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| EER | The headline metric | 在 False Accept = False Reject 时的阈值。 |
| Verification | 1:1 | “这是 Alice 吗？” |
| Identification | 1:N | “这是谁在说话？” |
| Open-set | Unknown possible | 测试集可以包含未注册的说话人。 |
| Enrollment | Registering | 计算说话人参考嵌入（注册）。 |
| AAM-softmax | The loss | 带有加性角度边距的 softmax；强制簇间分离。 |
| PLDA | Classic scoring | 概率 LDA；在嵌入之上进行似然比评分。 |
| DER | Diarization metric | 说话人分割错误率 — 漏检 + 误报 + 混淆。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 经典的深度嵌入论文。  
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020–2026 年的主导架构。  
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — 用于 SV 和分割的 SSL 骨干。  
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 生产级说话人分割 + 嵌入栈。  
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 当前各模型的 EER 排行。