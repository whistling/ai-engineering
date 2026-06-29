# Speech Recognition (ASR) — CTC, RNN-T, Attention

> 语音识别是在每个时间步对音频进行分类，然后由一个了解语言和静音的序列模型将这些分类拼接成字符串。CTC、RNN-T 和注意力是三种实现方式。选择其中之一并理解其原因。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (声谱图与梅尔), Phase 5 · 08 (用于文本的 CNNs 与 RNNs), Phase 5 · 10 (注意力)
**Time:** ~45 分钟

## 问题

你有一段 10 秒、16 kHz 的音频片段。你希望得到一个字符串："turn on the kitchen lights"。挑战在于结构性：音频帧与字符并非一一对应。单词 "okay" 可能占用 200 ms 或 1200 ms。静音穿插在话语中。某些音素比其他音素更长。输出 token 的数量事先未知。

三种表述可以解决这个问题：

1. **CTC (Connectionist Temporal Classification)。** 对每帧输出包含一个特殊 *blank* 的 `V+1` 类别分布。对于长度为 `U < T` 的目标字符串 `y`，任何在解码时能坍缩为 `y` 的帧对齐均计入。CTC 损失对所有这样的对齐求和。推理：对每帧取 argmax，坍缩重复并删除 blanks。

   优点：非自回归、快速、可流式、零前瞻。缺点：*条件独立假设* —— 每帧预测独立，因此没有内部语言模型。可通过束搜索或浅融合将外部 LM 引入以修正。

2. **RNN-T (Recurrent Neural Network Transducer)。** 增加一个 *预测器* 网络来嵌入 token 历史，并用一个 *合并器* 将预测器状态与编码器帧结合，得到对 `V+1`（`+1` 表示 null / 不发射）的联合分布。显式建模了 CTC 忽略的条件依赖。因为每步仅依赖过去的帧和过去的 token，所以可以流式处理。

   优点：可流式 + 内部语言模型。缺点：训练更复杂且占用更多内存（3D 损失格）；RNN-T 损失的内核本身就是一个完整的库级别功能。

3. **注意力编码器-解码器。** 编码器对 log-mel 帧做 6–32 层的 Transformer 编码。解码器跨注意力地对编码器输出进行注意以自回归地生成 token。没有对齐约束 —— 注意力可以查看音频的任何位置。若不限制注意力，通常不可流式（除非做 chunk 限制，如 Whisper-Streaming，2024）。

   优点：离线 ASR 的最高质量，易于用标准序列到序列工具训练。缺点：自回归导致延迟与输出长度成比例；若要支持流式需要额外工程。

到 2026 年，LibriSpeech test-clean 上的 SOTA WER 为 1.4%（Parakeet-TDT-1.1B, NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。差异很小；但部署差异很大。

## 概念

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC 直觉。** 让编码器输出 `T` 个帧级分布，每个分布覆盖 `V+1` 个 token（V 个字符 + `blank`）。对于长度为 `U < T` 的目标字符串 `y`，任何能在解码时坍缩为 `y` 的帧对齐都计入。CTC 损失对所有这些对齐求和。推理：对每帧取 argmax，坍缩重复，去掉 blanks。

优点：非自回归、可流式、零前瞻。缺点：*条件独立假设* —— 每帧预测相互独立，因此没有内部语言模型。通过束搜索或浅融合将外部语言模型加入可以弥补这一点。

**RNN-T 直觉。** 增加一个 *预测器* 网络来嵌入 token 历史，和一个 *合并器* 将预测器状态与编码器帧结合，输出对 `V+1`（`+1` 表示不发射）的联合分布。显式地建模了 CTC 忽略的条件依赖关系。可流式，因为每步只依赖过去的帧和过去的 token。

优点：可流式 + 内部语言模型。缺点：训练更复杂且占用内存（3D 损失格）；RNN-T 损失实现是一类完整的库功能。

**注意力编码器-解码器。** 编码器（6–32 层 Transformer）处理 log-mel 帧。解码器（6–32 层 Transformer）跨注意力编码器输出，自回归地生成 token。没有对齐约束——注意力可以在音频中任意跳转。除非限制注意力（如分片的 Whisper-Streaming），否则不可流式。

优点：离线 ASR 质量最高，使用标准 seq2seq 工具训练简单。缺点：自回归的延迟与输出长度成正比；不做工程无法流式。

### WER：那个关键数字

**词错误率（Word Error Rate）** = `(S + D + I) / N`，其中 S=替换，D=删除，I=插入，N=参考词数。相当于在词级别上的 Levenshtein 编辑距离。数值越低越好。WER 超过 20% 通常不可用；低于 5% 对朗读语音来说接近人类水平。2026 年在标准基准上的部分结果：

| Model | LibriSpeech test-clean | LibriSpeech test-other | Size |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

以上模型均为编码器-解码器或 RNN-T。纯 CTC 系统（如 wav2vec 2.0）在 test-clean 上大约在 1.8–2.1% 左右。

## 构建

### 第 1 步：贪心 CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: 每帧的概率向量列表
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：坍缩连续重复，丢弃 blanks。示例：`a a _ _ a b b _ c` → `a a b c`。

### 第 2 步：CTC 束搜索

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境会使用带语言模型融合的前缀树束搜索；这里只是概念性骨架。

### 第 3 步：计算 WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 第 4 步：对 Whisper 的推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

一句话调用 2026 年最强的一般用途 ASR。24 GB GPU 上运行约为实时的 ~20×。

### 第 5 步：使用 Parakeet 或 wav2vec 2.0 做流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要对编码器注意力做分片并保留状态；使用支持该功能的库（Parakeet 推荐 NeMo，或在 `transformers` 的 pipeline 中使用 `chunk_length_s`）。

## 使用建议

2026 年的堆栈选择：

| Situation | Pick |
|-----------|------|
| English, offline, max quality | Whisper-large-v3-turbo |
| Multilingual, robust | SeamlessM4T v2 |
| Streaming, low latency | Parakeet-TDT-1.1B or Riva |
| Edge, mobile, <500 ms latency | Whisper-Tiny 量化版 或 Moonshine (2024) |
| Long-form | 使用基于 VAD 的分片的 Whisper (WhisperX) |
| Domain-specific (medical, legal) | 微调 wav2vec 2.0 并结合领域语言模型融合 |

## 2026 年仍会导致问题的坑

- **没有 VAD。** 在静音上运行 Whisper 会产生幻觉（“Thanks for watching!”）。总是先用 VAD 做门控。
- **字符级 vs 词级 vs 子词级 WER。** 报告词级 WER 时应先做规范化（小写、去标点）。
- **语言识别漂移。** Whisper 的自动 LID 在嘈杂片段上可能错误地识别为日语或威尔士语；已知语言时请强制使用 `language="en"`。
- **长片段未分片处理。** Whisper 有 30 秒窗口。对于更长的音频，使用 `chunk_length_s=30, stride=5`。

## 上线

保存为 `outputs/skill-asr-picker.md`。为给定的部署目标选择模型、解码策略、分片方法和语言模型融合方案。

## 练习

1. **简单。** 运行 `code/main.py`。它对一个手工合成的 CTC 输出做贪心解码，并与参考计算 WER。
2. **中等。** 正确实现第 2 步中的前缀树束搜索（考虑 blank 合并规则）。在 10 个合成示例上与贪心结果比较。
3. **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 的前 100 条语句上使用 `whisper-large-v3-turbo` 计算 WER。与发表的结果比较。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| CTC | The blank-token loss | 对所有帧到 token 的对齐边缘化求和；非自回归。 |
| RNN-T | The streaming loss | CTC + 下一个 token 的预测器；能处理词序问题。 |
| Attention enc-dec | Whisper-style | 编码器 + 跨注意力解码器；离线质量最好。 |
| WER | The number you report | 词级别的 `(S+D+I)/N`。 |
| Blank | The emptiness | CTC 中表示“该帧没有发射”的特殊 token。 |
| LM fusion | External language model | 在束搜索期间加入加权的语言模型对数概率。 |
| VAD | The silence gate | 语音活动检测器；裁剪非语音段。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年的经典论文；v3-turbo 扩展于 2024 年。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 年 Open ASR Leaderboard 的领先者卡片。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 覆盖 25+ 模型的实时基准。