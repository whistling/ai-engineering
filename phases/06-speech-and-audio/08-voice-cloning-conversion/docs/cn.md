# 语音克隆与语音转换

> 语音克隆会用别人的声音朗读你的文本。语音转换会把你的声音重写成别人的声音，同时保留你说的内容。两者都基于相同的分解思路：把说话人身份与内容分离。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 6 · 06 (说话人识别), Phase 6 · 07 (文本转语音)  
**Time:** ~75 分钟

## 问题

在 2026 年，5 秒的音频片段足以在消费级 GPU 上生成任何人的高质量语音克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 等都提供零样本或少样本克隆。这项技术既是福音（无障碍 TTS、配音、辅助语音），也是武器（诈骗电话、政治深度伪造、知识产权盗用）。

两个密切相关的任务：

- **语音克隆（TTS 端）：** 文本 + 5 秒参考语音 → 用该说话人声音合成音频。
- **语音转换（语音端）：** 源音频（A 说 X） + 目标说话人 B 的参考声音 → B 说 X 的音频。

两者都会将波形分解为（内容、说话人、韵律），并将一个来源的内容与另一个说话人的声音重组。

你在 2026 年需要遵守的关键约束：**在欧盟（AI Act，于 2026 年 8 月可强制执行）和加利福尼亚（AB 2905，于 2025 年生效）中，水印和同意门是法律要求**。你的流水线必须输出不可听见的水印并拒绝未经同意的克隆。

## 概念

![语音克隆 vs 语音转换：分解，说话人替换，重组](../assets/voice-cloning.svg)

**零样本克隆。** 将 5 秒片段传入一个在成千上万说话人上训练过的模型。说话人编码器将片段映射为说话人嵌入；TTS 解码器在该嵌入和文本的条件下合成音频。

代表性实现：F5-TTS (2024)、YourTTS (2022)、XTTS v2 (2024)、OpenVoice v2 (2024)。

**少样本微调。** 录制目标说话人 5–30 分钟音频。对基础模型进行约一小时的 LoRA 微调。质量会从“还行”跃升到“无法区分”。Coqui 和 ElevenLabs 都支持该模式；社区也在 F5-TTS 上这么做。

**语音转换（VC）。** 有两类方法：

- **识别-合成（Recognition-synthesis）。** 运行类似 ASR 的模型以提取内容表示（例如软音素后验、PPG），然后用目标说话人嵌入重新合成。对语言和口音较为鲁棒。代表工作：KNN-VC (2023)、Diff-HierVC (2023)。
- **解缠（Disentanglement）。** 训练自编码器，在瓶颈处将内容、说话人和韵律在潜空间中分离。推理时替换说话人嵌入。质量较低但更快。代表工作：AutoVC (2019)、VITS-VC 变体。

**基于神经编解码器的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox 等将音频视为来自 SoundStream / EnCodec 的离散 token，在这些 codec token 上训练大规模自回归或 flow-matching 模型。短提示下的质量可比肩 ElevenLabs。

### 不是附加的伦理部分

**水印（Watermarking）。** PerTh (Perth) 和 SilentCipher (2024) 将约 16–32 位的 ID 不可察觉地嵌入音频。能在重编码、流式传输和常见编辑后存活。已达生产级别的开源实现。

**同意门（Consent gates）。** 每个克隆输出必须与可验证的同意记录配对。例如：“我，Rohit，于 2026-04-22 授权此语音用于 X 目的。”将同意记录存储在防篡改日志中。

**检测。** AASIST、RawNet2、Wav2Vec2-AASIST 提供检测器。ASVspoof 2025 挑战在对抗 ElevenLabs、VALL-E 2 和 Bark 输出时报告的 EER 在 0.8–2.3% 之间。

### 数字（2026）

| 模型 | 零样本？ | SECS（目标相似度） | WER（可懂度） | 参数量 |
|------|---------|---------------------|---------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0.70 对大多数听众来说通常无法区分目标说话人。

## 构建它

### 步骤 1：用识别-合成分解（main.py 中的仅代码演示）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上很简单；实现的复杂度集中在 `tts_model` 和说话人编码器上。

### 步骤 2：用 F5-TTS 进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录必须与音频完全匹配；不匹配会破坏对齐。

### 步骤 3：用 KNN-VC 做语音转换

```python
import torch
from knnvc import KNNVC  # 2023 年模型, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 使用 WavLM 提取源音频与目标池的逐帧嵌入，然后将每个源帧替换为池中最近邻的帧。非参数方法，对一分钟左右的目标语音即可工作。

### 步骤 4：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # 返回载荷字节
```

约 32 位的载荷，在 MP3 重编码和轻微噪声后仍可检测。

### 步骤 5：同意门

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 使用它

2026 年的技术栈选择：

| 情况 | 选择 |
|------|------|
| 5 秒零样本克隆，开源 | F5-TTS 或 OpenVoice v2 |
| 商业级生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 语音转换（重写） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 陷阱

- **参考转录不对齐。** F5-TTS 等要求参考文本与参考音频完全匹配，标点也要一致。
- **参考有混响。** 回声会毁掉克隆。请使用干声、近场麦克风录制。
- **情绪不匹配。** 训练参考为“高兴”的话，会把所有合成都做成高兴的感觉。将参考情绪与目标用途匹配。
- **语言泄露。** 克隆一个英语说话人然后让模型说法语，往往会带有原说话人的口音；使用跨语种模型（XTTS、VALL-E X）。
- **没有水印。** 从 2026 年 8 月起在欧盟法律上无法上架/发货。

## 上线

保存为 `outputs/skill-voice-cloner.md`。设计一个包含同意门 + 水印 + 质量目标的克隆或转换流水线。

## 练习

1. 简单。运行 `code/main.py`。通过计算两个“说话人”在替换前后的余弦相似度来演示说话人嵌入的交换。
2. 中等。使用 OpenVoice v2 克隆你自己的声音。测量参考与克隆之间的 SECS。用 Whisper 测量 CER。
3. 困难。对 20 个克隆应用 SilentCipher 水印，经过 128 kbps MP3 编码+解码，检测载荷。报告比特准确率。

## 关键词

| 术语 | 大家说法 | 实际含义 |
|------|----------|---------|
| Zero-shot clone | 5 秒就够 | 预训练模型 + 说话人嵌入；无需训练。 |
| PPG | Phonetic posteriorgram | 每帧的 ASR 后验，作为与语言无关的内容表示。 |
| KNN-VC | 最近邻转换 | 将每个源帧替换为目标池中最近的帧。 |
| Neural codec TTS | VALL-E 风格 | 在 EnCodec/SoundStream token 上训练自回归模型。 |
| Watermark | 不可听的签名 | 将比特嵌入音频，能在重编码后存活。 |
| SECS | 克隆保真度 | 目标与克隆说话人嵌入之间的余弦相似度。 |
| AASIST | 深度伪造检测器 | 反欺骗模型；检测合成语音。 |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源的 SOTA 零样本克隆。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 和 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 基于神经 codec 的 TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于解缠的语音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 基于检索的语音转换。
- [SilentCipher (2024) — 音频水印](https://github.com/sony/silentcipher) — 生产就绪的 32 位音频水印实现。
- [ASVspoof 2025 结果](https://www.asvspoof.org/) — 探讨检测器与合成器之间的军备竞赛，已更新至 2026 年。