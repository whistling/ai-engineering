# 语音活动检测与发言回合判定 — Silero、Cobra 与 Flush 技巧

> 每个语音代理的成败取决于两个决策：用户现在在说话吗？他们说完了吗？VAD 回答第一个问题。回合检测（VAD + 静音滞后 + 语义端点模型）回答第二个。任意一个出错，助手要么打断用户，要么永远不停。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 6 · 11 (实时音频), Phase 6 · 12 (语音助手)  
**Time:** ~45 分钟

## 问题概述

语音代理在每个 20 ms 的音频片段上要做三个不同的决策：

1. **这是语音帧吗？** — VAD。逐帧的二值判断。
2. **用户是否开始了新的发话？** — 起始检测。
3. **用户说完了吗？** — 端点检测（回合结束）。

简单的能量阈值方法在任何噪声下都会失败——交通声、键盘声、人群嘈杂。2026 年的答案：Silero VAD（开源、深度学习）+ 一个回合检测模型（语义端点）+ 一个基于 VAD 校准的静音滞后。

## 概念

![VAD 级联：能量 → Silero → 回合检测器 → flush 技巧](../assets/vad-turn-taking.svg)

### 三层 VAD 级联

**第 1 层：能量门。** 最便宜。对 RMS 设定阈值，比如 -40 dBFS。可以过滤明显的静音，但任何超过阈值的噪声都会触发。

**第 2 层：Silero VAD**（2020–2026，MIT）。1M 参数。基于 6000+ 语言训练。在单核 CPU 上对 30 ms 片段运行约 1 ms。5% 假阳性率下的真正率为 87.7%。开源默认选择。

**第 3 层：语义回合检测器。** LiveKit 的回合检测模型（2024–2026）或你自己的小型分类器。区分“句中短暂停顿”与“说话结束”。使用语言上下文（语调 + 最近词序列），而不仅仅依赖静音。

### 关键参数及其默认值

- **阈值（Threshold）。** Silero 输出概率；默认为 > 0.5 判为语音（也可设 > 0.3 提高灵敏度）。降低阈值 = 更少的首词被截断，但会增加误报。
- **最小语音时长。** 拒绝短于 250 ms 的语音 —— 通常是咳嗽或椅子噪声。
- **静音滞后（端点检测）。** 当 VAD 返回 0 后，等待 500–800 ms 再宣布回合结束。太短 → 打断用户。太长 → 感觉迟缓。
- **预录缓冲（Pre-roll buffer）。** 在 VAD 触发前保留 300–500 ms 的音频，防止“hey”被截掉。

### Flush 技巧（Kyutai 2025）

流式 STT 模型存在前窥延迟（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 s）。通常你需要在语音结束后等这么久才会得到转录。Flush 技巧：当 VAD 检测到语音结束时，**向 STT 发送 flush 信号以强制立即输出**。STT 以 ~4× 实时速度处理，所以 500 ms 的缓存约在 125 ms 内完成。

端到端：125 ms VAD + flush STT = 会话级延迟。

### 2026 年 VAD 对比

| VAD | TPR @ 5% FPR | 延迟 | 许可证 |
|-----|--------------|------|--------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | 商业 |
| pyannote segmentation | 95% | ~10 ms | 类 MIT |

Silero 是合适的默认选择。Cobra 用于合规/高精度场景。纯能量判定在 2026 年的生产环境中毫无立足之地。

## 构建实现

### 步骤 1：能量门

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤 2：在 Python 中使用 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步骤 3：回合结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步骤 4：Flush 技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能生效。Whisper 的流式实现不支持 —— 它是基于块的，总是等待完整块。

## 使用建议

| 场景 | VAD 选择 |
|------|---------|
| 开放、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究 / 说话者分离 | pyannote segmentation |
| 零依赖后备 | WebRTC VAD（遗留） |
| 需要高质量回合结束 | Silero + LiveKit 回合检测 层叠 |

经验法则：除非确实别无选择，切勿只用能量判定的 VAD 上线。

## 常见陷阱

- **固定阈值。** 在安静环境有效，嘈杂环境失效。要么在设备上校准，要么改用 Silero。
- **静音滞后太短。** 代理在句中打断用户。500–800 ms 是对话语音的最佳区间。
- **静音滞后太长。** 感觉迟钝。与目标用户做 A/B 测试。
- **没有预录缓冲。** 丢失用户前 200–300 ms 的语音。始终保留循环预录。
- **忽视语义端点。** “嗯，让我想想……”会有长暂停。用户讨厌在思考中被打断。使用 LiveKit 的回合检测或类似方案。

## 投产

保存为 `outputs/skill-vad-tuner.md`。为你的工作负载选择 VAD 模型、阈值、滞后、预录和回合检测策略。

## 练习题

1. **简单。** 运行 `code/main.py`。它模拟了一段 说话 + 静音 + 说话 + 咳嗽 的序列，并测试三层 VAD。
2. **中等。** 安装 `silero-vad`，处理一段 5 分钟录音，调节阈值以减少首词被截断和误触发。报告精确率/召回率。
3. **困难。** 构建一个迷你回合检测器：Silero VAD + 在最近 10 个词嵌入上运行的 3 层 MLP（使用 sentence-transformers）。在手工标注的回合结束数据集上训练。使 F1 比仅靠 Silero 提高 10%。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|---------|---------|
| VAD | Voice detector | 逐帧二值：这是语音吗？ |
| Turn detection | End-pointing | VAD + 静音滞后 + 语义端点。 |
| Silence hangover | Wait-after-speech | 发言后在宣布回合结束前等待的时间；500–800 ms。 |
| Pre-roll | Pre-speech buffer | 在 VAD 触发前保留 300–500 ms 的音频。 |
| Flush trick | Kyutai hack | VAD → flush-STT → 125 ms 而非 500 ms 的延迟。 |
| Semantic endpoint | "Did they mean to stop?" | 一个查看词序列而非仅依赖静音的 ML 分类器。 |
| TPR @ FPR 5% | ROC 点 | 标准 VAD 基准；Silero 为 87.7%，WebRTC 为 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考的开源 VAD。  
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业精度领先者。  
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 子 200 ms 的工程技巧。  
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 生产级语义端点实现。  
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 传统基线。  
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 达到说话人分离级别的分割工具。