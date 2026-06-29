# 实时音频处理

> 批处理流水线处理一个文件。实时流水线则要在下一批 20 毫秒到来前处理好当前的 20 毫秒。每个对话式 AI、广播演播室和电话机器人都靠这个延迟预算存亡。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 6 · 02（频谱）, Phase 6 · 04（ASR）, Phase 6 · 07（TTS）
**Time:** ~75 分钟

## 问题

你想要一个有生命感的语音助理。人的对话轮转延迟大约 ~230 ms（静音到回应）。任何超过 500 ms 的都感觉像机器人；超过 1500 ms 则感觉坏掉了。到 2026 年，实现完整的 **hear → understand → respond → speak** 回路的预算是：

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi（Kyutai，2024）实现了 200 ms 的全双工（full-duplex）。GPT-4o-realtime（2024）大约为 ~320 ms。2022 年的级联流水线发布时为 2500 ms。这 10× 的改进来自三种技术：（1）各处采用流式（streaming），（2）采用带部分结果的异步流水线，（3）可中断的生成。

## 概念

![流式音频管道，带环形缓冲、VAD 门控、打断](../assets/real-time.svg)

**Frame / chunk / window（帧 / 块 / 窗口）。** 实时音频以固定大小的区块流动。常见选择：20 ms（在 16 kHz 下为 320 个采样点）。所有下游组件都必须跟上这个节奏。

**Ring buffer（环形缓冲）。** 固定大小的循环缓冲。生产线程写入新帧，消费线程读取。防止热路径中的分配。大小 ≈ 最大延迟 × 采样率；16 kHz 下 2 秒的环形缓冲约为 32,000 个样本。

**VAD（Voice Activity Detection，语音活动检测）。** 在无人说话时门控下游工作。Silero VAD 4.0（2024）对每个 30 ms 帧在 CPU 上运行 <1 ms。`webrtcvad` 是较早的替代方案。

**Streaming ASR（流式 ASR）。** 随着音频到达就输出部分转录的模型。Parakeet-CTC-0.6B 在流式模式（NeMo，2024）下在 320 ms 延迟时能做到 2–5% 的 WER。Whisper-Streaming（Macháček 等，2023）将 Whisper 切块以实现近实时流式，延迟约为 ~2 s。

**Interruption（打断 / 抢话）。** 当用户在助理说话时发声，你必须 (a) 检测到抢话，(b) 停止 TTS，(c) 丢弃剩余的 LLM 输出。全部必须在 100 ms 内完成，否则用户会感觉助理“听不见”。

**WebRTC Opus 传输。** 20 ms 帧，48 kHz，自适应码率 8–128 kbps。是浏览器和移动端的标准。LiveKit、Daily.co、Pion 是 2026 年用于构建语音应用的主流栈。

**Jitter buffer（抖动缓冲）。** 网络包会乱序/延迟到达。抖动缓冲用于重排序和平滑；太小会出现可听间隙，太大则增加延迟。典型值 60–80 ms。

### 常见坑

- **线程争用。** Python 的 GIL + 大模型会饿死音频线程。使用 C 回调的音频库（sounddevice、PortAudio），并且让 Python 远离热路径。
- **采样率转换延迟。** 管道内重采样会增加 5–20 ms。要么在前端统一重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热。** 即使是快速的 TTS（如 Kokoro）在第一次请求时也有 100–200 ms 的预热。缓存模型并用一次虚拟请求预热。
- **回声消除。** 没有 AEC，TTS 输出会被麦克风再次捕获并触发 ASR 对机器人自己的声音识别。WebRTC AEC3 是开源默认方案。

```figure
nyquist-aliasing
```

## 构建

### 第 1 步：环形缓冲

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Capacity 决定最大缓冲延迟。16 kHz 下 32,000 个样本相当于 2 秒。

### 第 2 步：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

在生产环境中用 Silero VAD 替换：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 第 3 步：流式 ASR

```python
# 使用 NeMo 实现 Parakeet-CTC-0.6B 的流式处理
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 毫秒，look_ahead_ms=80 毫秒
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 第 4 步：打断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # 抢话（barge-in）
        # 然后将帧送入流式 ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

这依赖于异步 I/O 和可取消的 TTS 流式输出。对音轨调用 WebRTC peerconnection.stop() 是规范做法。

## 使用

2026 年栈：

| Layer | Pick |
|-------|------|
| Transport | LiveKit（WebRTC）或 Pion（Go） |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM first-token | Groq、Cerebras、vLLM-streaming |
| Streaming TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API 或 Moshi |

## 陷阱

- **缓冲 500 ms 以求保险。** 缓冲即是你的延迟下限。缩小它。
- **线程未绑定。** 音频回调运行在优先级低于 UI 的线程上会在负载下出现故障。
- **TTS 块太小。** 小于 200 ms 的块会让声码器产生可闻伪影。320 ms 的块是折中点。
- **没有抖动缓冲。** 真实网络会抖动；没有平滑会出现爆音/断裂。
- **一次性错误处理。** 音频流水线必须防崩溃。一次异常会终止会话。

## 发布

保存为 `outputs/skill-realtime-designer.md`。为每个阶段设计带具体延迟预算的实时音频管道。

## 练习

1. **简单。** 运行 `code/main.py`。模拟环形缓冲 + 能量型 VAD；为一个假的 10 秒流打印各阶段延迟。
2. **中等。** 使用 `sounddevice`，构建一个以 20 ms 帧处理麦克风并在每帧打印 VAD 状态的直通循环。
3. **困难。** 使用 `aiortc` 构建全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用 1 kHz 脉冲测量玻璃到玻璃延迟。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | 固定大小、无锁（或 SPSC 锁）的音频帧 FIFO。 |
| VAD | Silence gate | 标注语音与非语音的模型或启发式方法。 |
| Streaming ASR | Real-time STT | 随音频到达输出部分文本；具有有界的前视（lookahead）。 |
| Jitter buffer | Network smoother | 重新排序乱序包的队列；典型值 60–80 ms。 |
| AEC | Echo cancellation | 减去扬声器到麦克风的反馈路径。 |
| Barge-in | User interrupt | 系统在 TTS 中检测到用户语音；必须取消播放。 |
| Full duplex | Simultaneous both ways | 双向同时传输；Moshi 提供全双工能力。 |

## 深入阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 将 Whisper 切块以实现近实时流式的工作。  
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 实现 200 ms 的全双工系统。  
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 面向生产的音频代理编排。  
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 亚毫秒级 VAD，Apache 2.0。  
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源环境下的回声消除。