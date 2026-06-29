# 构建语音助手流水线 — 第6阶段压轴项目

> 将课程01-11 的所有内容串起来。构建一个能听、能推理并能讲话的语音助手。在2026年，这已经是一个工程问题而非研究问题——但集成细节决定能否真正上线。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11；Phase 11 · 09 (函数调用)；Phase 14 · 01 (Agent 循环)  
**Time:** ~120 分钟

## 问题描述

构建一个端到端助手：

1. 捕获麦克风输入（16 kHz 单声道）。
2. 检测用户语音的开始/结束。
3. 流式转录。
4. 将转录文本传给能调用工具（计时器、天气、日历）的 LLM。
5. 将 LLM 的文本流送入 TTS。
6. 将音频播放给用户。
7. 如果用户在回复中途打断，则停止。

延迟目标：在用户结束话语后 800 ms 内在笔记本 CPU 上产生第一个 TTS 音频字节。质量目标：不丢词、不在静默时产生幻觉字幕、不泄露语音克隆、不被提示词注入攻破。

## 概念

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### 七个组件

1. **音频采集。** 麦克风 → 16 kHz 单声道 → 20 ms 分片。通常在 Python 中用 `sounddevice`，生产环境可用原生 AudioUnit/ALSA/WASAPI。
2. **VAD（第11课）。** Silero VAD，阈值 0.5，最短语音 250 ms，静默保留 500 ms。发出“开始”和“结束”信号。
3. **流式 STT（第4-5课）。** Whisper-streaming、Parakeet-TDT，或 Deepgram Nova-3（API）。提供部分转录与最终转录。
4. **可调用工具的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。使用 JSON schema 定义工具。流式返回 token。
5. **流式 TTS（第7课）。** Kokoro-82M（最快的开源）或 Cartesia Sonic（商用）。在 LLM 输出约 20 个 token 后启动 TTS。
6. **回放。** 播放到扬声器；在低带宽网络下做 opus 编码。
7. **打断处理器。** 如果 VAD 在 TTS 播放期间触发，停止回放、取消 LLM，重启 STT。

### 你会遇到的三种失败模式

1. **首词被截断。** VAD 启动慢了半拍，用户的 “hey” 被丢失。将启动阈值设为 0.3 而不是 0.5。
2. **中途打断导致的混淆。** 用户打断后 LLM 仍在生成，助手压着用户说话。把 VAD → 取消-LLM 的链路接好。
3. **静默幻觉。** Whisper 在静默的预热帧上输出 “Thanks for watching”。始终用 VAD 做门控。

### 2026 年生产参考栈

| Stack | Latency | License | Notes |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | 2026 年行业默认 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | DIY 友好 |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | 单模型；不同架构，第15课详述 |
| Vapi / Retell (managed) | 300-500 ms | commercial | 上线最快；定制受限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | 隐私 / 边缘部署 |

## 构建步骤

### 步骤 1：麦克风采集与分片（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 步骤 2：基于 VAD 的回合捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 步骤 3：流式 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 步骤 4：LLM 循环内的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 步骤 5：打断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用说明

参见 `code/main.py`，里面有一个可运行的模拟例子，将七个组件用桩模块连接起来，这样即便没有硬件也能看到流水线形态。要实现真实系统，请将桩替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（gpt-4o）或 `anthropic`
- `kokoro` 或 `cartesia`
- 用于 I/O 的 `sounddevice`

## 陷阱与注意事项

- **永久记录 PII。** 整段音频在大多数司法辖区都属于个人敏感信息（PII）。建议 30 天保留并在静态存储时加密。
- **没有打断能力。** 用户会打断。你的助手必须能停止讲话。
- **阻塞型 TTS。** 同步 TTS 会阻塞事件循环。使用异步或独立线程。
- **没有工具调用错误处理。** 工具会失败。LLM 需要收到错误信息并重试一次，然后优雅退化。
- **过度激进的幻觉过滤。** 过滤过度会导致助手不断说“我无法帮助”。过滤不足则会任意输出。用保留集校准阈值。
- **没有唤醒词选项。** 一直监听会带来隐私风险。加入唤醒词门控（Porcupine 或 openWakeWord）。

## 上线交付

保存为 `outputs/skill-voice-assistant-architect.md`。根据预算、规模、语言与合规约束，产出一份全栈规范。

## 练习

1. 简单：运行 `code/main.py`。它用桩模块模拟一个完整回合并打印各阶段延迟。
2. 中等：把 STT 桩替换成真实的 Whisper 模型，对预录的 `.wav` 测量 WER 和端到端延迟。
3. 困难：加入工具调用：实现 `get_weather`（任意 API）和 `set_timer`。将 LLM 路由至这些工具并验证当用户说“设一个 5 分钟定时器”时正确函数被触发且语音回复确认。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Turn | 一次回合 | 一个由 VAD 界定的用户语音 + 一个 LLM-TTS 回复。 |
| Barge-in | 打断 | 用户在助手讲话时插话；助手应停止。 |
| Wake word | “Hey assistant” | 短关键字检测器；Porcupine、Snowboy、openWakeWord。 |
| End-pointing | 回合结束判定 | VAD + 最小静默长度决定用户已说完。 |
| Pre-roll | 预语音缓冲 | 在 VAD 触发前保留 200-400 ms 音频以避免首词被截断。 |
| Tool call | 函数调用 | LLM 输出 JSON；运行时派发执行；结果在循环内回填。 |

## 拓展阅读

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 生产级参考。  
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY 友好框架。  
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 托管的语音原生路径。  
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工参考（第15课）。  
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — 唤醒词门控。  
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM 函数调用指南。