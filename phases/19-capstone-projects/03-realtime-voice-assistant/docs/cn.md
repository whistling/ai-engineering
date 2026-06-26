# Capstone 03 — 实时语音助手（ASR 到 LLM 到 TTS）

> 一个感觉自然的语音代理需要端到端延迟低于 800ms，能判断用户已停止说话，支持插话（barge-in），并能在不中断音频的情况下调用工具。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都达到了这一水平。它们采用相同的架构形态：流式 ASR、话语终止检测器（turn-detector）、流式 LLM 和流式 TTS，所有链路通过 WebRTC 连接，并在每一跳上设定严格的延迟预算。构建一个，实现对 WER、MOS 和 假切断率（false-cutoff rate）的测量，并在丢包条件下运行。

**Type:** 结业项目  
**Languages:** Python（agent + pipeline），TypeScript（web 客户端）  
**Prerequisites:** Phase 6（语音与音频）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（agents）、Phase 17（基础设施）  
**Phases exercised:** P6 · P7 · P11 · P13 · P14 · P17  
**Time:** 30 小时

## 问题（Problem）

语音是 2025–2026 年发展最快的 AI 交互形态之一。技术门槛每个季度都在降低。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都将首音输出低于 800ms 变为可及。考核的不只是延迟，还有交互手感：不打断用户、不被用户打断、能从句中插话恢复、在对话中调用工具而不卡住音频、能在不稳定移动网络中存活。

用三个同步的 REST 调用无法做到这一点。架构必须是端到端的流水线式流（pipelined streaming）。构建后，失败模式会变得可见：针对电话音频调优的 VAD 在背景电视时误触；等待标点的 turn-detector 永远收不到标点；TTS 在发声前缓冲 400ms。本 capstone 要求在压力下逐一修复这些问题并发布延迟与质量报告。

## 概念（Concept）

流水线包含五个流式阶段：**audio in**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**turn detection**（VAD + 一个读取部分转录以判断是否完成的小型 turn-detector 模型）、**LLM**（在判断回合结束后即时流式输出 token）、**TTS**（在首个 LLM token 后约 200ms 内流出音频）。

三个横切关注点。**Barge-in（插话）**：当用户在代理发声时开始说话，TTS 需要立即取消，ASR 要立刻捕捉到用户语音。**工具调用**：会话中途的函数调用（天气、日历）必须在侧通道并行执行，不应阻塞音频；如果工具延迟超过 300ms，代理需预先发出一个确认短语（如“稍等一下”）。**反压（Backpressure）**：在丢包情形下，部分转录会被保留，VAD 会提高语音门控阈值，代理避免在未被确认的消息上继续发言。

测量指标是量化的。Hamming VAD 基准在 15 dB SNR 下 WER 不高于 8%。100 次测量呼叫的首音输出 p50 小于 800ms。假切断率低于 3%。TTS MOS 高于 4.2。单台 g5.xlarge 上支持 50 个并发呼叫。这些数字是交付物。

## 架构（Architecture）

```
浏览器 / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (或 Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      每 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM（流式）
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS 流式
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     音频返回给呼叫方
            |
            v
   OpenTelemetry 语音追踪 -> Langfuse
```

## 技术栈（Stack）

- 传输：LiveKit Agents 1.0（WebRTC）加 Twilio PSTN 网关；备用框架为 Pipecat 0.0.70  
- ASR：Deepgram Nova-3（流式，首个 partial <300ms）或在 GPU 上自托管的 faster-whisper Whisper-v3-turbo  
- VAD：Silero VAD v5，加上 LiveKit 的 turn-detector（读取部分转录的小型 transformer）  
- LLM：紧耦合时使用 OpenAI GPT-4o-realtime，或 Gemini 2.5 Flash Live，或级联的 Claude Haiku 4.5（流式补全，独立的音频路径）  
- TTS：首字节最小延迟的 Cartesia Sonic-2、ElevenLabs Flash v3，或自托管的开源 Orpheus  
- 工具：用于天气/日历/预订的 FastMCP 侧通道；若工具耗时 >300ms，agent 预发填充短语  
- 可观测性：OpenTelemetry 语音 span，Langfuse 的语音追踪与音频回放  
- 部署：自托管 Whisper + Orpheus 使用单台 g5.xlarge（24GB VRAM）；对最低延迟优先使用托管 API

## 构建步骤（Build It）

1. WebRTC 会话。搭建一个 LiveKit 房间和一个将麦克风音频流到服务端的 web 客户端。在服务器端，附加一个加入该房间的 agent worker。

2. ASR 流式化。将 20ms 的 PCM 帧发送到 Deepgram Nova-3（或在 GPU 上运行的 faster-whisper）。订阅部分转录（partial）和最终转录（final）。记录每个 partial 的延迟。

3. VAD 与回合检测。对帧流运行 Silero VAD v5。在 speech-end 事件触发时，将最新的部分转录交给 LiveKit 的 turn-detector。仅在 VAD 显示静音 ≥ 500ms 且 turn-detector 的 completion 分数 > 0.6 时，才断定“回合完成”。

4. LLM 流。回合完成后，用当前对话上下文加最终转录启动 LLM 调用。流式输出 tokens。在第一个 token 出现时，切交给 TTS。

5. TTS 流。Cartesia Sonic-2 流式返回音频块。服务器必须在第一个 LLM token 后 200ms 内发送出首个音频块。将音频块发到 LiveKit 房间；客户端通过 WebRTC 的抖动缓冲（jitter buffer）播放。

6. 插话（Barge-in）。当 VAD 检测到用户在 TTS 播放时开始说话，立即取消 TTS 流，丢弃剩余的 LLM 输出，并重新启用 ASR。发布一个 `tts_canceled` span。

7. 工具侧通道。将天气和日历注册为函数调用工具。被调用时并发触发该调用；若 300ms 内未返回，LLM 发出“稍等一下，我去查一下”之类的填充短语；工具返回后继续。

8. 评估工具链。记录 100 次通话。计算 WER（与留出的参考转录比对）、假切断率（TTS 在用户半句时被取消的比率）、首音输出 p50、TTS MOS（人工打分或 NISQA 自动代理）以及抖动-丢包测试（注入 3% 丢包）。

9. 负载测试。在一台 g5.xlarge 上用合成呼叫驱动 50 个并发呼叫。测量持续情况下的首音输出 p95。

## 使用示例（Use It）

```
caller: "明天东京的天气怎么样"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

（注：对话示例中部分转录保留原文以便比对；在实际实现中可替换为目标语言文本）

## 交付（Ship It）

`outputs/skill-voice-agent.md` 是交付物。针对一个领域（客户支持、日程安排或自助服务终端），部署一个带 ASR/VAD/LLM/TTS 流式流水线并调优至测量目标。评分标准：

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | End-to-end latency | 在 100 次记录呼叫中，首音输出 p50 小于 800ms |
| 20 | Turn-taking quality | 在 Hamming VAD 基准上假切断率低于 3% |
| 20 | Tool-use correctness | 会话中途调用工具能返回正确数据且不阻塞音频 |
| 20 | Reliability under packet loss | 注入 3% 丢包时的 WER 与回合接续稳定性 |
| 15 | Eval harness completeness | 可复现的测量和公开配置 |
| **100** | | |

## 练习题（Exercises）

1. 将 Deepgram Nova-3 替换为在 g5.xlarge 上运行的 faster-whisper v3 turbo。测量延迟与 WER 差异。识别 CPU 与 GPU 决策影响的关键环节。

2. 添加插话仲裁策略：当用户在工具调用期间插话，agent 应如何处理？比较三种策略（直接强制取消、等待工具完成再停止、将下一轮排队）。

3. 运行针对回合检测的对抗性测试：在句中给用户长时间停顿。为在不超过 900ms 的前提下降低假切断率，调优 VAD 的静音阈值和 turn-detector 的分数阈值。

4. 将相同 agent 部署到 Twilio 的 PSTN。比较 PSTN 与 WebRTC 的首音输出差异。解释抖动缓冲与编解码器的差异。

5. 为非英语语言（如日语、西班牙语）添加语音活动检测。测量 Silero VAD v5 与针对语言微调模型的误触发率差异。

## 关键词（Key Terms）

| 术语 | 人们常说 | 实际含义 |
|------|---------|---------|
| 回合检测（Turn detection） | “话语结束” | 给定 VAD 静音和部分转录后，判定用户已停止说话的分类器 |
| 插话（Barge-in） | “中断处理” | 当 VAD 检测到新的用户语音时，取消正在播放的 TTS |
| 首音输出（First-audio-out） | “延迟” | 从用户停止说话到服务器发出首个音频包的时间 |
| VAD | “语音门控” | 将音频帧分类为语音或静音的模型；Silero VAD v5 是 2026 年默认选择 |
| 抖动缓冲（Jitter buffer） | “音频平滑” | 客户端侧短时缓存数据包以吸收网络抖动 |
| 填充语（Filler） | “确认短语” | 当工具响应缓慢时，代理发出的短语以避免静默 |
| MOS | “平均意见分数” | 感知语音质量评分；NISQA 是常用的自动代理工具 |

（注：文中术语已采用标准中文 AI 工程术语，例如提示词工程、RAG、嵌入、微调、上下文窗口、少样本、思维链、护栏、函数调用、投机性解码、位置嵌入、自注意力、指令微调、分布式训练等）

## 延伸阅读（Further Reading）

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — WebRTC agent 框架参考  
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 另一个以 Python 为先的流式 agent 框架  
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型参考  
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — 流式 ASR 参考  
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型  
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考  
- [Retell AI architecture](https://docs.retellai.com) — 生产级语音代理架构参考  
- [Vapi.ai production stack](https://docs.vapi.ai) — 另一个生产参考