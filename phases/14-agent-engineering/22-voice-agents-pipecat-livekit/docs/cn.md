# Voice Agents: Pipecat and LiveKit

> 语音智能体在 2026 年成为一等公民的生产类别。Pipecat 为你提供一个基于帧的 Python 管道框架（VAD → STT → LLM → TTS → 传输）。LiveKit Agents 将 AI 模型通过 WebRTC 桥接到用户。优质堆栈的生产端到端延迟目标为 450–600ms。

**Type:** 学习  
**Languages:** Python (标准库)  
**Prerequisites:** Phase 14 · 01 (智能体循环), Phase 14 · 12 (工作流模式)  
**Time:** ~60 分钟

## 学习目标

- 描述 Pipecat 的基于帧的管道：下行（DOWNSTREAM，source→sink）与上行（UPSTREAM，控制）。
- 列出规范语音管道阶段以及 Pipecat 支持哪些传输方式。
- 解释 LiveKit Agents 的两类语音智能体（MultimodalAgent、VoicePipelineAgent）以及各自适用场景。
- 概述 2026 年的生产延迟预期及其如何驱动架构选择。

## 问题

语音智能体不是一个带上 TTS 的文本循环。延迟预算非常苛刻（~600ms），部分音频是默认输入，回合检测是一个模型，传输方式从电话 SIP 到 WebRTC 都可能存在。要么你构建一个基于帧的管道（Pipecat），要么依赖某个平台（LiveKit）。

## 概念

### Pipecat (pipecat-ai/pipecat)

- 基于帧的 Python 管道框架。
- `Frame` → `FrameProcessor` 链式处理。
- 两种流向：
  - **DOWNSTREAM（下行）** — source → sink（音频输入，TTS 输出）。
  - **UPSTREAM（上行）** — 反馈与控制（取消、指标、插话/打断）。
- `PipelineTask` 管理生命周期并提供事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）以及用于指标/追踪/RTVI 的观察者。

典型管道：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

传输：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 增加结构化会话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents (livekit/agents)

- 将 AI 模型通过 WebRTC 桥接到用户。
- 关键概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两类语音智能体：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或等价服务直接传输音频。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本级控制能力。
- 语义回合检测通过 transformer 模型实现。
- 原生 MCP 集成。
- 支持通过 SIP 的电话接入。
- LiveKit Inference 提供 50+ 无需 API key 的模型；通过插件可接入 200+ 其它模型。

### 商业平台

Vapi（在优化的高端堆栈上约 450–600ms）和 Retell（在 180 次测试呼叫上端到端约 600ms）构建于此类技术之上。如果你想要托管的语音栈而不想组建 WebRTC 团队，可以选择平台。

### 该模式常见失误

- **没有处理插话（barge-in）。** 用户打断但智能体仍在继续说话。需要 Pipecat 的 UPSTREAM 取消帧或 LiveKit 的等价机制。
- **忽视 STT 置信度。** 把低置信度的转录当作真理喂给 LLM。应基于置信度进行门控或请求确认。
- **TTS 在句中被截断。** 管道在话语中途取消时，TTS 需要知晓或能切断音频。
- **忽视延迟预算。** 每个组件都会增加 50–200ms。在上线前对链路进行求和评估。

### 典型 2026 延迟

- VAD：20–60ms
- STT（部分结果）：100–250ms
- LLM 首个 token：150–400ms
- TTS 首个音频：100–200ms
- 传输 RTT：30–80ms

端到端 450–600ms 属于高端体验。800–1200ms 常见。任何 >1500ms 的感觉就是坏体验。

## 构建它

`code/main.py` 是一个基于帧的玩具管道，包含：

- `Frame` 类型（audio、transcript、text、tts_audio、control）。
- `Processor` 接口，带 `process(frame)` 方法。
- 五阶段管道（VAD → STT → LLM → TTS → transport）以脚本化处理器实现。
- 一个用于演示插话的 UPSTREAM cancel 帧，展示如何中止。

运行它：

```
python3 code/main.py
```

跟踪输出会展示正常流程以及一次插话取消，停止 TTS 的播放。

## 使用建议

- 使用 **Pipecat** 可实现完全控制 —— 自定义处理器、以 Python 为先、可插拔的提供方。
- 使用 **LiveKit Agents** 适合以 WebRTC 为先的部署和电话接入场景。
- 使用 **Vapi / Retell** 在没有 WebRTC 团队时选择托管语音智能体。
- 使用 **OpenAI Realtime / Gemini Live** 可实现直接的音频输入/输出（适用于 MultimodalAgent）。

## 交付

`outputs/skill-voice-pipeline.md` 提供了一个以 Pipecat 形态搭建的语音管道脚手架，包含 VAD + STT + LLM + TTS + 传输 以及 插话处理。

## 练习

1. 为你的玩具管道添加一个指标观察者：统计每个阶段每秒的帧数。延迟在哪里积累？
2. 实现基于置信度的 STT 门控：低于阈值时，请求“你能再说一遍吗？”。
3. 添加语义回合检测：简单规则 —— 如果转录以“?”结尾，则视为回合结束。
4. 阅读 Pipecat 的传输文档。把标准库传输替换为 SmallWebRTCTransport 的配置（可用 stub）。
5. 对同一个查询测量 OpenAI Realtime 与 STT+LLM+TTS 级联的延迟差异。文本级控制带来多少延迟成本？

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Frame | "Event" | 管道中的类型化数据单元（audio、transcript、text、control） |
| Processor | "Pipeline stage" | 带有 process(frame) 的处理器 |
| DOWNSTREAM | "Forward flow" | 源到汇：音频输入，语音输出 |
| UPSTREAM | "Feedback flow" | 控制通路：取消、指标、插话 |
| VAD | "Voice activity detection" | 检测用户是否在说话 |
| Semantic turn detection | "Smart end-of-turn" | 基于模型的用户结束回合判定 |
| MultimodalAgent | "Direct audio agent" | 音频进、音频出；中间没有文本 |
| VoicePipelineAgent | "Cascade agent" | STT + LLM + TTS；提供文本级控制 |

（注：文中术语翻译遵循常见 AI 工程术语：提示词工程、RAG、嵌入、微调、上下文窗口、少样本、思维链、护栏、函数调用、智能体循环、有状态图、参与者模型）

## 延伸阅读

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — 基于帧的管道、处理器、传输
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC 与语音原语
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音，延迟基准测试资料