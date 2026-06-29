# Streaming Speech-to-Speech — Moshi, Hibiki, and Full-Duplex Dialogue

> 2024–2026 年重新定义了语音 AI。Moshi 提供了一个能够同时听与说的单模型，延迟为 200 ms。Hibiki 实现了逐块的流式语音到语音翻译。两者都放弃了 ASR → LLM → TTS 的管线式流程，转而采用基于 Mimi codec tokens 的统一全双工架构。这是新的参考设计。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** 第 6 阶段 · 13（神经音频编解码器），第 6 阶段 · 11（实时音频），第 7 阶段 · 05（完整 Transformer）  
**Time:** ~75 分钟

## 问题

基于第 11 与第 12 课构建的每个语音代理，都存在大约 300–500 ms 的基本延迟下限：VAD 触发，STT 处理，LLM 推理，TTS 生成。每个阶段都有自己的最小延迟。你可以调优并行化，但流水线结构会限制上界。

Moshi（Kyutai，2024–2026）提出了一个不同的问题：如果不存在流水线怎么办？如果有一个模型直接、持续地接收音频并输出音频，而文本只是作为一个“内部独白”的中间表示，而不是必需阶段呢？

答案是全双工语音到语音。理论延迟 160 ms（80 ms Mimi 帧 + 80 ms 声学延迟）。在单个 L4 GPU 上的实际延迟为 200 ms。相比最先进的管线式语音代理，这几乎减半。

## 概念

![Moshi 架构：两个并行的 Mimi 流 + 内部独白文本](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两个 Mimi codec 流，两者均为 12.5 Hz × 8 个 codebook：

- 流 1：用户音频（Mimi 编码，持续到达）
- 流 2：Moshi 自身的音频（由 Moshi 生成）

**Transformer。** 一个 7B 参数的时序 Transformer 同时处理两条音频流和一条文本“内部独白”流。在每个 80 ms 步长，它：

1. 消耗最新的用户 Mimi tokens（8 个 codebook）。
2. 消耗最近的 Moshi Mimi tokens（8 个 codebook，按生成顺序）。
3. 生成下一个 Moshi 文本 token（内部独白）。
4. 生成下一个 Moshi Mimi tokens（通过一个小的 Depth Transformer 生成 8 个 codebook）。

所有三条流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以在说话时听到用户；当用户打断时可以自行中断；可以在不打断主话语的情况下发出回话（例如 “mhm”）。

**Depth transformer。** 在一个帧内，8 个 codebook 并不是并行预测的——它们之间存在相互依赖。一个小的两层“深度变换器”在 80 ms 内顺序地预测它们。这是自回归编解码器语言模型的标准分解方式（VALL-E、VibeVoice 也使用类似方法）。

### 为什么内部独白文本有帮助

没有显式的文本，模型必须在其声学流中隐式地建模语言。Moshi 的洞见是：强制它在生成音频的同时也输出文本 token。文本流本质上是 Moshi 正在说话内容的逐词记录。这提升了语义一致性，便于替换语言模型头，并且能免费得到转录文本。

### Hibiki：流式语音到语音翻译

相同的架构，训练在翻译对上。源语言音频输入，目标语言音频输出，连续进行。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的需求——使用句级数据 + GRPO 强化学习来优化延迟。

最初支持四个语言对；适配新语言大约需要 ≈1000 小时训练数据。

### 更广泛的 Kyutai 生态（2026）

- **Moshi** — 全双工对话（优先支持法语，英语有良好支持）  
- **Hibiki / Hibiki-Zero** — 同步语音翻译  
- **Kyutai STT** — 流式 ASR（500 ms 或 2.5 s 的预览）  
- **Kyutai Pocket TTS** — 100M 参数 TTS 可在 CPU 上运行（2026 年 1 月）  
- **Unmute** — 在公共服务器上将这些功能组合成完整管线

在 L40S GPU 上吞吐量：64 个并发会话，3× 实时速率。

### Sesame CSM — 近亲

Sesame CSM（2025）使用了相似的思路——一个 Llama-3 主干加上 Mimi codec 头。但 CSM 是单向的（接受上下文 + 文本，生成语音），而非全双工。它是市场上最好的“语音存在感”TTS；但与 Moshi 的全双工能力不完全相同。

### 2026 年性能数据

| Model | Latency | Use case | License |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | 全双工英语 / 法语对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | 法语 ↔ 英语 流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | same | 5 个语言对，无需对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 上下文条件 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | 封闭，OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 ms | 封闭，Google API | 商业 |

## 构建

### 步骤 1：接口

Moshi 暴露一个 WebSocket 服务器，接收 80 ms 的 Mimi 编码音频块并返回 80 ms 的 Mimi 编码音频块。双向且持续。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 步骤 2：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输实现。

### 步骤 3：训练目标（概念性）

对于每个 80 ms 的帧 t：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`  
- 预测：`moshi_text[t]`，然后 `moshi_mimi[t, codebook_0..7]`

文本先于音频预测（内部独白）；音频在深度变换器内按 codebook 顺序预测。

### 步骤 4：Moshi 的优势与局限

Moshi 的优势：

- 在廉价硬件上实现 <250 ms 的端到端延迟。  
- 自然的回话短回应与被打断处理。  
- 无需管线胶水代码。

Moshi 的局限：

- 工具调用能力有限（未针对工具调用训练；需要单独的 LLM 路径）。  
- 长时间推理能力不足（Moshi 大约是 8B 规模的对话模型，并非 Claude/GPT-4 级别）。  
- 在小众主题上的事实准确性一般。  
- 大多数生产级企业用例仍然倾向于使用管线（2026 年仍然如此）。

## 使用

| 情景 | 选择 |
|-----------|------|
| 最低延迟的语音伴侣 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示 / 研究 | Moshi、CSM |
| 带工具调用的企业代理 | 管线（第 12 课），而不是 Moshi |
| 上下文中的自定义语音 TTS | Sesame CSM |
| 任意语言的语音到语音 | GPT-4o Realtime 或 Gemini 2.5 Live（商业） |

## 陷阱

- **工具调用受限。** Moshi 是对话模型，不是代理框架。与管线结合以支持工具。  
- **特定声音的条件化。** Moshi 使用单一训练的人设；克隆特定声线需要单独训练。  
- **语言覆盖。** 法语 + 英语表现优异；其他语言支持有限。Hibiki-Zero 有帮助，但仍需训练数据。  
- **资源成本。** 一个完整的 Moshi 会话占用一个 GPU 插槽；不是廉价的共享租户部署模式。

## 部署

保存为 `outputs/skill-duplex-pipeline.md`。根据语音代理的工作负载与需求，权衡选择管线架构或全双工架构，并给出理由。

## 练习

1. 简单：运行 `code/main.py`。它符号化地模拟了两条流 + 内部独白架构。  
2. 中等：从 HuggingFace 拉取 Moshi，运行服务器，测试一次对话。测量从用户结束说话到 Moshi 开始响应的真实墙钟延迟。  
3. 困难：将你的第 12 课管线代理与 Moshi 在 20 个匹配测试语句上比较 P50 延迟。写报告说明在哪些场景下管线在架构上仍然胜出。

## 术语表

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|---------|
| 全双工 (Full-duplex) | 一边听一边说 | 在同一模型上同时激活两条音频流。 |
| 内部独白 (Inner monologue) | 模型的文本流 | Moshi 在其音频输出的同时，输出对应的文本 token。 |
| 深度变换器 (Depth transformer) | 码本间的预测器 | 小型 transformer，在一个 80 ms 帧内按顺序预测 8 个 codebook。 |
| Mimi | Kyutai 的编解码器 | 12.5 Hz × 8 个 codebook；包含语义 + 声学信息；驱动 Moshi。 |
| 流式 S2S (Streaming S2S) | 音频 → 音频 实时 | 逐块的翻译/对话，无管线阶段。 |
| 回话短回应 (Back-channeling) | “mhm” 式的反应 | Moshi 可以在不打断主话语的情况下发出小型确认或回应。 |

## 延伸阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 论文。  
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 无对齐数据的流式翻译。  
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 规范。  
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — 安装 + 服务器。  
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 封闭的商业对等产品。  
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层的 STT/TTS 框架。