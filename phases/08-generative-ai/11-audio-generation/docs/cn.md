# 音频生成

> 音频是 16–48 kHz 的一维信号。一个五秒的片段是 80–240k 个样本。没有任何 transformer 会直接对这样的序列进行注意力计算。到 2026 年，每个生产级音频模型的解决方案都是相同的：神经编解码器（Encodec、SoundStream、DAC）将音频压缩成 50–75 Hz 的离散令牌，然后由 transformer 或扩散模型生成这些令牌。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)  
**Time:** ~45 分钟

## 问题

三类音频生成任务：

1. **文本到语音（Text-to-speech）。** 给定文本，生成语音。干净的语音带宽窄且具有强烈的语音结构——由 transformer 在令牌上建模效果良好。代表作品有 VALL-E（Microsoft）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定一个提示（文本、旋律、和弦进程、风格），生成音乐。分布更广、更复杂。代表作品有 MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效/声音设计。** 给定提示，生成环境音或 Foley。代表作品有 AudioGen、AudioLDM 2、Stable Audio Open。

这三类任务都运行在相同的基底之上：神经音频编解码器 + token-AR 或扩散生成器。

## 概念

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。卷积编码器将波形压缩为每时间步的向量；残差向量量化（RVQ）将每个向量转换为 K 个码本索引的级联。解码器则反向还原。举例：24 kHz 音频在 2 kbps 下，使用 8 个 RVQ 码本、75 Hz 的速率 = 600 令牌/秒。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 两种上层生成范式

**Token-autoregressive（令牌自回归）。** 将 RVQ 令牌展平为序列，运行 decoder-only transformer。MusicGen 使用 “delayed parallel” 在并行流中发出 K 个码本流，并对每个流应用偏移。VALL-E 从文本提示 + 3 秒的语音样本生成语音令牌。

**Latent diffusion（潜在扩散）。** 将编解码器令牌打包为连续潜变量或用分类扩散建模它们。Stable Audio 2.5 在连续音频潜变量上使用 flow matching。AudioLDM 2 使用 text-to-mel-to-audio 的扩散流程。

2024–2026 年的趋势：flow matching 在音乐上更胜一筹（推理更快、样本更干净），而 token-AR 在语音上仍占主导，因为它天然具备因果性并易于流式输出。

## 生产格局

| System | Task | Backbone | Latency |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms 首个令牌 |
| OpenAI GPT-4o audio | Full-duplex speech | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | 非流式 |
| Stable Audio 2.5 | Music / SFX | DiT + flow matching on audio latents | 1 分钟片段约 ~10s |
| Suno v4 | Full songs | 未公开；疑为 token-AR | 每首歌约 ~30s |
| Udio v1.5 | Full songs | 未公开 | 每首歌约 ~30s |
| MusicGen 3.3B | Music | Token-AR on Encodec 32kHz | 实时 |
| AudioCraft 2 | Music + SFX | Flow matching | 5s 音频约 ~5s |
| Riffusion v2 | Music | 频谱扩散 | ~10s |

## 构建它

`code/main.py` 模拟核心思路：在由两种不同“风格”生成的合成“音频令牌”序列上训练一个微小的 next-token transformer（风格 A 为交替的低高令牌，风格 B 为单调的斜坡）。基于风格条件进行采样。

### 步骤 1：合成音频令牌

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

（注：上面代码中的注释已翻译为中文以便开发者阅读）

### 步骤 2：训练一个微小的令牌预测器

一个基于二元模型（bigram-style）的预测器，带有风格条件。关键点是模式：编解码器令牌 → 交叉熵训练 → 自回归采样。

### 步骤 3：条件采样

给定风格令牌和起始令牌，从预测的分布中采样下一个令牌。持续生成 20–40 个令牌。

## 陷阱

- **编解码器质量限定输出质量。** 如果编解码器不能忠实表示某个声音，再好的生成器也无济于事。DAC 是当前开源中最好的选择。
- **RVQ 的误差累积。** 每个 RVQ 层建模前一层的残差。第一层的误差会向上传播。在更高层上使用温度为 0 的采样会有所帮助。
- **音乐结构。** 以 75 Hz 的速率，30 秒的令牌约为 20k+ 个令牌。对 transformers 来说非常困难。MusicGen 使用滑动窗口 + 提示续写；Stable Audio 使用更短片段 + 交叉淡化（crossfading）。
- **片段边界伪影。** 在生成片段之间交叉淡化需要仔细的重叠-相加（overlap-add）。
- **对干净数据的需求。** 音乐生成器需要数万小时的授权音乐数据。Suno / Udio 的 RIAA 诉讼（2024 年）将这一点推到了风口浪尖。
- **语音克隆的伦理问题。** 仅一个 3 秒的样本加上文本提示就足以被 VALL-E / XTTS / ElevenLabs 克隆声音。每个生产模型都需要滥用检测和退出名单（opt-out lists）。

## 使用建议

| Task | 2026 堆栈 |
|------|------------|
| 商业 TTS | ElevenLabs、OpenAI TTS，或 Azure Neural |
| 语音克隆（经同意） | XTTS v2（开源）或 ElevenLabs Pro |
| 背景音乐，速度优先 | Stable Audio 2.5 API、Suno，或 Udio |
| 含歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / Foley | AudioCraft 2、ElevenLabs SFX，或 Stable Audio Open |
| 实时语音代理 | GPT-4o realtime 或 Gemini Live |
| 开源权重的音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 部署说明（Ship It）

保存为 `outputs/skill-audio-brief.md`。该技能接受音频简报（任务、时长、风格、声音、许可），输出：模型 + 托管方案、提示格式（流派标签、风格描述符、结构标记）、编解码器 + 生成器 + 波形合成链（codec + generator + vocoder chain）、随机种子协议、以及评估计划（MOS / CLAP 分数 / TTS 的 CER / 用户 A/B 测试）。

## 练习

1. **简单。** 运行 `code/main.py` 并明确设置风格。验证生成的序列是否匹配该风格的模式。
2. **中等。** 添加 delayed parallel 解码：模拟 2 个令牌流，它们必须保持 1 步的偏移。训练一个联合预测器。
3. **困难。** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三种不同提示生成一个 10 秒片段；对风格一致性做 A/B 测试。

## 关键词

| Term | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Codec | “神经压缩” | 音频的编码器 / 解码器；典型输出是 50–75 Hz 的令牌。 |
| RVQ | “Residual VQ” | K 级量化器的级联；每级建模上一级的残差。 |
| Token | “一个编解码器符号” | 码本中的离散索引；常见大小为 1024 或 2048。 |
| Delayed parallel | “偏移码本” | 发出 K 条令牌流并错开偏移以减少序列长度。 |
| Flow matching | “2024 年音频的胜出方法” | 比扩散更直的路径式方法；采样更快。 |
| Voice prompt | “3 秒样本” | 说话人嵌入或令牌前缀，用以引导克隆声音。 |
| Mel spectrogram | “那张图” | 对数幅度感知谱图；许多 TTS 系统使用。 |
| Vocoder | “从梅尔到波形” | 将 mel 谱图转换回音频的神经组件。 |

## 生产注意：音频是一个流式问题

音频是唯一一种用户期望“随生成同时到达”的输出模态，而不是一次性全部得到。从生产角度看，这意味着 TPOT（每输出令牌所需时间）很重要，因为用户的听速是目标吞吐率 —— 不是他们的阅读速度。对于以 ~75 令牌/秒（Encodec）的 16 kHz 音频，服务器必须为每个用户生成 ≥75 令牌/秒以保持播放平滑。

两个架构性结论：

- **Flow-matching 音频模型无法轻易实现流式。** Stable Audio 2.5 和 AudioCraft 2 在一个批次中渲染固定长度的片段。要实现流式，你必须对片段分块并重叠边界——类似滑动窗口的扩散——这会在延迟上增加 100–300 ms 的开销，相较于 codec AR 模型。
  
如果产品是“实时语音聊天”或“实时音乐续写”，请选择 codec AR 路径。如果是“提交后渲染 30 秒片段”，flow-matching 在质量和总体延迟上更占优。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 编解码器标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 首个被广泛使用的神经音频编解码器。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025 年基于 flow matching 的文本到音乐模型。