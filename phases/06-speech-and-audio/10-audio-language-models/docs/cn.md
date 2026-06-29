# Audio-Language Models — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> 到 2026 年，音频-语言模型能对语音 + 环境声 + 音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上可与 GPT-4o Audio 匹敌。Audio Flamingo Next 在 LongAudioBench 上击败 Gemini 2.5 Pro。在多音频任务上，开源与闭源之间的差距基本上已被缩小——唯一的例外是多音频对比任务，所有模型的表现都接近随机水平。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 6 · 04 (ASR 自动语音识别), Phase 12 · 03 (Vision-Language Models 视觉-语言模型), Phase 7 · 10 (Audio Transformers 音频 Transformer)  
**Time:** ~45 分钟

## 问题

你有一段 5 秒的音频：狗叫声，有人喊 “stop!”，然后是静默。需回答的问题跨越多个维度：

- **转录。** “说了什么？” — 属于 ASR 领域。
- **语义推理。** “那个人处于危险中吗？” — 需要联合理解狗叫 + 喊声 + 静默。
- **音乐推理。** “哪些乐器在演奏主旋律？”
- **长音频检索。** “在这 90 分钟的讲座中，讲师在何处讲解了梯度下降？”

能用单一提示回答上述所有问题的模型即为一个 **音频-语言模型**（LALM / ALM）。它不同于纯 ASR：LALM 产生自由格式的自然语言答案，而不仅仅是转录文本。

## 概念

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### 三组件模板

每个 2026 年的 LALM 都有相同的骨架：

1. **Audio encoder。** Whisper encoder、BEATs、CLAP、WavLM，或每个模型自定义的编码器。  
2. **Projector。** 将音频编码器特征桥接到 LLM 的 token 嵌入空间的线性层或 MLP。  
3. **LLM。** 基于 Llama / Qwen / Gemma 的解码器。接受交错的文本 + 音频 token；生成文本。

训练流程：

- **Stage 1.** 冻结编码器与 LLM；仅训练 projector，使用 ASR / captioning 数据。  
- **Stage 2.** 在指令式音频任务（QA、推理、音乐理解）上进行全量或 LoRA 微调。  
- **Stage 3（可选）。** 添加语音输入/输出需要增加语音解码器。Qwen2.5-Omni 与 AF3-Chat 做到了这一点。

### 2026 年的模型地图

| Model | Backbone | Audio encoder | Output modality | Access |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio (closed) | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准现实检验（2026）

**MMAU-Pro。** 包含 1800 个涵盖语音 / 声音 / 音乐 / 混合的 QA 条目。包含多音频子集。

| Model | Overall | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench 上的 SOTA | — | — | — | — |

“多音频”列对所有模型来说都很致命。四选多项选择题的随机猜测概率 = 25%；大多数模型得分就在这一带。LALM 在比较两个或多个音频片段时仍然表现不佳。

### 2026 年 LALM 的适用场景

- **呼叫中心录音合规审计。** “坐席是否提及了必需的披露信息？”  
- **无障碍。** 向听障用户描述声事件（不仅仅是转录）。  
- **内容审核。** 检测暴力语言 + 威胁语气 + 背景上下文。  
- **播客 / 会议分章。** 语义摘要，而不仅仅是说话人切换。  
- **音乐目录分析。** “查找所有包含 B 段转调的曲目。”

### 尚不适合的场景

- 细粒度音乐理论（和弦以下的细节）。  
- 在长对话中进行说话人归因的推理（超过 10 分钟后性能下降）。  
- 多音频比较（22–26% 几乎等于随机）。  
- 实时流式推理（大多数为离线批量推理）。

## 构建它

### 步骤 1：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 步骤 2：projector 模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就这样。projector 通常是 1–3 层线性层。在 ASR 配对（音频 → 转录）上训练它是 Stage-1 的预训练任务。

### 步骤 3：在 MMAU / LongAudioBench 上基准测试

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

请分别报告各类（语音 / 声音 / 音乐 / 多音频）的分数。聚合数字会掩盖模型失败的具体领域。

## 使用建议

| Task | 2026 推荐 |
|------|-----------|
| 自由格式音频问答（开源） | Qwen2.5-Omni-7B |
| 长音频最佳开源模型 | Audio Flamingo Next |
| 最佳闭源模型 | Gemini 2.5 Pro |
| 语音输入 / 语音输出代理 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（音乐专用 AF-CLAP） |
| 呼叫中心审计 | 通过 API 使用 Gemini 2.5 Pro，结合对政策文档的 RAG |

## 陷阱

- **对多音频的过度信任。** 如果你的任务是“哪个片段有 X”，则随机水平的性能是真实存在的。  
- **长音频退化。** 超过 10 分钟后，大多数模型的说话人归因会崩溃。先做说话人分离（Lesson 6），再总结。  
- **静默时的幻觉。** LALM 中使用 Whisper 编码器的模型会继承 Whisper 风格的问题。在静默段落要做 VAD 门控。  
- **基准挑选偏差。** 厂商博客常突出最优类别。自己在 MMAU-Pro 的多音频子集上跑一遍。

## 部署

保存为 `outputs/skill-alm-picker.md`。为给定的音频理解任务选定 LALM + 基准子集 + 输出模态（文本或语音）。

## 练习

1. **简单。** 运行 `code/main.py`，查看一个玩具级的 projector 模式 + 将 (audio-embedding, text-tokens) 路由到输出 tokens 的假 LALM。  
2. **中等。** 在 100 个 MMAU-Pro 的语音条目上评估 Qwen2.5-Omni-7B。与论文报告的结果比较。  
3. **困难。** 构建一个最小的音频字幕基线：BEATs 编码器 + 2 层 projector + 冻结的 Llama-3.2-1B。仅对 projector 在 AudioCaps 上进行微调。与 SALMONN 在 Clotho-AQA 上比较。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder. |
| Projector | Adapter | 小型 MLP，将音频特征映射到 LLM 的嵌入空间。 |
| MMAU | The benchmark | 覆盖语音、声音、音乐的 10k 音频 QA 对。 |
| MMAU-Pro | Harder MMAU | 包含 1800 个多音频 / 强推理题的问题集。 |
| LongAudioBench | Long-form eval | 包含多分钟片段并带语义查询的评测集。 |
| Voice-in / voice-out | Speech-native | 模型直接接收语音并输出语音，无需经由文本中转。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。  
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 支持语音进/语音出。  
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源长音频领域的领先者。  
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench 的 SOTA。  
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱。  
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — 2026 年实时排名。