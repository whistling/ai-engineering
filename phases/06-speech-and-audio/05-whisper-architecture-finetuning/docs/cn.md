# Whisper — Architecture & Fine-Tuning

> Whisper 是一个处理 30 秒窗口的 transformer 编码器-解码器，基于 68 万小时的多语言弱监督音频-文本对训练。单一架构，多任务支持，在 99 种语言上表现稳健。2026 年参考级 ASR。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (注意力), Phase 7 · 05 (完整 Transformer)  
**Time:** ~75 分钟

## 问题

Whisper（OpenAI 于 2022 年 9 月发布）是首个以商品级方式发布的 ASR 模型：粘贴音频，得到文本，支持 99 种语言，对噪声稳健，并能在笔记本上运行。到 2024 年，OpenAI 发布了 Large-v3 和 Turbo 变体；到 2026 年，Whisper 已成为从播客转写到语音助手再到 YouTube 字幕的默认基线。

但 Whisper 不是一个可以永远作为黑箱对待的管道。领域漂移会击垮它——专业术语、说话者口音、专有名词、短片段、静音。你需要知道：

1. 它内部到底是什么。
2. 如何正确地提供分块、流式或长语音输入。
3. 何时进行微调以及如何微调。

## 概念

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**架构。** 标准的 transformer 编码器-解码器。

- 输入：30 秒的对数梅尔谱（log-mel spectrogram），80 个梅尔频带，10 ms hop → 3000 帧。短于窗口的剪辑会用零填充，长于窗口的会被分块。
- 编码器：卷积下采样（stride 2）+ N 个 transformer block。Large-v3：32 层，1280 维，20 个头。
- 解码器：N 个 transformer block，带因果自注意力（causal self-attn）+ 对编码器输出的交叉注意力（cross-attn）。与编码器相同规模。
- 输出：基于 BPE 的词表，51,865 个 token。

Large-v3 约有 15.5 亿参数。Turbo 采用 4 层解码器（从 32 层降至 4 层），将延迟降低约 8×，WER 损失 <1%。

**提示词格式（Prompt format）。** Whisper 是一个多任务模型，通过解码器提示词中的特殊 token 来引导任务：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 语言标签；控制翻译与转录行为。
- `<|transcribe|>` 或 `<|translate|>` — 前者对输入做逐字转录，后者将任意语言输入翻译为英语。
- `<|notimestamps|>` — 跳过词级时间戳（更快）。

提示词使得一个模型完成多种任务成为可能。将 `<|en|>` 改为 `<|fr|>` 即可转录法语。

**30 秒窗口。** 一切都以 30 秒为单位。更长的音频需要分块；更短的音频会被填充。窗口本身并不原生支持流式——这也是 WhisperX、Whisper-Streaming 和 faster-whisper 出现的原因。

**对数梅尔归一化。** 使用 (log_mel - mean) / std，其中均值与方差统计来自 Whisper 的训练语料。你必须使用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），而非 `librosa.feature.melspectrogram`。

### 2026 年的变体

| Variant | Params | Latency (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× 实时 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | 流式 | 2.0% |

### 微调（Fine-tuning）

2026 年的典型工作流程：

1. 收集 10–100 小时的目标域音频并对齐转录文本。
2. 使用 `transformers.Seq2SeqTrainer` 并加上 `generate_with_loss` 回调。
3. 参数高效方法：在注意力层的 `q_proj`、`k_proj`、`v_proj` 上使用 LoRA，可将 GPU 内存减少约 4×，WER 影响 <0.3。
4. 若数据少于 10 小时，冻结编码器；只微调解码器。
5. 使用 Whisper 自带的 tokenizer 和提示词格式；不要替换 tokenizer。

社区成果：在医疗听写数据上对 Medium 微调 20 小时，医疗词汇上的 WER 从 12% 降到 4.5%。对 Turbo 在 4 小时冰岛语数据上微调，WER 从 18% 降到 6%。

## 构建

### 步骤 1：直接运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 防止失控的重复
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应始终覆盖的重要默认项：`temperature=0.0`（采样的默认链为 0.0 → 0.2 → 0.4 …），`condition_on_previous_text=False`（防止级联式幻觉问题），以及 `no_speech_threshold=0.6`（静音检测阈值）。

### 步骤 2：分块处理长音频

```python
# whisperx 是 2026 年用于长音频并带词级时间戳的参考实现
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 增加了：(1) Silero VAD 作为门控，(2) 基于 wav2vec 2.0 的词级对齐，(3) 通过 `pyannote.audio` 的说话人分离。是 2026 年生产转写的主力方案。

### 步骤 3：用 LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M 可训练 / 809M 总计
```

然后按常规 Trainer 循环训练。每 1000 步保存一次 checkpoint。在留出集上用 WER 评估。

### 步骤 4：检查每层学到了什么

```python
# 在解码过程中抓取 cross-attention 权重以查看解码器关注的内容。
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

用热力图可视化——你会看到对角线对齐模式，因为解码器步进时在扫描编码器帧。那条对角线就是 Whisper 的词时间戳概念。

## 使用

2026 年的实践栈：

| 情况 | 选择 |
|------|------|
| 通用英语，离线 | 通过 `whisperx` 使用 Large-v3-turbo |
| 移动 / 边缘 | 量化的 Whisper-Tiny（int8）或 Moonshine |
| 多语种长音频 | 通过 `whisperx` + 说话人分离 使用 Large-v3 |
| 低资源语言 | 在 Medium 或 Turbo 上用 LoRA 微调 |
| 流式（2 秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（基于 CTranslate2）是 2026 年最快的 CPU+GPU 推理运行时——比原生实现快 4×，输出相同。

## 到 2026 年仍然会遇到的坑

- **在静音上产生幻觉。** Whisper 的训练语料包括了字幕，含有 “Thanks for watching!”、“Subscribe!”、歌词等。在调用前始终做 VAD 门控。
- **`condition_on_previous_text` 级联。** 一次幻觉会污染后续窗口。除非需要跨块流畅性，否则将其设为 False。
- **短片段填充。** 将 2 秒音频填充到 30 秒可能在尾部静音处产生幻觉。使用 `pad=False` 或进行 VAD 门控。
- **错误的梅尔统计。** 使用 librosa 的梅尔谱而非 Whisper 的会导致近乎随机的输出。请使用 `whisper.audio.log_mel_spectrogram`。

## 发布

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计一个 Whisper 微调或推理流水线。

## 练习

1. 简单：运行 `code/main.py`。它会对 Whisper 风格的提示词做分词，计算解码形状预算，并打印 10 分钟音频的分块计划。
2. 中等：安装 `faster-whisper`，转录一段 10 分钟的播客，将 WER 与人工转录对比。尝试 `language="auto"` 与 强制 `language="en"`。
3. 困难：使用 HF `datasets`，挑选一个 Whisper 表现差的语言（例如乌尔都语），在 Medium 上用 LoRA 在 2 小时数据上微调 2 个 epoch，报告 WER 变化。

## 术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|---------|
| 30-sec window | Whisper 的限制 | 强制的输入上限；更长音频需分块。 |
| SOT | Start-of-transcript | `<\|startoftranscript\|>` 启动解码器提示。 |
| Timestamps token | 时间对齐 | 每 0.02 s 的偏移在 51k 词表中是一个特殊 token。 |
| Turbo | 快速变体 | 4 层解码器，快 8×，WER 回退 <1%。 |
| WhisperX | 长音频包装器 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA fine-tune | 高效微调 | 在注意力中加入低秩适配器；训练约 0.3% 的参数。 |
| Hallucination | 静默失败 | Whisper 从噪声/静音中生成流畅英语。 |

## 进一步阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 原始架构与训练配方。  
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4 层解码器，8× 加速。  
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 面向长音频的词级对齐与说话人分离。  
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — 基于 CTranslate2 的实现，快 4×。  
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — 标准的 LoRA / 全量微调入门教程。