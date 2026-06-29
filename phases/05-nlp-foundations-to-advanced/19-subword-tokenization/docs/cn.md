# 子词标记化 — BPE、WordPiece、Unigram、SentencePiece

> 基于单词的分词器在遇到未见词时失效。基于字符的分词器会将序列长度拉长。子词标记化折中二者。每个现代大规模语言模型都会采用其中之一。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 5 · 01（文本处理），Phase 5 · 04（GloVe / FastText / Subword）  
**Time:** ~60 分钟

## 问题

你的词表有 50,000 个词。用户输入了 "untokenizable"。你的分词器返回 `[UNK]`。模型现在对该词没有任何信号。更糟的是：语料库中第 90 个百分位的文档包含 40 个罕见词，这意味着每篇文档丢失 40 位信息。

子词标记化解决了这个问题。常见词保持为单个 token。罕见词分解为有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据覆盖所有情况，因为任何字符串最终都可以表示为字节序列。

到 2026 年，每个前沿 LLM 都基于三种算法之一（BPE、Unigram、WordPiece），并封装在三种库之一（tiktoken、SentencePiece、HF Tokenizers）中。你无法在不选择其一的情况下部署语言模型。

## 概念

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE（Byte-Pair Encoding）**。从字符级词表开始。统计每一个相邻对。将出现频率最高的成对字符合并为新 token。重复直到达到目标词表大小。主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级 BPE（Byte-level BPE）**。相同算法但在原始字节（256 个基本 token）上运行，而不是 Unicode 字符。保证零 `[UNK]` —— 任意字节序列都可编码。GPT-2 使用 50,257 个 token（256 字节 + 50,000 次合并 + 1 个特殊）。

**Unigram。** 从一个巨大的候选词表开始。为每个 token 分配一个 unigram 概率。迭代剪枝那些删除后最少降低语料对数似然的 token。推理时具有概率性：可以对分词结果采样（对数据增强有用，称为子词正则化）。被 T5、mBART、ALBERT、XLNet、Gemma 使用。

**WordPiece。** 合并时不是按原始频率，而是按最大化训练语料似然的成对合并。被 BERT、DistilBERT、ELECTRA 使用。

**SentencePiece vs tiktoken。** SentencePiece 是直接在原始 Unicode 文本上 *训练* 词表（BPE 或 Unigram）的库，将空格编码为 `▁`。tiktoken 是 OpenAI 的快速 *编码器*，用于对预构建词表进行编码；它不进行训练。

经验法则：

- **训练新词表：** 使用 SentencePiece（多语种、无需预分词）或 HF Tokenizers。
- **针对 GPT 词表的快速推理：** tiktoken（cl100k_base、o200k_base）。
- **两者都要：** HF Tokenizers —— 一个库，既能训练也能服务。

```figure
bpe-merge
```

## 构建它

### 步骤 1：从头实现 BPE

参见 `code/main.py`。主循环：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法编码的三个事实：`</w>` 标记词尾，因此 "low"（后缀）和 "lower"（前缀）保持区分。频率加权使高频对先被合并。合并列表是有序的 —— 推理时按训练顺序应用合并。

### 步骤 2：用学到的合并进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现为 O(n·|merges|)。生产实现（tiktoken、HF Tokenizers）使用合并-秩查找和优先队列，运行近线性时间。

### 步骤 3：实际使用 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 或 "unigram"
    character_coverage=0.9995, # 对 CJK 可设更低（例如英文用 ~0.9995，日文用 ~0.995）
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格被编码为 `▁`，`character_coverage` 控制保留罕见字符与映射为 `<unk>` 的激进程度。

### 步骤 4：用于 OpenAI 兼容词表的 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快（Rust 后端）。与 GPT-4/5 的 tokenization 在字节计数、费用估算、上下文窗口预算上完全匹配。

## 到 2026 年仍会出现的问题

- **分词器漂移（Tokenizer drift）。** 在词表 A 上训练，在词表 B 上部署。token ID 不同；模型输出垃圾。在 CI 中检查 `tokenizer.json` 的哈希。
- **空白歧义。** BPE 中 "hello" 与 " hello" 产生不同的 token。始终明确指定 `add_special_tokens` 和 `add_prefix_space`。
- **多语种训练不足。** 以英语为主的语料会导致非拉丁脚本被拆分为 5–10 倍的 token。同样的提示在日语/阿拉伯语上在 GPT-3.5 上成本会高 5–10 倍。o200k_base 部分缓解了这个问题。
- **表情符号拆分。** 单个表情符号可能占用 5 个 token。在预算上下文时检查 checkpoint 的表情符号处理。

## 使用建议

2026 年栈：

| Situation | Pick |
|-----------|------|
| 从头训练单语模型 | HF Tokenizers（BPE） |
| 训练多语种模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 提供与 OpenAI 兼容的 API | tiktoken（针对 GPT-4+ 使用 `o200k_base`） |
| 特定领域词表（代码、数学、蛋白质） | 在领域语料上训练自定义 BPE，并与基础词表合并 |
| 边缘推理、小模型 | Unigram（较小的词表效果更好） |

词表大小是一个随规模变化的决策，而不是常数。粗略启发式：小于 1B 参数用 32k；1–10B 用 50–100k；多语种/前沿模型用 200k+。

## 部署建议

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. 简单：在 `code/main.py` 的小语料上训练一个 500 次合并的 BPE。对三个留出词进行编码。有多少个恰好产生 1 个 token，多少个产生 >1 个 token？
2. 中等：在 100 条英文维基百科句子上比较 `cl100k_base`、`o200k_base` 与你用 vocab=32k 训练的 SentencePiece BPE 的 token 数。报告每个的压缩比。
3. 困难：使用相同语料分别训练 BPE、Unigram、WordPiece。测量在一个小型情感分类器上的下游准确率。选择是否能将 F1 值变动超过 1 个百分点？

## 关键词

| Term | 人们常说 | 实际含义 |
|------|---------|---------|
| BPE | Byte-Pair Encoding | 贪心地合并最频繁的字符对，直到达到目标词表大小。 |
| Byte-level BPE | No unknown tokens ever | 在原始 256 字节上运行的 BPE；GPT-2 / Llama 使用该方法。 |
| Unigram | Probabilistic tokenizer | 从一个大型候选集合中通过对数似然进行剪枝；T5、Gemma 使用。 |
| SentencePiece | The whitespace one | 在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁`。 |
| tiktoken | The fast one | OpenAI 的 Rust 支持的 BPE 编码器，用于预构建词表。无训练功能。 |
| Merge list | The magic numbers | 有序的 `(a, b) → ab` 合并列表；推理时按顺序应用。 |
| Character coverage | How rare is too rare? | 训练语料中 tokenizer 必须覆盖的字符比例；典型约为 0.9995。 |

## 进一步阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文。  
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文。  
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 该库的论文。  
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简明参考。  
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — 配方与编码表。