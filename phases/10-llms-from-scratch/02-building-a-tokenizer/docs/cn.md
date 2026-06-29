# 从零构建一个分词器

> Lesson 01 给了你一个玩具。本课给你一件武器。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 10，Lesson 01（Tokenizers：BPE、WordPiece、SentencePiece）  
**Time:** ~90 分钟

## 学习目标

- 构建一个生产级的 BPE 分词器，处理 Unicode、空白规范化和特殊 tokens  
- 实现基于字节的回退，使分词器能够对任意输入（包括 emoji、CJK、以及代码）进行编码而不产生未知 token  
- 添加预分词的正则模式，在应用 BPE 合并之前在单词边界处拆分文本  
- 在语料上训练自定义分词器，并在多语种文本上将其压缩比与 tiktoken 进行比较

## 问题描述

你在 Lesson 01 中实现的 BPE 分词器对英文文本有效。现在扔给它日语。或者 emoji。或者混合了制表符和空格的 Python 代码。

它会崩溃。

并不是因为 BPE 是错的——而是因为实现不够完整。一个生产级的分词器需要处理任何编码下的原始字节，在拆分前进行 Unicode 规范化，管理永不参与合并的特殊 tokens，将预分词与子词拆分串联，并且这一切要快到不会成为处理 15 万亿 token 的训练管线的瓶颈。

GPT-2 的分词表有 50,257 个 token。Llama 3 有 128,256。GPT-4 大约有 100,000。这可不是小数目。那些词汇表背后的合并表是在数百 GB 文本上训练出来的，周边的机制——规范化、预分词、特殊 token 注入、聊天模板格式化——正是把只能处理 “hello world” 的分词器和能处理整个互联网的分词器区分开的地方。

你将构建这些机制。

## 概念

### 完整流水线

一个生产级分词器不是单一算法。它是由五个阶段组成的流水线，每个阶段解决不同的问题。

```mermaid
graph LR
    A[原始文本] --> B[规范化]
    B --> C[预分词]
    C --> D[BPE 合并]
    D --> E[特殊 Tokens]
    E --> F[Token ID]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有特定职责：

| 阶段 | 它的作用 | 为什么重要 |
|------|---------|------------|
| Normalize（规范化） | NFKC Unicode，可选小写化，可选去重音 | “fi” 连字（U+FB01）变成 “fi”（两个字符）。否则相同单词会被分到不同 tokens。 |
| Pre-Tokenize（预分词） | 在 BPE 之前把文本拆成块 | 防止 BPE 跨单词边界合并。比如 "the cat" 永远不应该产生 token "e c"。 |
| BPE Merge（BPE 合并） | 对字节序列应用学习到的合并规则 | 核心压缩机制。把原始字节变成子词 token。 |
| Special Tokens（特殊 Tokens） | 注入 [BOS]、[EOS]、[PAD]、聊天模板标记 | 这些 tokens 有固定 ID，永不参与 BPE 合并。模型需要它们来表示结构。 |
| ID Mapping（ID 映射） | 把 token 字符串映射为整数 ID | 模型看到的是整数，而不是字符串。 |

### 基于字节的 BPE

Lesson 01 的分词器是基于 UTF-8 字节运作的。这是正确的选择。但我们跳过了一个重要问题：当那些字节不是有效 UTF-8 时会发生什么？

基于字节的 BPE 通过把每个可能的字节值（0-255）都视为有效 token 来解决这个问题。你的基础词表正好有 256 个条目。任何文件——文本、二进制、损坏的——都可以被分词而不会产生未知 token。

GPT-2 添加了一个技巧：把每个字节映射到一个可打印的 Unicode 字符，这样词表对人类仍然可读。字节 0x20（空格）在他们的映射里变成了字符 "G"。这纯属表面化处理，算法本身并不在意。

真正的强大之处在于：基于字节的 BPE 处理地球上所有语言。中文字符每个字符通常是 3 个 UTF-8 字节。日语可能是 3-4 字节。阿拉伯语、天城文、emoji —— 都只是字节序列。BPE 算法在这些字节序列上寻找模式，方式和它在英文 ASCII 字节上寻找模式完全相同。

### 预分词

在 BPE 处理文本之前，你需要先把文本拆成块。这可以防止合并算法生成跨单词的 token。

GPT-2 使用的正则模式如下：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式会在缩写（"don't" 变为 "don" + "'t"）、可选前置空格的单词、数字、标点和空白处进行拆分。前导空格会保留并附着到单词上——因此 "the cat" 会变成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，完全跳过正则。它把原始字节流视为一长串序列，让 BPE 算法自己决定边界。这更简单，但也给了 BPE 更大的自由去创建跨词 token。

选择很重要。GPT-2 的正则防止了分词器学会把一个单词末尾的 "the" 和下一个单词开头的 "the" 合并在一起。SentencePiece 则允许这种合并，有时会产生更高效的压缩但 token 可解释性降低。

### 特殊 Tokens

每个生产级分词器都会为结构化标记保留 token ID：

| Token | 目的 | 使用者 |
|-------|------|--------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3、GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 批对齐填充 | BERT、T5 |
| `[UNK]` | 未知 token（基于字节的 BPE 会消除它） | BERT、WordPiece |
| `<|im_start|>` | 聊天消息边界开始 | ChatGPT、Qwen |
| `<|im_end|>` | 聊天消息边界结束 | ChatGPT、Qwen |
| `<|user|>` | 用户回合标记 | Llama 3 |
| `<|assistant|>` | 助手回合标记 | Llama 3 |

特殊 tokens 永远不会被 BPE 拆分。它们在合并算法运行之前做精确匹配，用固定 ID 替换，周围的文本则按常规被分词。

### 聊天模板

这是大多数人感到困惑且实现容易出错的地方。

当你向聊天模型发送消息时，API 接受的是消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型并不会看到 JSON。它看到的是扁平的 token 序列。聊天模板把消息转换为这个扁平序列，使用特殊 tokens。每个模型的格式都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你很有帮助。<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
你很有帮助。<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

如果模板搞错了，模型会输出垃圾。模型是在一种精确格式上训练的。任何偏离——缺少换行、tokens 顺序错误、多了空格——都会把输入推到训练分布之外。

### 速度

Python 对于生产级分词太慢了。

tiktoken（OpenAI）用 Rust 编写并提供 Python 绑定。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。这些比纯 Python 实现快 10-100 倍。

举个例子：如果为 Llama 3 预训练要分词 15 万亿 token，在 1,000,000 token/秒（快速 Python）下需要 174 天。在 100,000,000 token/秒（Rust）下只需要 1.7 天。

这里我们用 Python 来理解算法。在生产中，你会使用编译语言实现并仅在 Python 层留一个封装。

```figure
weight-tying
```

## 构建它

### 第 1 步：基于字节的编码

基础。把任意字符串转换为字节序列，为展示目的把每个字节映射到一个可打印字符，并实现反向过程。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

在多语种文本上测试字节数量：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 是 5 个字节。"你好" 是 6 个字节（每个字符 3 个字节）。火焰 emoji 是 4 个字节。基于字节的分词器不关心语言，字节就是字节。

### 第 2 步：用正则实现预分词

使用 GPT-2 的正则模式把文本拆成块。每个块由 BPE 独立分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库的 `re` 模块不支持这些，所以我们退回到 ASCII 字符类。对于生产级的多语种分词器，请安装 `regex`。

试试：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# 输出: [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格会保留并附着到单词上。缩写在撇号处拆分。标点成为独立的块。BPE 不会跨这些边界合并 token。

### 第 3 步：对字节序列应用 BPE

这是 Lesson 01 的核心算法，不过现在对预分词的块分别独立运行。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 第 4 步：特殊 Token 处理

特殊 tokens 需要精确匹配和固定 ID。它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 第 5 步：完整的分词器类

把一切串联起来：规范化、对特殊 tokens 拆分、预分词、BPE 合并、映射到 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 第 6 步：多语种测试

真正的测试。对英文、中文、emoji 和代码都试一试。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字符每个通常产生 3 个字节。emoji 产生 4 个字节。没有任意一种会让分词器崩溃，也不会产生未知 token。这就是基于字节的 BPE 的力量。

## 使用它

### 对比真实分词器

加载来自 Llama 3、GPT-4 和 Mistral 的实际分词器。看看它们如何处理相同的多语段落。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到相同文本在不同分词器上产生的 token 数不同。128K 词汇的 Llama 3 更积极地合并常见模式。GPT-4（约 100K）处于中间。Mistral（32K）会产生更多 token，但其嵌入层更小。

权衡总是一样的：更大的词表意味着更短的序列，但参数更多。

## 交付

本课会生成一个用于构建和调试生产分词器的 prompt，见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. 简单：添加一个 `get_token_bytes(id)` 方法，展示任意 token ID 的原始字节。用它来检查你最常见的合并 token 实际代表什么。  
2. 中等：实现类似 Llama 的预分词器，对空白和数字进行拆分但保留前导空格。将其与 GPT-2 正则方法在同一语料上的词汇表进行比较。  
3. 困难：添加一个聊天模板方法，接受一个 `{"role": ..., "content": ...}` 消息列表并生成 Llama 3 聊天格式的正确 token 序列。与 HuggingFace 的实现进行测试比较。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|--------|---------|
| 基于字节的 BPE (Byte-level BPE) | “在字节上工作的分词器” | 基于字节的 BPE，基础词表为 256 个字节值——能处理任意输入而不会产生未知 tokens |
| 预分词 (Pre-tokenization) | “在 BPE 之前拆分” | 使用正则或规则在 BPE 之前拆分，防止 BPE 跨单词边界合并 |
| NFKC 规范化 | “Unicode 清理” | 先做规范分解再做兼容性组合——“fi” 连字变为 “fi”，全角 “A” 变为 “A” |
| 聊天模板 (Chat template) | “消息如何变成 tokens” | 把角色/内容消息列表转换为扁平 token 序列的精确定义——模型依赖训练时使用的特定格式 |
| 特殊 tokens (Special tokens) | “控制 tokens” | 绕过 BPE 的保留 token ID —— [BOS]、[EOS]、[PAD]、聊天标记等，合并前做精确匹配 |
| Fertility（生育率） | “每词的 token 数” | 输出 token 与输入单词的比率——GPT-4 对英语约为 1.3，韩语为 2-3，值越高表示上下文浪费越多 |
| tiktoken | “OpenAI 分词器” | 用 Rust 实现的 BPE 并提供 Python 绑定——比纯 Python 快 10-100 倍 |
| 合并表 (Merge table) | “词汇表” | 在训练中学到的按顺序的字节对合并列表——这就是分词器学到的知识 |

## 延伸阅读

- [OpenAI tiktoken 源码](https://github.com/openai/tiktoken) —— GPT-3.5/4 使用的 Rust BPE 实现  
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) —— 支持 BPE、WordPiece、Unigram 的 Rust 分词库  
- [Llama 3 论文 (Meta, 2024)](https://arxiv.org/abs/2407.21783) —— 有关 128K 词汇表和分词器训练的细节  
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) —— 与语言无关的分词方法  
- [GPT-2 分词器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py) —— 原始的字节到 Unicode 映射