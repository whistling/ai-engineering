# 文本处理 — 分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理是桥梁。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 2 · 14（朴素贝叶斯）  
**Time:** ~45 分钟

## 问题

模型不能直接“读”到 The cats were running. 它读取的是整数。

每个 NLP 系统的前端都会遇到同样的三个问题。一个词从哪儿开始？词的词根是什么？当把 `run`、`running`、`ran` 视为同一事物对任务有利时该如何处理，何时又应把它们区分开。

如果分词搞错，模型就会从垃圾中学习。如果你的分词器把 `don't` 视为一个 token 而把 `do n't` 视为两个，训练分布就会分裂。如果你的词干提取器把 `organization` 和 `organ` 均归为同一词干，主题建模就会失败。如果词形还原器需要词性信息但你没有传入，动词会被当作名词处理。

本课从零构建三步预处理流程，然后展示 NLTK 和 spaCy 的实现，让你看到各自的权衡。

## 概念

三种操作。每种都有自己的职责和失败模式。

**分词（Tokenization）** 将字符串拆分为 tokens。“Token” 故意保持模糊，因为合适的粒度取决于任务。经典 NLP 常用词级分词；Transformer 常用子词；无空白语言常用字符级。

**词干提取（Stemming）** 用规则裁剪后缀。快速、激进、简单。`running -> run`。`organization -> organ`。后者就是失败模式。

**词形还原（Lemmatization）** 使用语法知识把词还原为词典形式。慢、准确，需要查表或形态分析器。`ran -> run`（需要知道 ran 是 run 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：当速度重要且能容忍噪声时使用词干（如搜索索引、粗略分类）。当语义重要时使用词形还原（如问答、语义检索、任何用户可读的场景）。

```figure
edit-distance
```

## 构建

### 步骤 1：一个基于正则的词级分词器

最简单且有用的分词器是按非字母数字字符拆分，同时把标点作为单独 token 保留。不是完美也不是最终方案，但一行代码就能跑起来。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三种模式按优先顺序匹配。带可选内撇号的词（`don't`、`it's`）。纯数字。任意单个非空白非字母数字字符作为独立 token（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式。`3pm` 被拆成 `['3', 'pm']`，因为我们在字母序列和数字序列之间切换。这对大多数任务已经足够。URL、邮箱、话题标签都会被破坏。用于生产时，请在通用模式之前添加针对这些场景的专门模式。

### 步骤 2：Porter 词干（仅第 1a 步）

完整的 Porter 算法有五个阶段的规则。仅实现第 1a 步可以覆盖最常见的英语后缀并讲清模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

按自上而下读取规则。`ies -> i` 规则就是为什么 `ponies -> poni` 而不是 `pony`。真正的 Porter 在 step 1b 会修正这个问题。规则是竞争关系——先出现的规则胜出。顺序往往比单条规则本身更重要。

### 步骤 3：基于查表的词形还原器

真正的词形还原需要形态学知识。可教学的可行版本使用一个小型的 lemma 表和回退策略。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个例子是关键的教学点。`watched` 不在我们的表中，而我们的回退策略仅处理 `ing`。真实的词形还原会覆盖 `ed`、不规则动词、比较级形容词、以及带发音变化的复数（如 `children -> child`）。这就是生产系统通常使用 WordNet、spaCy 的 morphologizer 或完整形态分析器的原因。

### 步骤 4：把它们串成流水线

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺少的部分是一个 POS 标注器。Phase 5 · 07（POS 标注）将构建一个。现在，默认把所有词当作 `NOUN` 并说明这一限制。

## 使用它

NLTK 和 spaCy 都包含生产级实现。各自只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理缩写、Unicode、以及你正则忽略的边缘情况。`PorterStemmer` 会运行完整的五个阶段。`WordNetLemmatizer` 需要把 NLTK 的 Penn Treebank 标注体系转换为 WordNet 的缩写集合。上面那段翻译逻辑是大多数教程跳过的重要部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整个流水线隐藏在 `nlp(text)` 后面。分词、词性标注和词形还原都会运行。相较于 NLTK，在大规模场景下更快、开箱更准确。权衡是你不容易单独替换某个组件。

### 何时选哪个

| Situation | Pick |
|-----------|------|
| 教学、研究、需要替换组件 | NLTK |
| 生产、多语言、对速度有要求 | spaCy |
| Transformer 管道（你无论如何都会用模型自带的 tokenizer） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 两个少有人提及的失败模式

大多数教程讲算法然后就停了。两个问题会真正咬到生产预处理流水线，但几乎从不被覆盖。

**可复现性漂移。** NLTK 和 spaCy 会在不同版本间改变分词和词形还原器的行为。spaCy 2.x 里产生 `['do', "n't"]` 的版本，可能在 3.x 里变成 `["don't"]`。你的模型在一种分布上训练，推理时运行在另一种分布上。准确率会悄然下降且没人知道原因。在 `requirements.txt` 中固定库版本。写一个预处理回归测试，锁定 20 个示例句子的期望分词。在每次升级时运行它。

**训练 / 推理不匹配。** 在训练时使用激进预处理（小写、停用词移除、词干提取），上线时对原始用户输入不做同样处理，性能会暴跌。这是生产 NLP 中最常见的失败。如果在训练时做了预处理，就必须在推理时运行完全相同的函数。把预处理作为模型包内的函数一起发布，而不是作为服务团队会改写的 notebook cell。

## 部署建议（Ship It）

一个可复用的提示，帮助工程师在不读三本教科书的情况下选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

你为经典 NLP 预处理提供建议。给定一个任务描述，你输出：

1. 分词选择（regex、NLTK word_tokenize、spaCy，或 transformer tokenizer）。说明原因。
2. 是否使用词干提取、词形还原、两者都用或两者都不用。说明原因。
3. 具体的库调用。列出函数名。如果涉及 NLTK，请写出 POS 标注到 WordNet 的转换方法。
4. 一个用户应当测试的失败模式。

拒绝在用户可见文本场景中推荐词干提取。拒绝在没有词性信息的情况下推荐词形还原。标注非英语输入需要不同的流水线。
```

## 练习

1. 简单。扩展 `tokenize` 以把 URL 作为单个 token 保留。测试：`tokenize("Visit https://example.com today.")` 应该产出一个 URL token。
2. 中等。实现 Porter 的 step 1b。如果一个词包含元音并以 `ed` 或 `ing` 结尾，则去掉该后缀。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. 困难。构建一个词形还原器：优先用 WordNet 查表，找不到条目时回退到你的 Porter 词干。用带标注的语料在准确率上对比纯 WordNet 和纯 Porter，做出衡量。

## 术语表

| Term | 人们常说的 | 实际含义 |
|------|-----------|---------|
| Token | 一个词 | 模型消费的单位。可以是词、子词、字符或字节。 |
| Stem | 词的根 | 基于规则的后缀裁剪结果。不一定是一个真实单词。 |
| Lemma | 词典形式 | 你会去查的形式。正确计算需要语法上下文。 |
| POS tag | 词性 | 像 NOUN、VERB、ADJ 的类别。准确词形还原需要它。 |
| Morphology | 词形规则 | 一个词根据时态、数、格如何变化。词形还原依赖于它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，仍然是最清晰的解释。  
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实流水线是如何连接的。  
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边缘情况。