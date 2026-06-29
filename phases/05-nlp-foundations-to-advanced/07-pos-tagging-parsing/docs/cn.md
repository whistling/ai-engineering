# 词性标注与句法解析

> 一度语法不再时髦。然后每个 LLM 流水线都需要验证结构化抽取，语法又回来了。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 5 · 01 (文本处理), Phase 2 · 14 (朴素贝叶斯)  
**Time:** ~45 分钟

## 问题描述

第01课承诺词形还原需要词性标签。如果不知道 `running` 是动词，词形还原器就无法还原为 `run`。如果不知道 `better` 是形容词，就无法还原为 `good`。

这个承诺背后隐藏了一个完整的子领域。词性标注为每个词元分配语法类别。句法解析恢复句子的树形结构：哪个词修饰哪个词，哪个动词支配哪些论元。传统 NLP 花了二十年来不断完善这两项任务。然后深度学习把它们压缩成预训练 Transformer 之上的一个标注任务，研究界也转向了别处。

应用界则不同。每个结构化抽取流水线在底层仍然使用词性和依存树。LLM 生成的 JSON 会根据语法约束进行验证。问答系统利用依存解析分解查询。机器翻译质量评估会检查解析树的一致性。

值得了解。本课介绍常用的标注集、基线方法，以及你什么时候该停止从头实现而调用 spaCy。

## 概念

**词性标注（POS tagging）** 为每个词元标注一个语法类别。**Penn Treebank (PTB)** 标注集是英语的默认选择。36 个标签，对非专业读者看起来有些吹毛求疵的区分：`NN` 单数名词、`NNS` 复数名词、`NNP` 专有名词单数、`VBD` 动词过去式、`VBZ` 动词第三人称单数现在时，等等。**Universal Dependencies (UD)** 标注集更粗（17 个标签）且与语言无关；它已成为跨语言工作的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法解析（Syntactic parsing）** 生成一棵树。两种主要风格：

- **短语结构解析（Constituency parsing）**。名词短语、动词短语、介词短语互相嵌套。输出是一棵非终结类别（NP、VP、PP）为内部节点、单词为叶子的树。
- **依存解析（Dependency parsing）**。每个词都有一个它依赖的中心词，并带有语法关系标签。输出是一棵树，每条边都是一个（head, dependent, relation）三元组。

依存解析在 2010 年代获得了胜出，因为它在语言间更容易泛化，尤其是自由语序语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 实现

### 步骤 1：最常见标签基线（most-frequent-tag baseline）

最笨但有效的词性标注器。对每个词，预测它在训练集中出现最多的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料上，这个基线能达到约 85% 的准确率。虽不高，但这是任何认真模型不该低于的底线。

### 步骤 2：二元 HMM 标注器（bigram HMM tagger）

建模序列的联合概率：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两个表：转移概率（给定前一个标签的标签概率）和发射概率（给定标签的词概率）。用计数并加 Laplace 平滑估计。使用 Viterbi 解码（对标签格进行动态规划）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

在 Brown 上，二元 HMM 可达约 93% 的准确率。从 85% 到 93% 的提升主要来自转移概率——模型学会了 `DET NOUN` 常见而 `NOUN DET` 罕见。

### 步骤 3：现代标注器为何能超越它们

转移+发射概率是局部的。它们无法捕捉到同一个词在不同上下文中的词性差异，例如在 "I bought a saw" 中 `saw` 是名词，但在 "I saw the movie." 中是动词。带任意特征（词缀、词形、前后词、词本身）的 CRF 可达约 97%。BiLSTM-CRF 或基于 Transformer 的模型可达 98% 以上。

该任务的天花板由标注者之间的不一致率决定。人工标注者在 Penn Treebank 上约 97% 一致。超过 98% 的模型很可能是在对测试集过拟合。

### 步骤 4：依存解析概述

从头实现完整的依存解析超出本课范围；规范教材见 Jurafsky 和 Martin。要了解的两大经典家族：

- **基于移进-归约的（Transition-based）** 解析器（arc-eager、arc-standard）类似 shift-reduce 解析器：它们读取词元，将其移入栈中，并应用 reduce 操作来创建弧。贪心解码速度快。经典实现是 MaltParser。现代的神经版本为 Chen 和 Manning 提出的转移式解析器。
- **基于图的（Graph-based）** 解析器（Eisner 算法、Dozat-Manning 的双仿射）对每一条可能的头-依赖边打分，然后选择最大生成树。速度较慢但更准确。

在大多数应用场景下，直接调用 spaCy 即可：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从 `dep` 列自下而上读，句子的语法结构就显现出来。

## 使用场景

每个生产级 NLP 库都将词性和依存解析作为标准流水线的一部分提供。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速、准确，集成了分词、命名实体识别和词形还原。使用 `token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP (stanza)**。Stanford 的 CoreNLP 后继项目。支持 60+ 种语言并接近最先进水平。
- **trankit**。基于 Transformer，UD 准确率优秀。
- **NLTK**。`pos_tag`。可用但较慢、较老。适合教学用途。

### 到 2026 年这仍然重要的地方

- **词形还原（Lemmatization）**。第01课需要词性来正确还原词形。始终如此。
- **从 LLM 输出进行结构化抽取。** 验证生成句子是否满足语法约束（例如主谓一致、必需的修饰成分）。
- **基于方面的情感分析。** 依存解析告诉你哪个形容词修饰哪个名词。
- **查询理解。** “movies directed by Wes Anderson starring Bill Murray” 可以通过解析分解成结构化约束。
- **跨语言迁移。** UD 标签和依存关系与语言无关，使对新语言的零样本结构化分析成为可能。
- **低算力流水线。** 如果无法部署 Transformer，词性 + 依存解析 + 地名表（gazetteer）依然能取得出人意料的效果。

## 发布建议（Ship It）

将以下内容保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: 设计一个用于下游 NLP 任务的经典 POS + 依存流水线。
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

给定一个下游任务（信息抽取、重写验证、查询分解、词形还原），你需要输出：

1. 使用的标注集。仅限英文的遗留流水线使用 Penn Treebank；多语言或跨语言使用 Universal Dependencies。
2. 库选择。生产环境大多使用 spaCy，学术级多语言使用 stanza，追求最高 UD 准确率可选 trankit。写明具体的模型 ID。
3. 集成模式。展示调用库并消费所需属性（`.pos_`、`.dep_`、`.head`）的 3-5 行示例代码。
4. 需要测试的故障模式。名词-动词歧义（`saw`、`book`、`can`）和介词短语附着歧义（PP-attachment）是经典陷阱。抽样 20 个输出进行人工检查。

拒绝建议自行实现解析器。从头构建解析器是研究项目，而不是应用任务。标记任何在消费 POS 标签时不处理大小写变体的流水线为脆弱。
```

## 练习

1. 简单题：在一个小的带标注语料上（例如 NLTK 的 Brown 子集）使用最常见标签基线，测量在留出句子上的准确率。验证约 85% 的结果。
2. 中等题：训练上面的二元 HMM 并报告每个标签的精确率/召回率。HMM 最常混淆哪些标签？
3. 困难题：使用 spaCy 的依存解析从 1000 句样本中抽取主谓宾三元组。在 50 个手工标注的三元组上评估。记录抽取失败的情况（通常是被动语态、并列结构和省略主语）。

## 关键词

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| POS tag | 词的类型 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准标注集 | 英语专用。对动词时态和名词数有精细区分。 |
| Universal Dependencies | 多语言标注集 | 比 PTB 更粗；与语言无关；跨语言工作的默认。 |
| Dependency parse | 句子树 | 每个词有一个头词，每条边有一个语法关系。 |
| Viterbi | 动态规划 | 在已知发射和转移的情况下，找到概率最高的标签序列。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 关于词性和解析的权威教材章节。  
- [Universal Dependencies project](https://universaldependencies.org/) — 用于每个多语言解析器的跨语言标注集和语料库集合。  
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 暴露的每个属性的实用参考。  
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经解析器带入主流的论文。