# Chunking Strategies, Compared

> Chunking 决定了检索器能返回的所有内容。边界定得不好，下游没有任何嵌入模型、重排序器或 LLM 能修复由此产生的损害。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 11 lessons 04（嵌入）, 06（RAG）, 07（高级 RAG）；Phase 19 Track B foundations（lessons 20-29）  
**Time:** ~90 分钟

## 学习目标
- 从头实现五种分块（chunking）策略：固定窗口（fixed-window）、基于句子、递归拆分（recursive-split）、语义聚类（semantic clustering）以及基于结构的 Markdown 标题分割。
- 在带有金标准答案跨度（gold-labeled answer spans）的测试语料上测量 recall@k，并解释为何在散文和技术文档上获胜的策略不同。
- 读取分块长度分布并识别每种策略引入的失败模式：孤立句子、符号中间被切割、仅包含标题的块、语义漂移。
- 在不运行基准的情况下，通过检查三个属性来为新语料挑选默认策略：文档类型、平均段落长度，以及格式是否携带显式结构。

## 问题

每个 RAG 管道都以将源文档切分为嵌入模型能够处理的小片段为起点，同时要保证每个片段包含一个自洽的想法。切割位置的选择不是一个超参数；它是检索器能返回内容的上限。

一个询问 “预算中止阈值（budget abort threshold）是什么样子” 的查询只有在包含该阈值的 chunk 可被检索到时才可能成功。如果固定窗口切割把阈值值从周围上下文中切掉，嵌入会移动到不同的簇，BM25 分数下降，重排序器看到噪声，LLM 生成的答案就会出错。2024 年的论文 “LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs” 仅从分块选择上就测得了检索召回率 35 个百分点的绝对波动。2025 年关于上下文分块标题（contextual chunk headers）的后续工作缩小了差距但未能完全消除。

本课并排构建五种策略，在带有金标准答案跨度的测试语料上运行它们，并让你自己阅读召回率数字。

## 概念

```mermaid
flowchart LR
  Doc[源文档] --> S1[固定窗口]
  Doc --> S2[基于句子]
  Doc --> S3[递归拆分]
  Doc --> S4[语义聚类]
  Doc --> S5[结构化 Markdown]
  S1 --> Chunks1[块]
  S2 --> Chunks2[块]
  S3 --> Chunks3[块]
  S4 --> Chunks4[块]
  S5 --> Chunks5[块]
  Chunks1 --> Index[嵌入索引]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k 与 金标准跨度]
```

### 固定窗口（Fixed-window）

蛮力基线。按每 N 个字符切割。可选地使用重叠，这样在位置 N 被切开的句子可以完整出现在从位置 N - overlap 开始的块中。快速、确定性强，但边界处理很差。把它当作对照，而不是默认选择。

### 基于句子（Sentence）

用正则或简单状态机在句子边界处拆分。把一个或多个句子打包到目标字符预算范围内。不会在单词中间切断。但仍然会在段落或节中间切割。许多早期 RAG 管道的默认选择，对于没有其他结构的散文是一个合理选择。

### 递归拆分（Recursive split）

由 2023 年代库普及的层次化策略。先尝试在最强的分隔符处切分（双换行、段落），失败则回退到下一个（单换行），然后是句子，最后是字符。当 chunk 符合预算时递归终止。对于结构不一致的文档很强，因为它能够根据文档各区域自适应。

### 语义聚类（Semantic clustering）

对每个句子进行嵌入。对连续句子按主题中心向量进行聚类。当到当前中心向量的相似度降到阈值以下时切割。边界反映语义而非字符。构建更慢并依赖嵌入模型，但对段落内部切换主题的文档更有韧性。

### 结构化 Markdown 标题（Structural markdown headers）

对于携带显式结构的文档（Markdown、reStructuredText、RFC 风格编号节），在标题边界处切割。每个 chunk 包含标题及其下面直到下一个相同或更高级别标题的全部内容。每个主题的 chunk 最小，但仅在语料格式规范良好时可用。

### recall@k 如何衡量边界选择

一个金标准标注的查询包含答案跨度在源文档中的确切字符偏移量。分块后，你要问：检索器返回的前 k 个 chunk 中是否有任何一个与金标准跨度有重叠？如果有，该查询的 recall@k 为 1；如果没有，则为 0。对查询集合取平均。对每种策略运行相同评估，差异会显示出哪种边界策略在你的语料上更耐受。

## 实现

`code/main.py` 实现了：

- `fixed_window(text, size, overlap)` - 基线方法。  
- `sentence_chunks(text, target)` - 简单句子打包器。  
- `recursive_split(text, separators, target)` - 层次递归。  
- `semantic_chunks(text, similarity_threshold)` - 基于质心的聚类，使用确定性模拟嵌入。  
- `structural_markdown(text)` - 基于标题的拆分器。  
- `mock_embed(text, dim)` - 基于哈希的嵌入，以便循环可以离线运行。  
- `DenseIndex` - 与 Phase 19 Track B 的混合检索课中使用的形状相同。  
- `eval_recall(strategy, corpus, queries, k)` - 比较循环。  
- 一个运行每种策略在测试语料上并打印 recall@k 表格的 `main()`。

运行它：

```bash
python3 code/main.py
```

输出是一个小表格，每行对应一种策略，每列对应不同的 k。句子分块在结构化测试语料上表现不好。结构化 Markdown 在 Markdown 测试语料上获胜。递归拆分在混合语料上表现稳健，因为递归能自适应。语义聚类在无有用结构线索的散文语料上获胜。

## 表格不会隐藏的失败模式

**孤立句子（Orphan sentences）。** 基于句子的打包会产生缺失主题句的块。嵌入因而指向错误的簇。

**符号中间被切割（Mid-symbol cuts）。** 固定窗口在代码或 YAML 中会把标识符切成两半。两半的嵌入成为噪声。

**仅含标题的块（Header-only chunks）。** 结构化 Markdown 可能会输出只包含 `## Title` 的块。过滤这些块或把下一个块的第一段附加上来。

**语义漂移（Semantic drift）。** 语义聚类在语料整体主题一致时会过度切割。一个 5000 字符的块会容纳很多具体答案，导致嵌入变得模糊。将语义方法与硬字符上限结合使用。

**陈旧的嵌入（Stale embeddings）。** 语义聚类使用嵌入模型。如果你更换模型，分块也会随之改变。将分块所用的模型与检索模型分开固定，或在一起重建索引。

## 在不运行基准的情况下选择默认策略

三个属性决定了新语料的默认分块器。

| Property | Value | Default |
|----------|-------|---------|
| Document type | Prose with no structure | 递归拆分，target 800 |
| Document type | Markdown / RFC / API docs | 结构化 Markdown |
| Document type | Code | 基于 AST（超出本课范围；见 Phase 19 lesson 02） |
| Paragraph length | Long, single topic | 基于句子，target 500 |
| Paragraph length | Short, mixed topics | 语义聚类，threshold 0.6 |

如果不确定，选择递归拆分。它是最稳健的单一策略基线。

## 使用建议

生产模式：

- 在发布新管道前运行评估；不要盲信库的默认策略。
- 每当你更改嵌入模型或语料构成时重新运行评估；获胜策略依赖于语料。
- 在每个 chunk 的元数据中持久化策略名称，以便以后能归因回归。

## 部署（Ship It）

Track F 的端到端 RAG 系统（lesson 69）使用了在此处选择的分块器作为第一阶段。lesson 68 中的评估工具会读取与本课中 `eval_recall` 返回形状相同的 recall@k。选择在你的语料上获胜的策略并向前传递。

## 练习

1. 增加第六种策略：使用 `tiktoken` 的 token-window 而不是字符计数。与相同测试语料上的固定窗口比较。  
2. 在散文测试语料中注入 30% 的代码块。重新运行表格。解释为什么除了结构化 Markdown 外每种策略的召回都下降。  
3. 用你项目的真实提供方的嵌入替换确定性嵌入。度量语义聚类的召回差异。报告策略之间的差距是扩大了还是缩小了。  
4. 为每个 chunk 增加一个 `summary` 字段：一句话的质心描述。把 summary 追加到 chunk 正文后重新运行评估。测量召回提升。

## 关键术语

| Term | 大众说法 | 实际含义 |
|------|---------|----------|
| Recall@k | “我们找对了 chunk 吗？” | 在 top-k chunk 中任意一个与金标准答案跨度重叠的查询所占比例 |
| Chunk overlap | “滑动窗口” | 在下一个块中重新包含上一个块的最后 N 个字符 |
| Structural splitter | “感知标题的分块” | 在 H1/H2/H3 边界处切割；标题文本是 chunk 的一部分 |
| Semantic chunker | “感知主题的分块” | 对句子进行嵌入，按质心相似度聚类，基于漂移切割 |
| Centroid drift | “主题切换” | 运行均值与下一句之间的余弦相似度低于阈值 |

## 延伸阅读

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (arXiv 2406.15319)](https://arxiv.org/abs/2406.15319)  
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)  
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)  
- Phase 11 lesson 06 - RAG fundamentals  
- Phase 11 lesson 07 - advanced RAG  
- Phase 19 lesson 65 - hybrid retrieval that ranks the chunks produced here  
- Phase 19 lesson 68 - the eval harness that scores the strategy choice in production