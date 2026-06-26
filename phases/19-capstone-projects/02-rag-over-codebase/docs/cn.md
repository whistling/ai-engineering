# Capstone 02 — 基于代码库的 RAG（跨仓语义搜索）

> 到 2026 年，每个严肃的工程组织都会运行一个能理解语义而不仅仅是字符串的内部代码搜索。Sourcegraph Amp、Cursor 的 codebase answers、Augment 的 enterprise graph、Aider 的 repomap、Pinterest 的内部 MCP —— 形态相同。摄取多个仓库，用 tree-sitter 解析，将函数和类级别的片段做嵌入，混合检索，重排序，并带引用地给出答案。本结业项目要求你构建一个能处理跨 10 个仓库共 200 万行代码，并能在每次 git push 时增量重建索引的系统。

**Type:** 结业项目  
**Languages:** Python（摄取）, TypeScript（API + UI）  
**Prerequisites:** Phase 5（NLP 基础）, Phase 7（transformers）, Phase 11（LLM 工程）, Phase 13（工具）, Phase 17（基础设施）  
**Phases exercised:** P5 · P7 · P11 · P13 · P17  
**Time:** 30 小时

## 问题

到 2026 年，每个前沿的编码代理都会随附一个代码库检索层，因为单靠上下文窗口无法解决跨仓问题。Claude 的 1M-token 上下文很有用；但它并不能消除排序检索的需求。对原始代码片段做朴素余弦搜索会在生成代码、单体仓库复制以及罕见符号的长尾上毒化结果。生产环境的答案是一个基于 AST 感知片段的混合检索（稠密 + BM25），配合重排序器，并以符号引用图作为后盾。

你需要通过对一个真实代码舰队建立索引来学习这些方法 —— 不是一个示例仓库 —— 并衡量 MRR@10、引用可信度和增量新鲜度。失败模式往往是基础设施问题：一个 10 万文件的单体仓库、一推改动半数文件、一个需要跨四个仓库才能正确回答的查询。

## 概念

一个 AST 感知的摄取管道使用 tree-sitter 解析每个文件，提取函数和类节点，并在节点边界处进行切块，而不是使用固定的 token 窗口。每个片段获得三种表示：稠密嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项、以及一段简短的自然语言摘要。摘要增加了第三种可检索模态 —— 用户问 “X 是如何被授权的”，摘要会提到 “authz”，即使代码中只有 `check_permission`。

检索是混合的。查询同时触发稠密和 BM25 检索，合并 top-k，然后将并集交给一个 cross-encoder 重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表传给长上下文合成器（Claude Sonnet 4.7，带提示缓存，或自托管的 Llama 3.3 70B），并要求按文件和行范围对每个声明提供引用。没有引用的答案会被后置过滤拒绝。

增量新鲜度是基础设施难题。Git push 触发 diff：哪些文件改了、哪些符号改了。只有受影响的片段需要重新嵌入。受影响的跨文件符号边（imports、方法调用）需要重新计算。索引保持一致，而不是在每次提交都重处理 200 万行代码。

## Architecture

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：tree-sitter，支持 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 作为回退
- 稀疏索引：Tantivy（Rust）+ BM25F，按符号名与主体做字段加权
- 向量数据库：Qdrant 1.12（支持混合检索），或对小于 5000 万向量的团队使用 pgvector + pgvectorscale
- 片段摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，使用提示缓存
- 重排序器：Cohere rerank-3 或 自托管的 bge-reranker-v2-gemma-2b
- 编排：LlamaIndex Workflows（用于摄取），LangGraph（用于查询代理）
- 合成器：Claude Sonnet 4.7（1M 上下文），带提示缓存
- 符号图：Neo4j（托管）或 kuzu（嵌入式），用于导入和调用边
- 可观测性：Langfuse 为每次检索与合成步骤产生 span

## 构建步骤

1. **摄取 walker。** 在每次 push hook 里遍历 git 历史，收集变更文件。对每个文件用 tree-sitter 解析，提取函数与类节点及其完整源码跨度。产生片段记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **片段摘要器。** 将片段批量发送给 Haiku 4.5，请求摘要，使用系统前置语的提示缓存。提示词： "Summarize this function in one sentence, naming its public contract and side effects." 将摘要和片段一起存储。

3. **嵌入池。** 两条并行队列：稠密（Voyage-code-3 批量 128）和摘要（对摘要字符串也使用相同模型）。将向量写入 Qdrant，payload 包含 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名权重 4、符号主体权重 1、摘要权重 2。支持“查找名为 X 的函数”以及“查找实现了 X 的函数”两类查询。

5. **符号图。** 对于每个片段，记录边：imports（该文件使用了来自 repo Z 的符号 Y）、calls（此函数调用类 C 的方法 M）、继承。存储在 kuzu。查询时用于跨仓检索扩展。

6. **查询代理。** 使用 LangGraph 构建三节点流水线。`retrieve` 并行触发稠密 + BM25，按 (repo, path, symbol) 去重。`rerank` 对 top-50 运行 cross-encoder，并保留 top-10。`synth` 使用重排序后的片段上下文调用 Claude Sonnet 4.7，缓存系统提示，要求文件:行的引用。

7. **引用强制。** 解析模型输出；任何未带 `(repo/path:start-end)` 锚点的声明都会被标记为需重问或丢弃。只返回带引用的答案给用户。

8. **增量重建索引。** 每次 webhook 时计算符号级别 diff。只有文本变更的片段会被重新嵌入。只有 imports 发生变化的片段会重新计算符号边。指标示例：在 200 万 LOC 的舰队上，50 文件的 push 在 60 秒内完成重索引。

9. **评估。** 标注 100 个跨仓问题并给出金标文件:行答案。衡量 MRR@10、nDCG@10、引用可信度（可验证锚点占比）以及 p50/p99 延迟。

## 使用示例

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[检索]     12 chunks dense + 7 chunks bm25, 16 unique after dedup
[重排序]   top-5 kept (cohere rerank-3)
[合成]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
回答:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

（注：示例中保留了代码路径与符号名等内联代码项不变。）

## 交付

交付物为 `outputs/skill-codebase-rag.md`。给定一组仓库，它应能立起摄取管道、混合索引与查询代理，并对任意跨仓问题返回带引用的答案。评分标准：

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | 检索质量 | 在 100 个保留问题集上的 MRR@10 与 nDCG@10 |
| 20 | 引用可信度 | 答案中带可验证 file:line 锚点的声明比例 |
| 20 | 延迟与规模 | 在索引语料大小下 10k QPS 时的 p95 查询延迟 |
| 20 | 增量索引正确性 | 从 git push 到 50 文件提交可搜索的时间 |
| 15 | 用户体验与答案格式 | 引用可点击性、片段预览、后续交互能力 |
| **100** | | |

## 练习

1. 将 Voyage-code-3 替换为自托管的 nomic-embed-code。测量 MRR@10 的差异。报告在启用重排序后差距是否收敛。

2. 向语料中注入 20% 的生成代码（由 LLM 生成的样板），并重新评估。观察检索中毒现象。为 payload 添加一个 "generated" 标志并对这类命中降低权重。

3. 在你的语料规模下比较 Qdrant 混合检索与 pgvector + pgvectorscale。报告单查询（batch size 1）下的 p99。

4. 添加基于抽样的漂移检测：每周重新运行 100 问评测。若 MRR@10 降 > 5%，触发告警。

5. 扩展到跨语言符号解析：例如一个 Python 函数通过 gRPC 调用 Go 服务。使用符号图将两者连接。

## 关键词

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| AST-aware chunking | “函数级切分” | 在 tree-sitter 节点边界处切割代码，而不是固定 token 窗口 |
| Hybrid search | “稠密 + 稀疏” | 并行运行 BM25 和向量检索，合并 top-k，再重排序 |
| Cross-encoder rerank | “二阶段排序” | 对每个 (query, candidate) 对一起评分的模型，比余弦相似度更准确 |
| Prompt caching | “缓存的系统提示” | 2026 年 Claude / OpenAI 的特性，可对重复前缀 token 折扣高达 90% |
| Symbol graph | “代码图” | 跨文件、跨仓的 imports、调用、继承等边 |
| Citation faithfulness | “有根答案率” | 用户能通过点击锚点并阅读引用跨度来验证的声明比例 |
| Incremental re-index | “Push 到可搜索时间” | 从 git push 到变更符号可被查询的实时时间 |

## 深入阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓代码智能  
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本结业项目的参考深度解析  
- [Aider repo-map](https://aider.chat/docs/repomap.html) — 基于 tree-sitter 的排名式仓库视图  
- [Augment Code enterprise graph](https://www.augmentcode.com) — 商业符号图 RAG  
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现  
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 细节  
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — cross-encoder 参考文档  
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — 内部平台参考