# ColPali 与原生视觉文档 RAG

> 传统的 RAG 将 PDF 解析为文本、划分为片段、对片段做嵌入并存储向量。每一步都会丢失信号：OCR 丢失图表数据，分块打断表格行，文本嵌入忽略图像。ColPali（Faysse 等，2024 年 7 月）提出了更简单的问题：为什么还要提取文本？直接用 PaliGemma 对页面图像做嵌入，使用 ColBERT 风格的 late interaction（晚交互）做检索，并保留文档携带的布局、图形、字体和格式化信号。公开基准显示：在视觉丰富的文档上，端到端准确率比文本 RAG 高出 20–40%。ColQwen2、ColSmol 和 VisRAG 继承并扩展了这一模式。本课阅读视觉原生 RAG 的思路并构建一个微型的类 ColPali 索引器。

**Type:** 构建  
**Languages:** Python（标准库，multi-vector indexer + MaxSim 评分器）  
**Prerequisites:** Phase 11（LLM Engineering — RAG 基础），Phase 12 · 05（LLaVA）  
**Time:** ~180 分钟

## 学习目标

- 解释 bi-encoder 检索（每个文档一个向量）与 late-interaction 检索（每个文档多个向量）之间的区别。  
- 描述 ColBERT 的 MaxSim 操作以及 ColPali 如何将其从文本 token 推广到图像 patch。  
- 构建一个微型的 ColPali 风格索引器：页面 → patch 嵌入 → 对查询 token 嵌入做 MaxSim → 返回 top-k 页面。  
- 在发票 / 财务报告用例上比较 ColPali + Qwen2.5-VL 生成器 与 文本-RAG + GPT-4 的表现。

## 问题背景

对 PDF 使用文本-RAG 会丢掉文档的大部分信息。财报的第三季度营收增长常常在图表里；医学报告的结论存在于带注释的影像；法律合同的签名块是一个布局事实，而不是纯文本事实。

文本-RAG 流程：

1. PDF → 通过 OCR / pdftotext 得到文本。  
2. 文本 → 300–500 token 的片段。  
3. 片段 → bi-encoder 嵌入（一个向量）。  
4. 用户查询 → 嵌入 → 余弦相似度 → top-k 片段。  
5. 片段 + 查询 → LLM 生成答案。

五个有损步骤。图表无法被捕获，表格会被跨片段打断，多栏布局被展平，图注消失。

ColPali 的修正办法：跳过 OCR，直接对页面图像做嵌入。使用 ColBERT 风格的 late interaction，在检索时允许模型关注细粒度的 patch。

## 概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia, arXiv:2004.12832）是一种文本检索方法。它不是为每个文档产出一个向量，而是为每个 token 产出一个向量。在查询时：

- 查询 token 得到自己的嵌入（N_q 个向量）。  
- 文档 token 得到嵌入（N_d 个向量，通常缓存）。  
- 得分 = 对每个查询 token 求在文档 token 上的最大余弦相似度之和：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 操作。每个查询 token “选择” 与其最匹配的文档 token，最终得分是这些最大值的求和。

优点：召回率强，能处理词级语义。缺点：每个文档有 N_d 个向量，存储代价高。

### ColPali

ColPali（Faysse 等，arXiv:2407.01449）将 ColBERT 模式应用到图像上。

- 每页由 PaliGemma（ViT + language）编码为 patch 嵌入：每页 N_p 个向量。  
- 每个用户查询（文本）被编码为查询 token 嵌入：N_q 个向量。  
- 得分 = Σ_i max_j cos(q_i, p_j)，即对查询文本 token 与页面图像 patch 做 MaxSim。  
- 按总得分检索 top-k 页面。

在文档摄取时：对每页用 PaliGemma 做嵌入，存储所有 patch 嵌入。在查询时：对查询做 token 嵌入，计算与所有已存页面嵌入的 MaxSim，返回 top-k 页面。

优点：端到端在视觉丰富文档上比文本-RAG 高 20–40%。每个 patch 含有局部布局和内容信号。  
缺点：每页 N_p 个 patch × 4 字节浮点 × D 维向量导致存储迅速膨胀。可以用 PQ / OPQ 量化缓解。

### ColQwen2 与 ColSmol

ColQwen2（illuin-tech，2024–2025）用 Qwen2-VL 替代 PaliGemma。更好的基础编码器带来更好的检索效果。

ColSmol 是面向本地 / 边缘部署的小尺度变体。一个约 1B 参数的 ColSmol 检索器可在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu 等，arXiv:2410.10594）是另一种变体：不是对 patch 做 MaxSim，而是用 VLM 将整页池化为单一向量，然后做 bi-encoder 检索。索引更快、存储更小，但召回较弱。

质量与成本的权衡：对质量需求高选 ColPali，对可扩展性和成本敏感选 VisRAG。

### M3DocRAG

M3DocRAG（Cho 等，arXiv:2411.04952）把多模态检索扩展到多页、多文档推理。它跨文档检索页面，并为 VLM 组合出一个多页上下文。

### ViDoRe — 基准

ColPali 的配套基准。视觉文档检索评估（Visual Document Retrieval Evaluation）。任务包括财务报告、学术论文、行政文档、病历、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上大约得分 ~80% nDCG@5；同样文档上的文本-RAG 得分约为 ~50–60%。

### 端到端 RAG 流程

对于视觉原生 RAG：

1. 摄取：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch 嵌入。  
2. 查询：用户文本 → 查询 token 嵌入 → 对所有索引页面做 MaxSim → top-k 页面。  
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 给出答案。

整个流程不使用 OCR。图形、图表、字体和布局都能流向最终答案。

### 存储计算

一个 50 页的财务报告，每页 729 个 patch，每个嵌入 128 维：

- ColPali：50 * 729 * 128 * 4 字节 ≈ 18 MB 原始，PQ 后约 4 MB。  
- 文本-RAG：50 个片段 * 768 维 * 4 字节 ≈ 150 kB。

ColPali 每文档大约多占 30 倍存储。规模化时，OPQ / PQ 可把比例降到 ~5–10 倍，通常可接受。

### 文本-RAG 仍占优的场景

- 纯文本文档且没有布局信号（维基文章、聊天记录）。文本-RAG 更简单且更省存储。  
- 百万级页面归档，存储成本主导预算。  
- 严格的合规要求需要可提取的 OCR 文本与检索并行。

对 2026 年的大多数视觉丰富场景——财务报告、学术论文、法律合同、病历、用户体验文档——视觉原生 RAG 更占优势。

## 使用示例

`code/main.py`：

- 玩具 patch 编码器：将“页面”（小网格的特征向量）映射为一组 patch 嵌入数组。  
- MaxSim 评分器：计算查询 token 嵌入集合与页面 patch 集合之间的 ColBERT 风格得分。  
- 为 5 个玩具页面建立索引，运行 3 个查询，返回带得分的 top-k。

## 交付产物

本课产出 `outputs/skill-vision-rag-designer.md`。给定一个文档-RAG 项目，选择 ColPali / ColQwen2 / VisRAG / 文本-RAG 并估算存储需求。

## 练习

1. 一个 200 页的年报，729 个 patch/页，128 维嵌入，4 字节浮点。计算原始存储与 PQ 压缩（8x）后的存储。  
2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。相比简单的均值相似，这个求和捕获了什么？  
3. ColPali 将页面索引为 patch 集。如果改为像 ColBERT 那样在词级别建立索引，会有哪些变化？权衡是什么？  
4. 为一个 100 万页语料设计端到端流水线，查询时延预算为每次 500ms。选择 ColQwen2 或 VisRAG 并给出理由。  
5. 阅读 M3DocRAG（arXiv:2411.04952）。描述其多页注意力模式以及它与单页 ColPali 检索的不同点。

## 关键术语

| 术语 | 大家如何称呼 | 实际含义 |
|------|---------------|----------|
| Late interaction | “ColBERT-style” | 使用按 token 或按 patch 的多向量 + MaxSim 的检索，而不是单一文档向量 |
| MaxSim | “Max-over-patches” | 对每个查询 token 选择最高相似度的文档 token/patch；对查询 token 求和 |
| Bi-encoder | “Single-vector” | 每个文档一个向量；更快但粒度丢失 |
| Multi-vector | “Many-vectors-per-doc” | 为每个文档/页面存储 N_p 个向量；存储成本增加但召回提升 |
| Patch embedding | “Page feature” | 来自 VLM 编码器的每个图像 patch 的向量，按页缓存 |
| ViDoRe | “Vision doc bench” | ColPali 的视觉文档检索基准套件 |
| PQ quantization | “Product quantization” | 一种压缩方法，在缩小存储约 8 倍的同时尽量保持向量相似度 |

## 深入阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)  
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)  
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)  
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)  
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)