# Capstone 04 — 多模态文档问答（以视觉为先的 PDF、表格、图表）

> 到 2026 年，文档问答的前沿从先 OCR 再文本的做法转向了以视觉为先的晚期交互方法。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面视为图像，使用多向量晚期交互进行嵌入，让查询直接对图像补丁进行注意力匹配。在财务 10-K、科研论文和手写笔记上，这种模式显著优于先 OCR 再文本的方法。你的任务是构建端到端流水线，处理 1 万页并将结果与 OCR-then-text 进行并列发布。

**Type:** 结业项目  
**Languages:** Python（流水线），TypeScript（查看器 UI）  
**Prerequisites:** Phase 4（计算机视觉）、Phase 5（NLP）、Phase 7（transformers）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）  
**Phases exercised:** P4 · P5 · P7 · P11 · P12 · P17  
**Time:** 30 小时

## 问题

企业中有大量 OCR 管线会破坏的 PDF：带有旋转表格的扫描 10-K、充斥公式的科研论文、只作为图像才能理解的图表、手写批注。把这些当作以文本为先的管道会丢失一半信息。到 2026 年的答案是对原始页面图像进行晚期交互多向量检索。ColPali（Illuin Tech）首创了该思路；ColQwen2.5-v0.2 和 ColQwen3-omni 提升了精度。在 ViDoRe v3 上，以视觉为先的检索在分数上比 OCR-then-text 有明显优势 — 在图表、表格和手写内容上差距更大。

权衡是存储和延迟。一个 ColQwen 嵌入大约是每页 ~2048 个补丁向量，而不是单个 1024 维向量。原始存储会膨胀。DocPruner（2026）实现了 50% 的剪枝且无可测量的精度损失。你需要对 1 万页建立索引，测量 ViDoRe v3 的 nDCG@5，保证回答在 2 秒内返回，并与 OCR-then-text 基线做直接比较。

## 概念

晚期交互意味着每个查询 token 要与每个页面补丁 token 打分，并对每个查询 token 取最大得分后求和。你能得到细粒度匹配，而不需要单个 pooled 向量。多向量索引（Vespa、Qdrant 多向量或 AstraDB）存储每个补丁的嵌入，并在检索时运行 MaxSim。

回答器是一个视觉-语言模型，接受查询和 top-k 检索到的页面图像，输出带证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于公式和科学记号，可以使用 OCR 回退（Nougat、dots.ocr）作为可选文本通道。

评估是二维矩阵。一轴：内容类型（纯文本段落、密集表格、柱/折线图、手写笔记、公式）。另一轴：检索方法（以视觉为先的晚期交互 vs OCR-then-text vs 混合）。每个单元格记录 nDCG@5 和答案准确率。最终交付的是报告。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，纵向归一化  
- 晚期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（vidore 团队在 Hugging Face）  
- 索引：带多向量字段的 Vespa，或 Qdrant 多向量，或 支持 MaxSim 的 AstraDB  
- 剪枝：DocPruner 2026 策略（保留高方差补丁，50% 压缩，<0.5% 精度损失）  
- OCR 回退（公式 / 密集表格）：dots.ocr 或 Nougat  
- VLM 回答器：自托管 Qwen3-VL-30B 或 Gemini 2.5 Pro 托管；InternVL3 作为备用  
- 评估：ViDoRe v3 基准，M3DocVQA 用于多页推理评测  
- 查看器 UI：Next.js 15，使用画布覆盖展示证据区域

## 构建步骤

1. Ingest。遍历一个由 10-K、科研论文和扫描文档组成的语料，累计 1 万页。将每页渲染为 1536x2048 的 PNG。持久化记录 `{doc_id, page_num, image_path}`。

2. Embed。对每个页面图像运行 ColQwen2.5-v0.2。输出形状约为每页 2048 个补丁的嵌入，维度 128。应用 DocPruner 保留最高信号的一半。写入 Vespa 的多向量字段或 Qdrant 多向量。

3. Query。对每个传入查询，使用查询塔生成 token 级嵌入。对索引运行 MaxSim：对每个查询 token，在页面补丁嵌入上取点积最大值并求和。返回 top-k 页面。

4. Synthesize。将查询与 top-5 页面图像一并传给 Qwen3-VL-30B。Prompt 示例：“仅使用提供的页面回答。按 (doc_id, page) 引用每个断言，并注明区域名称（figure、table、paragraph）。”

5. 证据区域。后处理答案以提取被引用的区域。如果 VLM 输出边界框（Qwen3-VL 支持），在查看器中渲染为覆盖层。

6. OCR 回退。对于被判定为公式密集的页面（基于图像方差的启发式），运行 Nougat 或 dots.ocr，并将 OCR 文本作为与图像并行的额外通道传入。

7. 评估。运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页 QA 准确率）。同时对相同语料运行 OCR-then-text 管线并使用相同的合成器。生成按内容类型 × 方法的比较矩阵。

8. UI。先做 Streamlit 原型；生产级使用 Next.js 15，支持逐页证据区域覆盖显示。

## 使用示例

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付

`outputs/skill-doc-qa.md` 描述交付物：一个针对特定语料调优的以视觉为先的多模态文档问答系统，并在 ViDoRe v3 上与 OCR-then-text 基线进行评估比较。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA accuracy | 基准指标，与 OCR-text 基线及已发布排行榜比较 |
| 20 | Evidence-region grounding | 被引用区域中实际包含答案跨度的比例 |
| 20 | Storage and latency engineering | DocPruner 压缩比、索引 p95、答复 p95 |
| 20 | Multi-page reasoning | 在人工标注的 100 题多页集合上的准确率 |
| 15 | Source-inspection UX | 查看器清晰度、覆盖层保真度、并列比较工具 |
| **100** | | |

## 练习

1. 在相同语料上比较 ColQwen2.5-v0.2 与 ColQwen3-omni。哪些页面一个能答对而另一个错过？向索引添加“内容类别”标签以按类型路由。

2. 激进剪枝嵌入（75%、90%）。找到压缩悬崖：ViDoRe nDCG@5 跌破 OCR 基线的临界点。

3. 构建混合方案：并行运行 OCR-then-text 和 ColQwen，用 RRF 融合，然后用 cross-encoder 重排序。混合是否优于任一单独方法？在哪些情形下帮助最大？

4. 用更小的 VLM（Qwen2.5-VL-7B）替换 Qwen3-VL-30B。测量 accuracy-per-dollar 曲线。

5. 加入手写笔记支持。渲染手写语料，用 ColQwen 嵌入并测量检索效果。与手写 OCR 管线比较。

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Late interaction | "ColPali-style retrieval" | 查询 token 与页面补丁独立打分；MaxSim 聚合 |
| Multi-vector | "Per-patch embedding" | 每个文档有多个向量，而非一个 pooled 向量 |
| MaxSim | "Late-interaction scoring" | 对每个查询 token，在文档向量上取最大相似度；求和 |
| DocPruner | "Patch compression" | 2026 年的剪枝方法，保留 50% 补丁且精度影响极小 |
| ViDoRe v3 | "Document-retrieval benchmark" | 2026 年的视觉文档检索评测标准 |
| Evidence region | "Cited bounding box" | 在源页面上定位答案跨度的边界框 |
| OCR fallback | "Equation channel" | 在以视觉为先外，还用作公式/表格重的文本通道 |

（术语说明示例：将“Embeddings”译为“嵌入”，“Fine-tuning”译为“微调”，“Context window”译为“上下文窗口”，“few-shot”译为“少样本”，“chain-of-thought”译为“思维链”，“guardrails”译为“护栏”，“function calling”译为“函数调用”，“speculative decoding”译为“投机性解码”，“positional embeddings”译为“位置嵌入”，“self-attention”译为“自注意力”，“instruction tuning”译为“指令微调”，“distributed training”译为“分布式训练”。）

## 延伸阅读

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 参考的晚期交互文档检索实现  
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — 基础方法论文  
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — 生产可用的检查点  
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线  
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — 参考的服务端栈  
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 备选索引  
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 备选的托管索引  
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式的 OCR 回退