# 文档与图表理解

> 文档不是照片。PDF、学术论文、发票或手写表单具有版面、表格、图表、脚注、页眉以及纯图像理解无法捕捉的语义结构。前 VLM（视觉语言模型）时代的堆栈是一个流水线：Tesseract OCR + LayoutLMv3 + 表格抽取启发式方法。VLM 时代替代了这一切，出现了无 OCR 的模型——Donut (2022)、Nougat (2023)、DocLLM (2023)——它们直接输出结构化标记。到 2026 年，前沿做法只是“把页面图像以 2576px 原生分辨率送入 Claude Opus 4.7”，结构化标记输出随之而来。本课按文档 AI 的三个时代来讲解。

**Type:** 构建  
**Languages:** Python (stdlib，布局感知文档解析器骨架)  
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)  
**Time:** ~180 分钟

## 学习目标

- 解释文档 AI 的三个时代：OCR 管道、无 OCR、VLM 原生（VLM-native）。
- 描述 LayoutLMv3 的三路输入流：文本、布局（边界框）、图像 patch，以及统一的掩码训练目标。
- 对比 Donut（无 OCR，图像 → 标记）、Nougat（学术论文 → LaTeX）、DocLLM（布局感知生成）、PaliGemma 2（VLM 原生）。
- 为新任务（发票、学术论文、手写表单、中文收据）选择合适的文档模型。

## 问题

“理解这个 PDF”看起来很简单，但实际很难。信息分散在：

- 文本内容（约 90% 的信号）。
- 布局（标题、脚注、侧栏、双栏格式）。
- 表格（行、列、合并单元格）。
- 图形与图表。
- 手写标注。
- 字体与排版（标题与正文的区别）。

原始 OCR 会导出文本但丢失其余信息。一个关心发票的系统需要知道“Total: $1,245”是来自右下角，而不是脚注。

## 概念

### 时代 1 — OCR 管道（pre-2021）

经典堆栈：

1. PDF → 每页图像。
2. Tesseract（或商业 OCR）提取文本并输出每词边界框。
3. 布局分析器识别文本块（标题、表格、段落）。
4. 表格结构识别器解析表格。
5. 领域规则 + 正则表达式抽取字段。

在干净的印刷文本上效果良好。对手写、倾斜扫描、复杂表格、非英语脚本会失效。每一种失败模式都需要自定义例外路径。

### TrOCR (2021)

TrOCR（Li 等，arXiv:2109.10282）用一个在合成与真实文本图像上训练的 Transformer 编码器-解码器，取代了 Tesseract 的经典 CNN-CTC。对手写与多语言文本是质的提升。依然属于流水线（先检测再 TrOCR 再布局），但 OCR 步骤大幅改进。

### 时代 2 — 无 OCR（2022-2023）

首批无 OCR 模型的想法是：完全跳过检测，直接将图像像素映射到结构化输出。

Donut（Kim 等，arXiv:2111.15664）：
- 编码器-解码器 Transformer，编码器为 Swin-B。
- 输出可为表单理解的 JSON、摘要的 Markdown，或任何任务特定的 schema。
- 无 OCR、无显式布局检测、无检测步骤。

Nougat（Blecher 等，arXiv:2308.13418）：
- 专门在学术论文上训练。
- 输出为 LaTeX / Markdown。
- 能处理方程、多栏布局、图表。
- 成为每个 arXiv 解析器调用的模型。

这些是专用模型，不是通用模型。Donut 在学术论文上会失败；Nougat 在发票上会失败。

### LayoutLMv3 (2022)

另一条路线。LayoutLMv3（Huang 等，arXiv:2204.08387）保留 OCR，但增强布局理解：

- 三路输入流：OCR 文本 token、每个 token 的 2D 边界框、图像 patch。
- 跨三种模态的掩码训练目标（掩码文本、掩码 patch、掩码布局）。
- 下游任务：分类、实体抽取、表格问答。

LayoutLMv3 是基于 OCR 的文档理解的巅峰。对表单和发票表现强劲。需要上游 OCR。是标准化文档基准上在 VLM 出现前的最佳精度模型。

### DocLLM (2023)

DocLLM（Wang 等，arXiv:2401.00908）是 LayoutLM 的生成式近亲。基于布局 token 条件生成自由文本答案。对文档问答更好；仍依赖 OCR 输入。

### 时代 3 — VLM 原生（2024+）

从 2024 年起，VLM 足够强大可以完全替代流水线。把整页图像以高分辨率输入 VLM，提出问题，直接得到答案。

- LLaVA-NeXT 336-tile AnyRes 适用于小型文档。
- Qwen2.5-VL 动态分辨率本地支持 2048+ 像素。
- Claude Opus 4.7 支持 2576px 文档输入。
- PaliGemma 2（2025 年 4 月）专门为文档 + 手写训练。

VLM 原生与 OCR 管道的差距迅速缩小。到 2026 年，VLM 原生在下列方面胜出：

- 场景文本（手写 + 印刷混合、多语种混合）。
- 含合并单元格的复杂表格。
- 嵌入文本中的数学公式。
- 带注释的图形。

OCR 管道仍在以下场景中占优：

- 在每页延迟至关重要的大规模纯扫描工作负载下更省成本。
- 管道的可靠性（确定性失败路径对比 VLM 的幻觉）。
- 需要可审计 OCR 输出的监管环境。

### Claude 4.7 / GPT-5 前沿

在 2576 像素原生输入下，前沿 VLM 在文档理解上已接近人类准确率。2026 年初的基准数据：

- DocVQA：Claude 4.7 ~95.1，PaliGemma 2 ~88.4，Nougat ~77.3，流水线式 LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2，GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

闭源模型的差距主要是分辨率和基 LLM 的规模。7B 规模的开源模型落后几分，但在赶上。

### 数学方程与 LaTeX 输出

学术论文需要精确的 LaTeX 方程输出。Nougat 在这方面做了针对性训练。以 LaTeX 作为目标训练的 VLM（如 Qwen2.5-VL-Math、Nougat 的衍生版）能产生可用的 LaTeX。没有明确 LaTeX 训练的 VLM 会产出可读但不精确的转录。

对于 2026 年的学术论文处理流水线：在 PDF 上链式调用 Nougat，再对棘手页面用 VLM 处理。

### 手写

仍然是最难的子任务。印刷与手写混合（医生笔记、填表）是 OCR 管道在成本上仍然占优的场景。纯手写的 VLM 在进步（Claude 4.7、PaliGemma 2）。

### 2026 年建议

针对新文档 AI 项目：

- 纯印刷发票且规模大：LayoutLMv3 + 规则，成本效益高。
- 混合文档（学术 + 手写 + 表单）：VLM 原生（PaliGemma 2 或 Qwen2.5-VL）。
- 全量 arXiv 入库：对数学用 Nougat，对图表用 VLM。
- 监管场景：以 OCR 管道为主，辅以 VLM 校验交叉验证。

## 使用示例

`code/main.py`：

- 一个玩具级的布局感知 tokenizer：给定 (text, bbox) 对，生成类似 LayoutLMv3 的输入。
- 一个 Donut 风格的任务 schema 生成器：表单的 JSON 模板。
- 比较每页在 OCR 管道、Donut、Nougat 和 VLM 原生下的 token 预算。

## 部署产出

本课产出 `outputs/skill-document-ai-stack-picker.md`。给定一个文档 AI 项目（领域、规模、质量、监管要求），在 OCR 管道、无 OCR 专家型、VLM 原生 三者间做抉择。

## 练习

1. 你的项目是每天 1,000 万张发票。哪种堆栈能在不丢失准确率的前提下最小化每页成本？

2. 为什么 LayoutLMv3 在表单问答上优于纯 CLIP 风格的 VLM，但在场景文本上表现不及 VLM？bbox 流让步了什么？

3. Nougat 能生成 LaTeX。提出一个测试用例，说明 VLM 原生在 LaTeX 保真度上胜过 Nougat 的情形，以及一个 Nougat 占优的情形。

4. 阅读 PaliGemma 2 论文（Google，2024）。相比 PaliGemma 1，哪项关键训练数据的增加提升了文档准确率？

5. 设计一个监管安全的混合方案：以 OCR 管道为主，VLM 为次级交叉校验。出现分歧时你如何裁决？

## 关键术语

| 术语 | 人们如何描述 | 实际含义 |
|------|-----------------|------------------------|
| OCR pipeline | “Tesseract-style” | 分阶段堆栈：检测 -> OCR -> 布局 -> 规则；确定性但脆弱 |
| OCR-free | “Donut-style” | 图像到输出的 Transformer，跳过显式 OCR；单模型方案 |
| Layout-aware | “LayoutLM” | 输入包含每个 token 的边界框坐标；模态间统一掩码训练 |
| VLM-native | “Frontier VLM” | 将页面图像直接输入 Claude/GPT/Qwen 等高分辨率 VLM；无流水线 |
| DocVQA | “Doc benchmark” | 文档 VQA 标准基准；最常引用的分数 |
| Markup output | “LaTeX / MD” | 结构化输出格式（如 LaTeX/Markdown），用于下游自动化 |

## 延伸阅读

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)  
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)  
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)  
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)  
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)