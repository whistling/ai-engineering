# ASCII 艺术与视觉越狱

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753)。将危险请求中的安全相关词元进行掩蔽，用相同字母的 ASCII 艺术渲染替换它们，并发送被伪装的提示。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法稳健识别 ASCII 艺术词元。该攻击能绕过 PPL（困惑度过滤器）、改写防御和重分词。相关工作：ViTC 基准衡量对非语义视觉提示的识别能力；StructuralSleight 将该方法推广为一类不常见的文本编码结构（树、图、嵌套 JSON）的编码攻击。

**Type:** 构建  
**Languages:** Python（标准库，ArtPrompt token-masking harness）  
**Prerequisites:** Phase 18 · 12（PAIR）、Phase 18 · 13（MSJ）  
**Time:** ~60 分钟

## 学习目标

- 描述 ArtPrompt 攻击：单词识别步骤、ASCII 艺术替换、最终伪装提示的生成流程。
- 解释为什么标准防御（PPL、改写、防重分词）在 ArtPrompt 上失效。
- 定义 ViTC 并描述其度量内容。
- 描述 StructuralSleight 如何将其推广到任意的不常见文本编码结构（UTES）。

## 问题概述

通过改写和角色扮演的攻击（第 12 课）以及通过长上下文的攻击（第 13 课）针对的是文本层面的模式。ArtPrompt 则作用于识别层面：模型未能解析被禁止的词元，而是解析由字符渲染的图像。安全过滤器只看到无害的标点符号；模型看到了一个单词。

## 概念

### ArtPrompt，两步法

步骤 1. 单词识别。给定一个有害请求，攻击者使用 LLM 识别安全相关的词汇（例如在“如何制造炸弹”中识别出 “bomb”）。

步骤 2. 伪装提示生成。将每个识别出的词替换为其 ASCII 艺术渲染（例如 7x5 或 7x7 的字符块，构成字母形状）。模型接收到的是由标点和空格组成的网格——能力足够强的模型可以将其识别为该单词；安全过滤器则只看到这个字符网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 在他们的基准子集上都未能稳健阻断该攻击，攻击成功率在 75% 以上。

### 为什么标准防御会失效

- **PPL（困惑度过滤器）。** ASCII 艺术会导致高困惑度——但所有新颖输入也会如此。将阈值设为能阻断 ArtPrompt 的同时，也会拦截合法的结构化输入。
- **改写（Paraphrase）。** 对提示进行改写会破坏 ASCII 艺术，但实际上，进行改写的 LLM 经常会保留或重构这些艺术字符。
- **重分词（Retokenization）。** 改变分词方式并不能改变模型对字母形状的视觉识别。

根本问题在于安全过滤器通常工作于词元或语义层面，而 ArtPrompt 在视觉识别层面生效。

### ViTC 基准

衡量对非语义视觉提示（如 ASCII 艺术、wingdings、其他非文本语义的视觉内容）的识别能力。ArtPrompt 的有效性与 ViTC 的准确率相关：模型越擅长读取视觉文本，ArtPrompt 在该模型上越有效。这体现了能力—安全的权衡。

### StructuralSleight

将 ArtPrompt 推广：不常见文本编码结构（UTES）。例如树、图、嵌套 JSON、CSV 嵌套在 JSON 中、diff 风格的代码块等。如果某种结构在训练数据中罕见但模型能解析，它就可以被用来走私有害内容。

防御含义：安全检测必须在模型能解析的所有结构表示上泛化，而这些结构集合既大且还在增长。

### 图像模态类比

视觉 LLM（如 GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。使用真实图像的 ArtPrompt 风格攻击比 ASCII 艺术更强，因为图像编码器会产生更丰富的信号。

### 在 Phase 18 中的位置

第 12–14 课描述了三类正交的攻击向量：迭代精炼（PAIR）、上下文长度（MSJ）与编码（ArtPrompt/StructuralSleight）。第 15 课从以模型为中心的攻击转向系统边界攻击（间接提示注入）。第 16 课描述防御工具链的应对。

## 使用指南

`code/main.py` 构建了一个简易的 ArtPrompt。你可以用它将有害查询中的特定词用 ASCII 艺术字形进行伪装，验证被伪装的字符串能否通过简单的关键字过滤器，并（可选）使用一个简单的识别器将伪装字符串解码回原文。

## 产出

本课会产生 `outputs/skill-encoding-audit.md`。给定一份越狱防御报告，它会列举所覆盖的编码攻击家族（ASCII 艺术、base64、1337 文化（leet-speak）、UTF-8 同形字符、UTES）以及拦截每种攻击的防御层级。

## 练习

1. 运行 `code/main.py`。验证被伪装的字符串能通过一个简单的关键字过滤器。报告所需的字符级改变量。
2. 为相同目标词实现第二种编码：base64。比较该方法与 ArtPrompt 的过滤绕过率及恢复难度。
3. 阅读 Jiang et al. 2024 第 4.3 节（五模型结果）。提出一个原因，解释为何 Claude 在相同基准上比 Gemini 更能抵抗 ArtPrompt。
4. 设计一个预生成防御，检测提示中呈 ASCII 艺术形态的区域。衡量该方法在合法代码、表格与数学符号上的误报率。
5. StructuralSleight 列出了 10 种编码结构。绘制一个通用防御草图，能够处理所有 10 种结构，并估算每条被防护提示的计算成本。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| ArtPrompt | “ASCII 艺术攻击” | 两步越狱：用 ASCII 艺术渲染掩蔽安全词的流程 |
| Cloaking | “隐藏单词” | 用模型可读但过滤器不可读的视觉表示替换被禁止的词元 |
| UTES | “不常见结构” | Uncommon Text-Encoded Structure — 用于走私内容的树、图、嵌套 JSON 等结构 |
| ViTC | “视觉—文本能力” | 衡量模型读取非语义视觉编码能力的基准 |
| Perplexity filter | “PPL 防御” | 拒绝高困惑度提示；失败的原因是合法的结构化输入也会得高分 |
| Retokenization | “重分词防御” | 用不同分词器预处理提示；失败原因是识别发生在视觉层面 |
| Homoglyph | “长相相似字符” | 与拉丁字母外观相同的 Unicode 字符；可绕过子串匹配检查 |

## 深入阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII 艺术越狱论文  
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES 的推广  
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 补充的迭代攻击  
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — 补充的长度攻击