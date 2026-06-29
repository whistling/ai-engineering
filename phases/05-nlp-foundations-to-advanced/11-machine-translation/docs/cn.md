# 机器翻译

> 翻译是过去三十年为自然语言处理研究提供资金的任务，直到现在仍在继续资助这个领域。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 5 · 10（注意力机制）, Phase 5 · 04（GloVe、FastText、子词）  
**Time:** ~75 分钟

## 问题概述

模型读取一种语言的句子并生成另一种语言的句子。长度会变化。词序会变化。有些源语言词映射到多个目标词，反之亦然。成语拒绝一对一映射。英文的 "I miss you" 在法语里是 "tu me manques"——字面意思是“你对我来说是缺失的”。词级别的对齐在这种情况下毫无意义。

机器翻译这一任务迫使 NLP 发明了编码器-解码器、注意力机制、Transformer，最终催生了整个大规模语言模型范式。每一次进步都因为翻译质量可以被量化，而且人与机器之间的差距一直非常顽固。

本课跳过历史介绍，教授 2026 年的工作流：预训练的多语种编码器-解码器（如 NLLB-200 或 mBART）、子词分词、束搜索、BLEU 与 chrF 评估，以及仍会在生产环境中未被发现的那几类失败模式。

## 概念

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

现代机器翻译是基于在平行语料上训练的 Transformer 编码器-解码器。编码器以源语言的分词方式读取输入。解码器使用交叉注意力（第 10 课）利用编码器输出逐个子词地生成目标语言。解码时使用束搜索以避免贪心解码的陷阱。输出随后进行反分词、恢复大小写，并与参考译文进行打分。

影响实际 MT 质量的三个运作选择：

- Tokenizer（分词器）。在混合语言语料上训练的 SentencePiece BPE。跨语言共享词表是 NLLB 实现零样本对（zero-shot pairs）的关键。
- 模型规模。NLLB-200 distilled 600M 可以在笔记本上运行。NLLB-200 3.3B 是已发布的生产默认。54.5B 是研究上的上限。
- 解码策略。一般内容使用束宽 4-5。使用长度惩罚以避免输出过短。需要术语一致性时使用受限解码（constrained decoding）。

```figure
seq2seq-alignment
```

## 构建

### 步骤 1：调用预训练 MT 模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三点很重要。`src_lang` 告诉分词器该使用哪种脚本和分段策略。`forced_bos_token_id` 告诉解码器要生成哪种语言。两者都是 NLLB 的特定技巧；mBART 和 M2M-100 有自己的约定，彼此不可互换。

### 步骤 2：BLEU 与 chrF

BLEU 测量输出与参考之间的 n-gram 重叠。使用四个参考 n-gram 大小（1-4），取精确率的几何平均，并对过短输出施加简短惩罚（brevity penalty）。得分范围在 [0, 100]。这是常用指标，但解释起来令人沮丧：30 BLEU 是“可用”，40 是“好”，50 是“非常好”；小于 1 BLEU 的差异通常是噪声。

chrF 测量字符层面的 F 分数。对于形态丰富的语言，chrF 更敏感，而 BLEU 可能会漏掉许多匹配。常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

务必使用 `sacrebleu`。它对分词进行规范化，使得不同论文之间的分数可比。自己实现 BLEU 是导致误导性基准出现的常见原因。

### 三层评估体系（2026）

现代 MT 评估使用三类互补的度量。发布时至少使用两种。

- Heuristic（启发式）：BLEU、chrF。快速、基于参考、可解释，但对同义改写不敏感。用于传统比较和回归检测。
- Learned（学习型）：COMET、BLEURT、BERTScore。基于人为评判训练的神经模型；比较翻译与源文和参考之间的语义相似度。自 2023 年以来，COMET 在与人工评分的一致性方面表现最好，是 2026 年在需要质量保证时的生产默认选择。
- LLM-as-judge（以 LLM 作为评审，免参考）：提示大模型对翻译在流利性、充分性、语气、文化适切性等维度给出评分。当评分量表设计合理时，GPT-4 作为评审与人工一致率约为 80%。用于没有参考译文的开放式内容评估。

实用的 2026 堆栈：使用 `sacrebleu` 计算 BLEU 与 chrF，使用 `unbabel-comet` 计算 COMET，并用提示化的 LLM 作为最终面向人的信号。在信赖任一度量到生产数据之前，务必用 50-100 个人工标注样本对其进行校准。

免参考度量（COMET-QE、BLEURT-QE、LLM-as-judge）可以在没有参考译文的情况下评估翻译，这对长尾语言对（没有参考译文的情况）尤其重要。

### 步骤 3：生产中会出什么问题

上述工作流能在 80% 的情况下生成流利翻译，但在剩余 20% 中会“悄无声息”地出错。常见失败模式：

- 幻觉（Hallucination）。模型捏造源文中不存在的内容。在不熟悉的领域词汇中很常见。症状：输出流畅但包含源文未陈述的事实。缓解办法：对领域术语使用受限解码（glossary/constrained decoding）、对监管类内容进行人工复核、监控输出显著长于输入的情况。
- 生成错语种（Off-target generation）。模型翻译成了错误的语言。NLLB 在罕见语言对上对此问题尤为敏感。缓解办法：校验 `forced_bos_token_id` 并始终对输出运行语言识别模型进行检查。
- 术语漂移（Terminology drift）。例如 “Sign up” 在文档 A 中被译为 “s'inscrire”，在文档 B 中被译为 “créer un compte”。对于 UI 文本和面向用户的字符串，一致性比单次质量更重要。缓解办法：使用术语表约束解码或后编辑字典替换。
- 礼貌/语域不匹配（Formality mismatch）。如法语的 “tu” vs “vous”，日语的敬语等级。模型会倾向训练集中更常见的形式。对面向客户的内容通常是不合适的。缓解办法：如果模型支持，可以用带有礼貌等级标记的前缀提示；或在正式语料上微调小模型。
- 对短输入的长度爆炸（Length explosion on short input）。非常短的源句常常导致翻译过长，因为长度惩罚在源长度约低于 ~5 个 token 时会突然失效。缓解办法：对生成长度设置与源长度成比例的硬性上限。

### 步骤 4：针对领域进行微调

预训练模型是通用模型。法律、医疗或游戏对话的翻译在使用领域平行语料微调后会明显受益。配方并不复杂：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条高质量的平行示例通常胜过几十万条嘈杂的网络抓取数据。训练数据的质量是生产中能操控的最重要杠杆。

## 使用建议

2026 年的生产级 MT 堆栈：

| 用例 | 推荐起点 |
|------|----------|
| 任意语言对，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 以英语为中心、高质量，50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短批次、廉价推理，英语-法/德/西 | Helsinki-NLP / Marian 模型 |
| 对延迟敏感的浏览器端 | ONNX 量化后的 Marian（~50 MB） |
| 追求最高质量、愿意付费 | 使用 GPT-4 / Claude / Gemini 并通过提示实现翻译 |

到 2026 年，LLM 在若干语言对上已超越专业 MT 模型，尤其擅长习语翻译和长上下文。权衡点是每 token 的成本与延迟。当上下文长度、风格一致性或通过提示进行领域适应比吞吐量更重要时，选择 LLM。

## 发布流程（Ship It）

将以下内容保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: 评估机器翻译输出以便发布。
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

给定源文本和候选译文，输出：

1. 自动评分估计。给出预期的 BLEU 与 chrF 范围。说明是否有参考译文可用。
2. 五点可人工验证的检查表：（a）内容保留（无幻觉），（b）正确的语言，（c）语域 / 礼貌等级匹配，（d）与提供的术语表的一致性（如有），（e）无截断或长度爆炸。
3. 一个与领域相关的专项检查项。例如：法律领域检查命名实体和法条引用；医疗领域检查药品名称和剂量；UI 领域检查占位符变量 `{name}`。
4. 置信标记： "Ship" / "Ship with review" / "Do not ship"。将该标记与步骤 2 中发现问题的严重性关联起来。

未经对输出进行语言识别（language-ID）检查，拒绝发布翻译。除非用户明确选择免参考评分（COMET-QE、BLEURT-QE），否则拒绝在无参考的情况下进行评估。对任何超过 1000 token 的内容标记为可能需要分块翻译。
```

## 练习

1. 简单：使用 `nllb-200-distilled-600M` 将一段 5 句的英文段落翻译为法语，再翻回英文。测量往返（round-trip）与原文的相似度。你应当看到语义保留但词选择上存在漂移。
2. 中等：对翻译输出实现语言识别检查，使用 `fasttext lid.176` 或 `langdetect`。将其集成到 MT 调用流程中，以便在返回结果前捕捉到生成错语种的情况。
3. 困难：在你选择的 5,000 对领域语料上微调 `nllb-200-distilled-600M`。在微调前后对保留集计算 BLEU。报告哪些类型的句子获得了改进，哪些出现了退化。

## 术语表

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| BLEU | Translation score（翻译分） | n-gram 精确率并带有简短惩罚。范围 [0, 100]。 |
| chrF | Character F-score（字符 F 分） | 字符级别的 F 分数。对形态丰富的语言更敏感。 |
| NMT | Neural MT（神经机器翻译） | 在平行语料上训练的 Transformer 编码器-解码器。2017 年后成为默认方案。 |
| NLLB | No Language Left Behind | Meta 的 200 语言机器翻译模型家族。 |
| Constrained decoding | 受限解码 / 控制输出 | 强制特定 token 或 n-gram 出现/不出现于输出中。 |
| Hallucination | 幻觉 / 捏造内容 | 模型输出中没有源文支持的内容。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 论文。  
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 说明为何 `sacrebleu` 是报告 BLEU 的唯一正确方式。  
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 论文。  
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 实用的微调操作指南。