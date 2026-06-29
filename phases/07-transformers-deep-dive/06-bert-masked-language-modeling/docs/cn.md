# BERT — 掩码语言建模

> GPT 预测下一个词。BERT 预测缺失的词。一句话的差别 —— 以及半个十年的所有以嵌入为中心的进展。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)  
**Time:** ~45 分钟

## 问题

在 2018 年，每个 NLP 任务——情感分析、命名实体识别、问答、蕴含——都在自己的有标签数据上从头训练自己的模型。那时还没有一个可以微调的“理解英语”的预训练检查点。ELMo（2018）表明可以用双向 LSTM 预训练上下文嵌入；这有帮助但不够通用。

BERT（Devlin 等，2018）提出：如果我们用一个 transformer encoder，在互联网上的每个句子上训练它，并强制它从两侧上下文中预测缺失的词会怎样？然后你在下游任务上微调一个头。参数效率成为一大突破。

结果是：在 18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）统治了当时存在的所有 NLP 排行榜。到 2020 年，地球上每个搜索引擎、内容审查流水线和语义搜索系统里都装着一个 BERT。

在 2026 年，encoder-only 模型仍然是分类、检索和结构化抽取的正确工具——相对于 decoder 它们在每个 token 上运行快 5–10×，并且它们的嵌入构成了每个现代检索栈的基础。ModernBERT（2024 年 12 月）将架构推到了 8K 上下文，结合了 Flash Attention + RoPE + GeGLU。

## 概念

![掩码语言建模：选择标记、掩码它们、预测原始词](../assets/bert-mlm.svg)

### 训练信号

取一句话：`the quick brown fox jumps over the lazy dog`。

随机掩码 15% 的 token：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型在被掩码的位置预测原始 token。因为 encoder 是双向的，所以预测位置 1 的 `[MASK]` 可以使用位置 2+ 的 `brown fox jumps`。这是 GPT 无法做到的事情。

### BERT 的掩码规则

在被选中用于预测的 15% token 中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为一个随机 token。
- 10% 保持不变。

为什么不总是用 `[MASK]`？因为在推理时从未出现 `[MASK]`。如果在预训练中 100% 的被掩码位置都出现 `[MASK]`，会在预训练和微调之间引入分布偏移。10% 的随机 + 10% 的不变保持模型的鲁棒性（避免过度依赖 `[MASK]`）。

### 下一句预测（NSP）——以及为何被移除

原始 BERT 还做了 NSP：给定两句 A 和 B，预测 B 是否跟在 A 之后。RoBERTa（2019）做了对比消融，证明 NSP 反而有害。现代 encoder 已经跳过了它。

### 在 2026 年的变化：ModernBERT

2024 年的 ModernBERT 论文用当代原语重建了 block：

| 组件 | 原始 BERT (2018) | ModernBERT (2024) |
|------|------------------|-------------------|
| 位置 | Learned absolute | RoPE |
| 激活函数 | GELU | GeGLU |
| 归一化 | LayerNorm | Pre-norm RMSNorm |
| 注意力 | 全局稠密 | 交替的局部（128）+ 全局 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

并且与 2018 年的栈不同，它本地支持 Flash-Attention。在序列长度 8K 时，推理比 DeBERTa-v3 快 2–3×，且在 GLUE 分数上更好。

### 2026 年仍然选择 encoder 的用例

| 任务 | 为什么 encoder 胜过 decoder |
|------|------------------------------|
| 检索 / 语义搜索嵌入 | 双向上下文 = 每个 token 更高质量的嵌入 |
| 分类（情感、意图、毒性） | 一次前向传播；无生成开销 |
| NER / token 标注 | 每个位置的输出，天然双向 |
| 零样本蕴含（NLI） | 在 encoder 之上接一个分类头 |
| RAG 的重排序器 | Cross-encoder 打分，比 LLM 重排序器快 10x |

```figure
transformer-residual
```

## 实现

### 步骤 1：掩码逻辑

见 `code/main.py`。函数 `create_mlm_batch` 接受一列 token ID、一个词表大小和一个掩码概率。返回的是应用了掩码的 input IDs 和 labels（只在被掩码的位置有标签，其余位置为 -100 —— PyTorch 的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 步骤 2：在一个小语料上运行 MLM 预测

在词表大小为 20、200 句子的语料上训练一个 2 层 encoder + MLM 头。此处不做反向传播 —— 我们做前向传播的健全性检查。完整训练需要 PyTorch。

### 步骤 3：比较掩码类型

展示三路规则如何在没有 `[MASK]` 的情况下仍保持模型可用性。在未掩码句子和掩码句子上进行预测。因为训练时模型见过这三种模式，两者都应产生合理的 token 分布。

### 步骤 4：微调头

用一个分类头替换 MLM 头，在一个玩具情感数据集上微调。仅训练头；encoder 冻结。这是每个 BERT 应用遵循的模式。

## 使用方法

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768) 表示 (batch, seq_len, hidden_dim)
```

**嵌入模型就是经过微调的 BERT。** 像 `sentence-transformers` 中的 `all-MiniLM-L6-v2` 这样的模型是用对比损失训练的 BERT。encoder 是一样的，区别在于损失函数。

**Cross-encoder 重排序器也是经过微调的 BERT。** 形式为 `[CLS] query [SEP] doc [SEP]` 的成对分类。query 和 doc 之间的双向注意力正是 cross-encoder 相对于 biencoder 的质量优势来源。

**什么时候在 2026 年不选 BERT。** 任何生成型任务都不选。encoder 无法以自回归方式生成合理的 token。另外：在 1B 参数以下且一个小 decoder 可以以更灵活的方式匹配质量的场景（如 Phi-3-Mini、Qwen2-1.5B）也可考虑选择 decoder。

## 上线

见 `outputs/skill-bert-finetuner.md`。该 skill 规划了一个 BERT 微调（骨干模型选择、头规格、数据、评估、停止准则）以用于新的分类或抽取任务。

## 练习

1. **简单。** 运行 `code/main.py` 并打印 10,000 个 token 的掩码分布。确认大约 15% 被选中，其中大约 80% 变为 `[MASK]`。
2. **中等。** 实现整词掩码（whole-word masking）：如果一个词被分成子词，掩码所有子词或都不掩码。在一个 500 句子的语料上测量这是否提高了 MLM 的准确率。
3. **困难。** 在公开数据集上用 10,000 句训练一个微小的（2 层，d=64）BERT。对 `[CLS]` token 进行 SST-2 情感微调。在参数匹配的解码器-only 基线下比较——哪个更好？

## 术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|---------|
| MLM | "Masked language modeling" | 训练信号：随机替换 15% 的 token 为 `[MASK]`，预测原始 token。 |
| Bidirectional | "两边看" | Encoder 的注意力没有因果掩码——每个位置能看到所有其他位置。 |
| `[CLS]` | "pooler token" | 在每个序列前添加的特殊 token；其最终嵌入被用作句子级表示。 |
| `[SEP]` | "segment separator" | 分隔成对序列（例如 query/doc、句子 A/B）。 |
| NSP | "Next sentence prediction" | BERT 的第二个预训练任务；RoBERTa 证明它无用，2019 年后被移除。 |
| Fine-tuning | "适配到任务" | 通常冻结 encoder 的大部分参数；在其上训练一个小头用于下游任务。 |
| Cross-encoder | "一个重排序器" | 一个同时接受 query 和 doc 作为输入的 BERT，输出相关性分数。 |
| ModernBERT | "2024 刷新" | 使用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力并支持 8K 上下文的 encoder 重建。 |

## 深入阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。  
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；取消了 NSP。  
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 替换 token 检测在相同计算预算下胜过 MLM。  
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。  
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 规范的 encoder 参考实现。