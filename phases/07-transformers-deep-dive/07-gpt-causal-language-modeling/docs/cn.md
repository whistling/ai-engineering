# GPT — Causal Language Modeling

> BERT 看到两边。GPT 只看到过去。三角掩码是现代 AI 中最具影响力的一行代码。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (自注意力), Phase 7 · 05 (完整 Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 分钟

## 问题

语言模型回答一个问题：在给定前 `t-1` 个标记的情况下，第 `t` 个标记的概率分布是什么？在这个信号上训练——下一个标记预测——你就会得到一个可以逐个标记生成任意文本的模型。

要在整条序列上端到端并行训练，你需要每个位置的预测只依赖于更早的位置。否则模型会通过查看答案来轻易作弊。

因果掩码（causal mask）实现了这一点。它是一个上三角矩阵，包含在 softmax 之前加入到注意力分数上的 `-inf` 值。经过 softmax 后，这些位置的权重变为 0。每个位置只能关注自身和更早的位置。因为你对整条序列一次性应用它，你可以在一次前向传播中得到 N 个并行的下一个标记预测。

GPT-1 (2018)、GPT-2 (2019)、GPT-3 (2020)、GPT-4 (2023)、GPT-5 (2024)、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi —— 它们都是仅解码器的因果 transformer，核心循环相同。只是更大、更好的数据和更好的 RLHF。

## 概念

![因果掩码创建了一个三角形的注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定长度为 `N` 的序列，构造一个 `N × N` 矩阵：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 之前将 `M` 加到原始注意力分数上。`exp(-inf) = 0`，所以被屏蔽的位置贡献为零。注意力矩阵的每一行都是只关于先前位置的概率分布。

实现代价：一次 `torch.tril()` 调用。计算时间：纳秒级。对领域的影响：一切。

### 并行训练，串行推理

训练：对整条 `(N, d_model)` 序列做一次前向传播，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。在序列维度上并行。这就是 GPT 训练可扩展的原因——你可以在一次 GPU 运行中处理 100 万个标记的批次。

推理：按标记生成。输入 `[t1, t2, t3]` 得到 `t4`。再输入 `[t1, t2, t3, t4]` 得到 `t5`。再输入 `[t1, t2, t3, t4, t5]` 得到 `t6`。KV 缓存（Lesson 12）保存了 `t1…tn` 的隐藏状态，这样你就不用每一步都重新计算。但推理的串行深度 = 输出长度。这就是自回归代价，也是为什么解码是每个大模型延迟的瓶颈。

### 损失 —— 平移一位

给定标记 `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对于每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是整条序列的交叉熵。

你听过的每一个 transformer LM 都在用这个损失。预训练、微调、SFT —— 损失相同，数据不同。

### 解码策略

训练结束后，采样策略比人们想的更重要。

| 方法 | 它做了什么 | 何时使用 |
|------|------------|----------|
| Greedy | 每步取 argmax | 确定性任务、代码补全 |
| Temperature | 将 logits 除以 T 后采样 | 创意任务，T 越大多样性越高 |
| Top-k | 只从 top-k 标记中采样 | 剪掉低概率尾部 |
| Top-p (nucleus) | 从累计概率 ≥ p 的最小集合中采样 | 2020 年后常用默认；适应分布形状 |
| Min-p | 保留 `p > min_p * max_p` 的标记 | 2024 年后；比 top-p 更好地拒绝长尾 |
| Speculative decoding | 草稿模型提出 N 个标记，大模型验证 | 在相同质量下可降低 2–3× 延迟 |

在 2026 年，对于开源权重模型，min-p + temperature 0.7 是个合理默认。投机性解码是任何生产推理栈的标配。

### 让“GPT 配方”奏效的原因

1. **仅解码器。** 没有 encoder 的额外开销。每层只需一遍注意力 + FFN。
2. **扩展。** 124M → 1.5B → 175B → 万亿级。Chinchilla 缩放定律（Lesson 13）告诉你如何分配算力。
3. **上下文学习（In-context learning）。** 在 ~6B–13B 规模出现。模型可以在不微调的情况下遵循少样本示例。
4. **RLHF。** 基于人类偏好对预训练文本进行后训练，将模型转为聊天助手。
5. **Pre-norm + RoPE + SwiGLU。** 在大规模训练中稳定。

自 GPT-2 起核心架构没变太多。有趣的变化都发生在数据、规模和后训练上。

```figure
causal-mask
```

## 实现

### 第 1 步：因果掩码

见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前把它加到注意力分数上。这就是整个机制。

### 第 2 步：一个两层的 GPT-ish 模型

堆叠两个解码器块（掩码自注意力 + FFN，无交叉注意力）。加入词嵌入、位置编码，以及一个解嵌入（与词嵌入矩阵权重绑定 —— GPT-2 的常见技巧）。

### 第 3 步：端到端的下一个标记预测

在一个 20 标记的小词表上，在每个位置输出 logits。对齐到平移一位的目标计算交叉熵损失。不要反向传播 —— 这是一次前向传播的完整性检查。

### 第 4 步：采样

实现 greedy、temperature、top-k、top-p、min-p。对固定提示运行并比较输出。一个采样函数大约 10 行代码。

## 使用

PyTorch，2026 年惯用写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 运行前向传播，取出最终位置的 logits，采样下一个标记，附加上去，然后重复。每个生产级别的 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都实现了相同的循环并进行了大量优化 —— 批量预填充、连续批处理、KV 缓存分页、投机性解码。

**GPT vs BERT，一句总结：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失决定模型是否能生成。

## 发布

见 `outputs/skill-sampling-tuner.md`。该技能为新的生成任务选择采样参数，并在需要确定性解码时发出标记。

## 练习

1. **简单。** 运行 `code/main.py` 并验证因果注意力矩阵在 softmax 后是下三角的。抽查：第 3 行（row 3）应该只有列 0–3 有非零权重。
2. **中等。** 实现宽度为 4 的束搜索（beam search）。在 10 个短提示上比较 beam-4 与 greedy 的困惑度（perplexity）。束搜索总是更好吗？（提示：通常在翻译任务中是，但在开放式聊天中未必。）
3. **困难。** 实现投机性解码：使用一个小的 2 层模型作为草稿，6 层模型作为验证器。在 100 次长度为 64 的补全上测量时钟时间加速。确认输出与验证器的 greedy 输出匹配。

## 术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|---------|
| 因果掩码 (Causal mask) | "三角形" | 在注意力分数上加的上三角 `-inf` 矩阵，使得位置 `i` 只能看到位置 `≤ i`。 |
| 下一标记预测 (Next-token prediction) | "损失" | 模型分布与每个位置真实下一个标记的交叉熵。 |
| 自回归 (Autoregressive) | "逐个生成" | 将输出反馈为输入；训练时可并行，生成时不可并行。 |
| Logits | "softmax 前的分数" | LM 头的原始输出，采样在这些上进行。 |
| Temperature | "创造力旋钮" | 将 logits 除以 T；T→0 = 贪心，T→∞ = 均匀。 |
| Top-p | "Nucleus 采样" | 截断分布到累计概率 ≥ p 的最小集合；从剩下的集合中采样。 |
| Min-p | "比 top-p 更好" | 保留 `p ≥ min_p × max_p` 的标记；根据分布尖锐程度自适应截断。 |
| Speculative decoding | "草稿 + 验证" | 低成本模型提出 N 个标记；大模型并行验证。 |
| Teacher forcing | "训练技巧" | 训练时喂入真实的上一个标记，而不是模型的预测。序列到序列的 LM 的标准做法。 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 与上下文学习。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机性解码论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 典型的因果 LM 参考代码。