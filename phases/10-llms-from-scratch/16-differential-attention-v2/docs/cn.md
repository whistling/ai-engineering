# Differential Attention (V2)

> Softmax attention 在每个非匹配 token 上分配少量概率。对 100k+ 的 token 来说，这些噪声会累积并淹没信号。Differential Transformer（Ye et al., ICLR 2025）通过将注意力计算为两个 softmax 的差来修正这一点，从而减去共享的噪声底。DIFF V2（Microsoft，2026 年 1 月）是面向生产栈的重写：解码延迟与基线 Transformer 匹配，无需自定义内核，兼容 FlashAttention。本课从 V1 到 V2 全流程讲解，并提供一个可在标准库 Python 中运行的差分操作玩具实现。

**Type:** 构建  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 7 · 02 (self-attention -> 自注意力), Phase 7 · 15 (attention variants -> 注意力变体), Phase 10 · 14 (architecture walkthrough -> 架构讲解)  
**Time:** ~60 分钟

## 学习目标

- 精确说明为何 softmax 注意力存在噪声底（noise floor），以及它为何随上下文长度增长。
- 推导 differential attention 的公式，并解释为何相减可以抵消共享噪声分布而保留信号。
- 讲解 V1 到 V2 的差异：哪些更快、哪些更简单、哪些更稳定，以及为何每个修改对生产预训练是必要的。
- 用纯 Python 从头实现 differential attention，并在合成的信号 + 噪声查询上实证验证噪声消除特性。

## 问题陈述

标准的 softmax 注意力在数学上有一个属性，在大规模场景下会成为实际问题。对于查询 `q`，注意力权重为 `softmax(qK^T / sqrt(d))`。Softmax 永远不会给出精确的零——每个非匹配 token 都会得到一些正质量。那个残余质量就是噪声，并且随上下文长度线性放大。在 128k token 下，即便每个非匹配 token 仅得到 0.001% 的概率，127,999 个这样的 token 累加起来也约占总量的 12%。模型必须学会在一个随上下文增长的噪声底上进行路由。

经验上，这表现为注意力头干扰：长上下文 RAG 中的虚构引用（hallucinated citations）、100k token 检索任务中的“mid-context 丢失”（lost-in-the-middle）失败，以及在超过 32k 的“干草堆中找针”基准上的微妙精度下降。Differential Transformer 论文（arXiv:2410.05258, ICLR 2025）测量到差距：DIFF Transformer 在困惑度（perplexity）、长上下文准确率和虚构率上优于相同规模的基线模型。

DIFF V1 有三个问题使其难以进入前沿预训练流水线。它的 value 缓存（value cache）在每次解码步必须加载两次；它依赖自定义 CUDA 内核，破坏了 FlashAttention 的兼容性；并且其按头的 RMSNorm 在 70B+ 规模训练后期会导致不稳定。DIFF V2（Microsoft unilm 博客，2026 年 1 月）修复了这三点。本课同时讲解两个版本，构建差分算子，并在玩具查询上基准噪声消除效果。

## 概念

### softmax 的噪声底

对于查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 会是零。如果 `k_i` 与 `q` 完全无关，分数 `q . k_i` 也不是恒为 0 —— 它围绕 0 波动，方差约为 `||q||^2 / d`。经过 softmax 归一化后，每个无关 token 仍然会以 `O(1/N)` 的量级对加权和做出贡献。所有无关 token 的总体贡献是 `O((N-1)/N) = O(1)` —— 并非一个小量。

模型想要的是类似硬 top-k 的行为：对匹配 token 给出高权重，对其它位置近似为零。Softmax 太平滑，无法直接做到这一点。

### 差分思想

将每个头的 Q 和 K 投影拆成两部分：Q = (Q_1, Q_2)，K = (K_1, K_2)。计算两个注意力映射：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出为：

```
DiffAttn = (A_1 - lambda * A_2) V
```

相减会抵消两个映射共享的噪声分布。如果两个分支在 127k 个无关 token 上有大致均匀的权重（在随机初始化时会如此），这些部分会相互抵消。信号——即在少数实际相关 token 上的尖峰权重——只有在两个映射以相同幅度同时出现时才会被抵消，而一旦模型训练，这种完全相同的峰值不会同时出现在两个分支上。

`lambda` 是按头可学习的标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以为负值。`lambda_init` 默认为一个小的正数，比如 0.8。

### 为什么这像按头噪声消除

把两个嘈杂的麦克风录音想象成两个分支。两者都录到说话者的声音加上相关的背景噪声。把一个信号从另一个中相减，共同的噪声就会被消掉。说话者的声音会保留，因为两个信号在相位或幅度上有足够差异而不会完全抵消。按头的 `lambda` 正是学习这种平衡。

### V1 与 V2：差别

V1 为了保持参数量与基线 Transformer 相同，将每个头的维度减半以获得两个查询分支。这损失了头的表达能力，更重要的是，每个头的 value cache 大小也减半。解码时必须为每个步骤加载 value cache 两次（每个 softmax 分支一次），结果是尽管参数量匹配，解码仍比基线慢。

V2 将查询头数翻倍并保持 KV 头数不变（借用上投影参数）。头维度保持与基线相同。相减后，将多出的维度投影回去以匹配基线 Transformer 的 O_W 投影。这样同时实现三件事：

1. 解码速度与基线匹配（KV 缓存只需加载一次）。
2. FlashAttention 无需修改即可运行（无自定义内核）。
3. 解码时的算术强度（arithmetic intensity）上升（每次从 HBM 加载更多计算量）。

V2 还移除了 V1 中用于稳定相减的按头 RMSNorm。在 70B 级别的预训练尺度上，那个 RMSNorm 导致训练后期不稳定。V2 用更简单的初始化方案替代它，既保持训练稳定性又去掉了额外模块。

### 何时采用它

| 工作负载 | 益处 |
|----------|------|
| 长上下文 RAG (64k+) | 更清晰的注意力图，减少虚构引用 |
| 针眼寻针基准（needle-in-haystack） | 在 32k 以上有显著精度提升 |
| 多文档问答 | 降低跨文档干扰 |
| 8k 的代码补全 | 较小增益，不值得架构改动 |
| 短对话 (< 4k) | 与基线基本无差异 |

收益随上下文长度增长。在 4k token 时噪声底足够小，标准注意力已经足够。在 128k 时它会对性能造成明显伤害。

### 与其他 2026 年技术的兼容性

| 功能 | 与 DIFF V2 兼容？ |
|------|------------------|
| GQA | 是（V2 增加的是 Q 头，不是 KV 头） |
| MLA (DeepSeek) | 原则上兼容，但暂无联合发表的论文 |
| MoE | 是（注意力独立于 MLP 模块） |
| RoPE | 是（不变） |
| YaRN / 长上下文扩展 | 是（正是 DIFF 最有用的场景） |
| FlashAttention | V2 中兼容（V1 不兼容） |
| Speculative decoding | 是（注意力改动对 spec-decode 循环不可见） |

```figure
differential-attention
```

## 构建实现

`code/main.py` 在纯 Python 中实现了 differential attention。一个已知信号 + 噪声结构的玩具查询允许你直接测量噪声消除比率。

### 步骤 1：标准 softmax 注意力

使用标准库矩阵操作：列表的列表、手动矩阵乘、带数值稳定性（减去最大值）的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步骤 2：将 Q、K 拆成两半

V1 风格：将头维度减半。V2 风格：保持头维度并把头数翻倍。玩具实现采用 V1 的 bookkeeping（便于教学）——数学等价，只有记账不同。

### 步骤 3：两个 softmax 分支 + 相减

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以为负。这没问题——value cache 仍然处理有符号的贡献。随后的 V 投影会吸收符号。

### 步骤 4：噪声消除度量

构造一个长度为 1024 的合成序列。在已知位置放入信号 token，其余位置填充噪声。计算 (a) 标准 softmax 注意力在信号位置上的权重和 (b) differential attention 在信号位置上的权重。测量各自的信号-噪声比（signal-to-noise）。DIFF 注意力通常能把信号-噪声比提升 3x–10x，具体取决于两个分支被训练为不同的程度。

### 步骤 5：V1 与 V2 的参数对账

给定配置（hidden=4096, heads=32, d_head=128），打印：

- 基线 Transformer：Q、K、V 各自大小为 `hidden * hidden`，MLP 为 `4 * hidden`。
- DIFF V1：Q、K 各为 `hidden * hidden`，V 为 `hidden * hidden`（不变），头维在内部减半。增加按头的 `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q 大小为 `2 * hidden * hidden`，K 大小为 `hidden * hidden`，V 大小为 `hidden * hidden`。在 O_W 前把多出的维度投影回去。增加相同的 `lambda` 参数。

玩具实现会测量 V2 的额外参数开销（大约每个注意力块增加 `hidden * hidden`），并打印出来。

## 使用场景

截至 2026 年 4 月，DIFF V2 尚未在每个生产推理服务器中全面部署，但在 vLLM 和 SGLang 中的集成工作正在进行中。同时该模式已出现在：

- Microsoft 内部的长上下文生产模型中。
- 多个开源模型训练复现中，目标上下文长度 256k+。
- 将 DIFF 注意力与滑动窗口注意力在交替层中结合的混合架构。

在 2026 年你会考虑使用它的场景：

- 从头训练一个目标为 64k+ 有效上下文的新模型。从一开始就加入 differential attention；后期重新训练代价很高。
- 微调一个长上下文模型，当“mid-context 丢失”主导你的评估时。对 Q 投影做 LoRA 可以近似 DIFF 结构。

不应该使用它的场景：

- 你正在服务一个已训练且在长上下文上表现稳定的密集模型。对现有权重进行重训成本通常难以收回。
- 你的上下文长度始终低于 16k。噪声底可忽略。

## 部署产出物

本课会生成 `outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、虚构（hallucination）轮廓和训练预算，它会为在新预训练任务或 LoRA 微调中加入 differential attention 制定集成计划。

## 练习

1. 运行 `code/main.py`。验证在合成查询上报告的差分注意力的信噪比高于标准 softmax 注意力。改变噪声振幅并展示标准注意力变得不可用的交叉点（crossover point）。

2. 计算从基线到 DIFF V1、以及从基线到 DIFF V2 在 7B 级模型（hidden=4096, heads=32, d_head=128, layers=32）下的参数差异。展示哪些组件参数增加了、哪些保持不变。

3. 阅读 DIFF V1 论文（arXiv:2410.05258）的第 3 节和 DIFF V2 Hugging Face 博客的第 2 节。用两句话解释为什么 V1 需要按头的 RMSNorm，以及为什么 V2 可以移除它而不导致训练发散。

4. 实现消融实验：计算 `lambda = 0`（纯第一分支 softmax）和 `lambda = 1`（完全相减）下的 differential attention。在合成查询上测量信噪比随 `lambda` 扫描的变化，找出使信噪比最大的 `lambda`。

5. 将玩具扩展到 GQA + DIFF V2。选取 8 个 KV 头和 32 个 Q 头。展示 KV 缓存大小与具有相同 (8, 32) 配置的基线 GQA 模型相匹配。

## 术语表

| 术语 | 人们说的 | 实际含义 |
|------|--------|--------|
| Differential attention | “两个 softmax 相减” | 把 Q、K 拆成两半，计算两个 softmax 映射，用第二个（按 `lambda` 缩放）从第一个中减去，然后乘以 V |
| Noise floor | “softmax 的非零尾部” | softmax 在每个无关 token 上放的 O(1/N) 权重，在长上下文中求和为 O(1) |
| lambda | “相减的比例” | 按头可学习的标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | “ICLR 2025 的版本” | 原始 Differential Transformer；为保持参数量将头维减半，需要自定义内核，解码更慢 |
| DIFF V2 | “2026 年 1 月的修复” | 将 Q 头翻倍保持 KV 不变；匹配基线的解码速度并兼容 FlashAttention |
| Per-head RMSNorm | “V1 的稳定器” | V1 在差分后应用的额外归一化；V2 为避免训练后期不稳定而移除它 |
| Signal-to-noise ratio | “注意力浪费程度” | 真正信号位置的权重与无关位置平均权重之比 |
| Lost in the middle | “长上下文失败模式” | 检索任务中位于长上下文中间的文档准确率下降的经验现象 —— DIFF 注意力可以减少这种现象 |
| Arithmetic intensity | “每字节加载的 FLOPs” | V2 通过在一次 KV 加载中双倍查询数提高了该比率；这对内存受限的解码很重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 原始论文，包含噪声消除理论与长上下文消融实验  
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — 面向生产栈的重写，匹配基线解码并兼容 FlashAttention  
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 理论分析为何相减能恢复预训练注意力结构  
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — 参数共享变体  
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF 所基于的基线 Transformer  
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — DIFF 注意力针对的长上下文基准