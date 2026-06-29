# Mixture of Experts (MoE)

> 一个稠密的 70B Transformer 会对每个 token 激活所有参数。一个 671B 的 MoE 每个 token 只激活 37B 的参数，却在每个基准上都胜出。稀疏性是本十年最重要的扩展思路。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 7 · 05 (Full Transformer), 阶段 7 · 07 (GPT)
**Time:** ~45 分钟

## 问题

稠密 transformer 在推理时的 FLOPs 等于其参数量（前向传播大约乘以 2）。扩展稠密模型的话，每个 token 都要付出全部计算代价。到 2024 年，前沿研究在计算资源上遇到瓶颈：要显著更聪明，你需要对每个 token 指数级更多的 FLOPs。

Mixture of Experts 打破了这种关联。将每个 FFN 替换为 `E` 个独立专家 + 一个 router（路由器），它为每个 token 选择 `k` 个专家。总参数量 = `E × FFN_size`。每个 token 的激活参数 = `k × FFN_size`。2026 年的典型配置：`E=256`，`k=8`。存储随 `E` 缩放，计算随 `k` 缩放。

到 2026 年，最前沿几乎完全是 MoE：DeepSeek-V3（671B 总量 / 37B 激活）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，前十名的开源模型全部是 MoE。

## 概念

![MoE 层：路由器为每个 token 从 E 个专家中选择 k 个](../assets/moe.svg)

### FFN 替换

稠密 transformer block:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE block:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是独立的 FFN（通常是 SwiGLU）。路由器是一个单层线性层。每个 token 选择自己的 `k` 个专家，得到这些专家输出的加权混合。

### 负载均衡问题

如果路由器把 90% 的 token 分配给专家 3，其他专家就会被饿死。已经尝试过三种修正方法：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）。添加一个与专家使用率方差成比例的惩罚项。有效，但会引入超参数并产生第二条梯度信号。
2. **专家容量 + token 丢弃**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出的 token 会跳过该层。会损害质量。
3. **无辅助损失的均衡**（DeepSeek-V3）。为每个专家增加一个可学习的偏置项，影响路由器的 top-k 选择。偏置在训练损失之外更新，对主目标没有惩罚。2024 年的关键突破。

DeepSeek-V3 的方法：在每次训练步骤后，检查每个专家的使用率是否高于或低于目标。将偏置按 `±γ` 进行微调。选择使用 `scores + bias`。用于门控的专家概率仍使用原始的 `scores` 不变。路由与表达解耦。

### 共享专家

DeepSeek-V2/V3 还将专家分为 *shared*（共享）和 *routed*（路由）。每个 token 都会通过所有共享专家。路由专家通过 top-k 被选中。共享专家捕捉通用知识；路由专家进行专门化。V3 运行 1 个共享专家加上 256 个路由专家中的 top-8。

### 细粒度专家

经典 MoE（GShard、Switch）：每个专家的宽度与完整 FFN 相当。`E` 较小（8–64），`k` 也较小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（大约为 FFN 大小的 1/8）。`E` 很大（256+），`k` 也更大（8+）。总参数量相同，但组合数增长更快。`C(256, 8) = 400 万亿` 种可能的“专家”组合。质量提升而延迟保持不变。

### 成本曲线

每个 token、每层：

| Config | Active params / token | Total params |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 在几乎每个基准上都优于 Llama 3 70B（稠密），同时每个 token 的激活 FLOPs 更少。更多参数 = 更多知识。更多激活 FLOPs = 每个 token 更多计算。MoE 将两者解耦。

### 弯路：内存

所有专家的权重无论是否被激活都需要驻留在 GPU 上。一个 671B 的模型需要约 1.3 TB 的 VRAM（fp16 权重）。前沿 MoE 部署需要专家并行性 —— 在 GPU 之间分片专家，并在网络上路由 token。延迟主要由 all-to-all 通信主导，而不是矩阵乘法。

## 构建

参见 `code/main.py`。一个仅用标准库实现的紧凑 MoE 层，包含：

- `n_experts=8` 个类似 SwiGLU 的专家（为演示每个专家用一个线性层）
- top-k=2 路由
- softmax 归一化的门控权重
- 通过每专家偏置实现的无辅助损失均衡

### 步骤 1：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 对所选专家的原始分数做 softmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，但不影响门控权重。这就是 DeepSeek-V3 的技巧 —— 偏置用来纠正负载不均，而不直接引导模型的预测。

### 步骤 2：对路由器运行 100 个 token

跟踪哪些专家被激活以及频次。在没有偏置的情况下，使用率会偏斜。通过偏置更新循环（对过度使用的专家减 `γ`，对使用不足的加 `γ`），使用率会在若干次迭代后收敛到接近均匀的分布。

### 步骤 3：参数量比较

打印 MoE 配置的“稠密等价”。DeepSeek-V3 形状示例：256 个路由专家 + 1 个共享专家，8 个激活，d_model=7168。总参数量惊人。激活参数量是稠密 Llama 3 70B 的大约七分之一。

## 使用

HuggingFace 加载示例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年的生产推理：vLLM 原生支持 MoE 路由。SGLang 拥有最快的专家并行路径。两者都会自动处理 top-k 选择和专家并行。

何时选择 MoE：
- 你想以更低的每 token 推理成本获得前沿质量。
- 你拥有 VRAM / 专家并行基础设施。
- 你的工作负载以 token 为主（聊天、代码），而不是以上下文长度为主（长文档）。

何时不该选择 MoE：
- 边缘部署 —— 你需要为所有激活 FLOP 支付完整存储成本。
- 延迟极其敏感的单用户服务 —— 专家路由会增加开销。
- 小模型（<7B） —— MoE 的质量优势通常在计算阈值（约 6B 激活参数）之上才显现。

## 发布

参见 `outputs/skill-moe-configurator.md`。该技能根据参数预算、训练 token 数量和部署目标为新的 MoE 选择 E、k 和共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失的偏置更新如何在 50 次迭代内使专家使用率均衡。
2. **中等。** 用基于哈希的路由器替换学习型路由器（确定性、无学习）。比较质量和负载均衡。为什么学习型路由器更好？
3. **困难。** 实现 GRPO 风格的“rollout-matched routing”（DeepSeek-V3.2 的技巧）：记录推理期间哪些专家被激活，在梯度计算时强制相同的路由。在一个玩具的策略梯度设置中测量效果。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Expert | "One FFN among many" | 一个独立的前馈网络；参数专门用于 FFN 计算的一个稀疏切片。 |
| Router | "The gate" | 一个很小的线性层，对每个 token 与每个专家打分；进行 top-k 选择。 |
| Top-k routing | "k active experts per token" | 每个 token 的 FFN 计算恰好经过 k 个专家，并由门控加权。 |
| Auxiliary loss | "Load-balance penalty" | 一个额外的损失项，用来惩罚专家使用的不均衡。 |
| Auxiliary-loss-free | "DeepSeek-V3's trick" | 仅在路由选择上通过每专家偏置实现均衡；没有额外的梯度。 |
| Shared expert | "Always on" | 每个 token 都会经过的额外专家；捕捉通用知识。 |
| Expert parallelism | "Shard by expert" | 将不同专家分配到不同 GPU；在网络上路由 token。 |
| Sparsity | "Active params < total params" | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3 约为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 该思路的来源。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典的 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 无辅助损失 MoE + MTP 的技术报告。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基于偏置的均衡方法论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本课路由器使用的细粒度 + 共享专家拆分。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 早期的共享专家论文。