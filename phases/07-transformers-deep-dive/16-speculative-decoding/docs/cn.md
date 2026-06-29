# 投机性解码 — 起草、验证、重复

> 自回归解码是串行的。每个 token 都要等前一个。投机性解码打破了这条链：一个廉价模型起草 N 个 token，昂贵模型一次性验证这 N 个。当起草正确时，你只需为 N 次生成支付一次大型前向计算的代价。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)  
**Time:** ~60 分钟

## 问题

在 H100 上，采样一个 token 的 70B LLM 大约需要 ~30 ms。一个 3B 的草稿模型大约需要 ~3 ms。如果让 3B 草稿模型向前起草 5 个 token，然后仅运行 70B 一次来验证这 5 个 token，总耗时为 `5×3 + 30 = 45 ms`（最多可接受 5 个 token）——而直接顺序生成需要 `5×30 = 150 ms`。这就是投机性解码的全部卖点：用一点额外的 GPU 内存（草稿模型）换来 2–4× 更低的解码延迟。

关键在于必须保持分布不变。Leviathan 等人（2023）和 Chen 等人同时提出的投机性采样保证了输出序列在分布上**与大型模型单独生成时完全一致**。没有质量折衷。只是更快。

到 2026 年，四类草稿-验证器配对主导推理实现：

1. **Vanilla speculative（Leviathan 2023）。** 单独的草稿模型（例如 Llama 3 1B）+ 验证器（例如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 验证器上附加多个解码头并行预测位置 `t+1..t+k`。无需单独草稿模型。
3. **EAGLE 家族（Li 2024, 2025）。** 轻量草稿重用验证器的隐藏状态；相比 vanilla 有更高的接受率；典型 3–4× 加速。
4. **Lookahead decoding（Fu 2024）。** 雅可比迭代；根本不需要草稿模型。自我投机。小众但无依赖。

到 2026 年，所有生产级推理栈默认都集成了投机性解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 至少支持 vanilla + EAGLE-2。

## 概念

### 核心算法

给定一个验证器 `M_q` 和更便宜的草稿 `M_p`：

1. 令 `x_1..x_k` 为已解码的前缀。
2. **起草**：使用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，并记录草稿概率 `p_1..p_N`。
3. **并行验证**：在 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 上仅运行 `M_q` 一次，得到位置 `k+1..k+N+1` 的验证器概率 `q_1..q_{N+1}`。
4. **从左到右对每个草稿 token 做接受/拒绝**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 首次被拒绝：从“残差”分布 `(q_j - p_j)_+`（归一化后）中采样 `t_j`。`j` 之后的所有草稿都被丢弃。
6. 如果所有 `N` 都被接受：从 `q_{N+1}` 中再采样一个额外 token `t_{N+1}`（免费的奖励 token）。

残差分布技巧是保持输出分布与 `M_q` 直接采样等价的数学洞见。

### 什么决定加速比

令 `α` = 每个草稿 token 的期望接受率。令 `c` = 草稿到验证器的代价比。每一步：

- 朴素生成每个 token 都要一次大模型调用。
- 投机性生成在 `α` 较高时，每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 只需一次大模型调用。

经验法则：在 `α = 0.75` 且 `N = 5` 时，大模型调用减少约 3×。草稿成本是 5× 的便宜。总体时钟耗时降约 2.5×。

α 取决于：

- 草稿对验证器的近似程度。相同家族 / 相同训练数据会显著提升 α。
- 解码策略。贪心草稿对贪心验证器：高 α。高温采样：难以匹配；接受率下降。
- 任务类型。代码与结构化输出更可预测，接受率更高；自由创作接受率更低。

### Medusa — 无需独立草稿模型的起草

Medusa 用验证器上的额外输出头替代了草稿模型。在位置 `t`：

```
共享主干 → 隐状态 h_t
    ├── head_0：预测 t+1 的 token（标准 LM 头）
    ├── head_1：预测 t+2 的 token
    ├── head_2：预测 t+3 的 token
    ├── head_3：预测 t+4 的 token
```

每个头输出自己的 logits。在推理时，你从每个头采样以得到候选序列，然后使用一种树形注意力方案在一次前向中验证所有候选续写。

优点：无需第二个模型。缺点：增加可训练参数；需要一个监督微调阶段（~1B token）；与一个优秀的外部草稿相比，接受率略低。

### EAGLE — 通过重用隐藏状态得到更好的草稿

EAGLE-1/2/3（Li 等，2024–2025）将草稿模型做成一个很小的 transformer（通常 1 层），它以验证器的最后一层隐藏状态作为输入。因为草稿能看到验证器的特征表示，它的预测与验证器输出分布高度相关。接受率从 ~0.6（vanilla）提升到 0.85+。

EAGLE-3（2025）加入了对候选续写的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 与 Qwen 3 的默认投机性路径。

### KV 缓存的处理

验证在一次前向中将 `N` 个草稿 token 输入验证器。这会将验证器的 KV 缓存扩展 `N` 条目。如果某些草稿被拒绝，你必须将缓存回滚到已接受的前缀长度。

生产实现（如 vLLM 的 `--speculative-model`，TensorRT-LLM 的 LookaheadDecoder）通过临时 KV 缓存处理：先写入，接受时再提交。这不是概念上难，但确实有点繁琐。

## 实现

参见 `code/main.py`。我们实现核心的投机性采样算法（拒绝步骤 + 残差分布），采用：

- 一个“大模型”，它在一个手写分布上做确定性 softmax（以便我们可以解析地验证接受率的数学）。
- 一个作为“大模型扰动”的“草稿模型”。
- 一个接受/拒绝循环，产生与直接采样相同的边缘分布。

### 第 1 步：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是验证器对该草稿 token 的概率。`p_prob` 是草稿模型的概率。Leviathan 定理表明，这个伯努利决策，随后在拒绝时从残差中采样，能精确地保持验证器的分布。

### 第 2 步：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素用 `q` 减去 `p`，将负值截断为零，再归一化。在任何拒绝时从这个分布采样。

### 第 3 步：一次投机性步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个被接受 → 一个奖励 → 在一次验证器前向中产生六个 token。

### 第 4 步：测量接受率

在不同草稿质量水平下运行 10,000 次投机性步骤。绘制接受率与草稿与验证器分布之间 KL 散度的关系。你应能看到干净的单调关系。

### 第 5 步：验证分布等价性

经验上：投机性循环产生的 token 直方图应当与直接从验证器采样产生的直方图匹配。这就是 Leviathan 定理的实证。卡方检验能在抽样误差范围内确认（chi-square p > 0.05）。

## 使用方法

生产示例：

```bash
# vLLM 使用 EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM 使用 vanilla 草稿模型
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

到 2026 年中，TensorRT-LLM 在 Medusa 路径上最快。`faster-whisper` 为 Whisper-large 包装了投机性解码并使用一个小型草稿。

**选择草稿策略：**

| Strategy | When to pick | Speedup |
|----------|--------------|---------|
| Vanilla draft (1B/3B Llama family) | 快速原型，无需训练 | 1.8–2.3× |
| Medusa heads | 你可以微调验证器 | 2–3× |
| EAGLE-2 / 3 | 生产环境，极致加速 | 3–4× |
| Lookahead | 无草稿、无训练、无额外参数 | 1.3–1.6× |

**什么时候不要使用投机性解码：**

- 单序列生成 1–5 个 token。开销占优。
- 高温/极具创造性的采样（α 下降）。
- 内存受限的部署（草稿模型会增加显存）。

## 上线部署

参见 `outputs/skill-spec-decode-picker.md`。该技能会为新的推理工作负载选择一种投机性解码策略（vanilla / Medusa / EAGLE / lookahead）和调优参数（N、草稿温度等）。

## 练习

1. **简单。** 运行 `code/main.py`。在 50,000 个 token 上确认投机性 token 分布与直接从验证器采样的分布匹配（卡方检验 p > 0.05）。
2. **中等。** 对 `α = 0.5, 0.7, 0.85` 绘制随 `N` 变化的加速比（每次验证的大模型 token 数）。找出每个 α 的最优 `N`。（提示：期望每次验证产生的 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个微型 Medusa：以第 14 课的 capstone GPT 为基础，增加 3 个额外的 LM 头来预测位置 t+2, t+3, t+4。用 joint multi-head loss 在 tinyshakespeare 上训练。将接受率与通过截断同一模型得到的 vanilla 草稿比较。
4. **困难。** 实现回滚：以 10-token 前缀 KV 缓存开始，输入 5 个草稿 token，模拟在位置 3 的拒绝。验证下次迭代时你的缓存读取确实匹配“前缀 + 首 2 个被接受的草稿”。

## 关键术语

| 术语 | 大家如何说 | 实际含义 |
|------|------------|----------|
| Draft model | “便宜的那个” | 提出候选 token 的较小模型；通常比验证器便宜 10–50×。 |
| Verifier | “大的那个” | 我们要保持其分布的目标模型；每次投机性步骤运行一次。 |
| Acceptance rate (α) | “草稿有多常正确” | 验证器接受草稿的每 token 概率。典型 0.7–0.9。 |
| Residual distribution | “拒绝时的回退” | `(q - p)_+` 归一化；在拒绝时从该分布采样以保持验证器分布。 |
| Bonus token | “免费的那个” | 所有 N 个草稿都被接受时，从验证器的下一步分布再采样一个 token。 |
| Medusa | “无草稿的投机” | 验证器上多个 LM 头并行预测位置 t+1..t+k。 |
| EAGLE | “基于隐藏态的草稿” | 条件于验证器最后一层隐藏状态的微小 transformer 草稿。 |
| Lookahead decoding | “雅可比迭代” | 使用定点迭代的自我投机；无需草稿模型。 |
| Tree attention | “一次验证多候选” | 在一次前向中考虑多个草稿续写的分支验证方法。 |
| KV rollback | “撤销被拒草稿” | 使用临时 KV 缓存；接受时提交，拒绝时丢弃。 |

## 进一步阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法与等价定理。  
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 并行提出；清晰的伯努利拒绝证明。  
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；树形注意力验证。  
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；基于隐藏态的草稿。  
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；动态树深度。  
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。  
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead，无草稿方法。  
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 生产级的权威参考，四种策略全接入。  
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的参考实现代码。