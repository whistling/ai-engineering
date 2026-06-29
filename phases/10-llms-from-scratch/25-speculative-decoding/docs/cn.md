# 投机性解码与 EAGLE

> 前沿的大型语言模型生成一个标记需要对数十亿参数完成一次完整的前向传递。那次前向传递在很大程度上是过度配置的：大多数情况下，一个更小的模型就能正确猜测接下来的 3-5 个标记，而大模型只需要对这个猜测进行“验证”。当猜测正确时，你以生成一个标记的成本得到了 5 个标记。Speculative decoding（Leviathan 等，2023）将这一思想形式化，EAGLE-3（2025）将接受率提升到约每次验证 ~4.5 个标记 —— 在匹配输出分布的前提下实现 4-5 倍的加速。

**Type:** 构建  
**Languages:** Python（含 numpy）  
**Prerequisites:** Phase 10 Lesson 12（Inference Optimization），Phase 10 Lesson 04（Pre-training Mini-GPT）  
**Time:** ~75 分钟

## 问题

在 H100 上，70B 级模型的解码吞吐率通常为 40–80 标记/秒。每个标记都需要一次完整的前向传递并从 HBM 读取所有模型权重。你不能在不改变输出分布的情况下让模型变小；你也不能在内存范围外增加 batch 大小。除非你能让模型在一次前向传递中输出超过一个标记，否则就被困住了。

自回归生成本看起来是固有的串行：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里存在并发机会。如果你有一个廉价的预测器，它说“接下来的 4 个标记很可能是 [a, b, c, d]”，你就可以在一次**大模型的单次前向传递**中并行验证所有 K+1 个位置，并接受最长的匹配前缀。

Leviathan、Kalai、Matias（2023，"Fast Inference from Transformers via Speculative Decoding"）通过一个巧妙的接受/拒绝规则实现了这一点，并且该规则保持了目标模型的采样分布不变。相同的输出分布下，可以 2–4× 加速。

## 概念

### 双模型设置

- **目标模型** `M_p`：你希望从中采样的大而慢、高质量的模型。分布：`p(x)`。  
- **草稿模型** `M_q`：一个小而快、质量较低的模型。分布：`q(x)`。通常小 5–30×。

每一步：

1. 草稿模型自回归地提出 `K` 个标记：`x_1, x_2, ..., x_K ~ q`。  
2. 目标模型对所有 `K+1` 个位置并行运行一次前向传递，产生每个被提议标记的 `p(x_k)`。  
3. 从左到右根据下述修正的拒绝采样规则对每个标记进行接受/拒绝。接受最长的匹配前缀。  
4. 如果有任何标记被拒绝，则从修正后的分布中采样替代标记并停止。否则从 `p(· | x_1...x_K)` 中再采样一个额外的奖励标记。

如果草稿与目标完全匹配，你将在一次目标前向中得到 K+1 个标记。如果草稿在位置 1 就错了，你只得到 1 个标记。

### 精确性规则

Speculative decoding 在概率分布上是**可证明等价于从 p 采样**的。拒绝规则如下：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示按点的差的正部分。当草稿与目标接近（`p ≈ q`）时，接受率接近 1。当它们不一致时，残差分布被构造为使得整体样本仍然精确遵循 `p`。

贪婪情况：对于 temperature=0 的采样，只需检查 `argmax(p) == x_t`。如果相等则接受；否则输出 `argmax(p)` 并停止。

### 期望加速

如果草稿模型的逐标记接受率为 `α`，每次目标前向产生的期望标记数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

在 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 个标记/前向。一次目标前向的大致成本为 `cost_q * K + cost_p`（K 步草稿加一次目标验证）。如果 `cost_p >> cost_q * K`，吞吐率加速比约为 `3.36×`。

实际上的唯一关键参数是 `α`，完全取决于草稿与目标的一致性。好的草稿至关重要。

### 训练草稿：蒸馏

随机的小模型作为草稿通常很差。标准方案是从目标蒸馏：

1. 选择小的架构（对于 70B 目标约 ~1B，对于 7B 目标约 ~500M）。  
2. 在大量文本语料上运行目标模型；存储其下一标记分布。  
3. 使用 KL 散度将草稿训练成匹配目标的分布（不是对真实标记进行训练）。

结果：`α` 在代码上通常为 0.6–0.8，在自然语言聊天上为 0.7–0.85。生产环境通常能得到 2–3× 的加速。

### EAGLE：树状草稿 + 特征重用

Li、Wei、Zhang、Zhang（2024，"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）观察到标准投机性解码的两个低效点：

1. 草稿进行 K 个串行步骤，每一步都做完整堆栈计算。但草稿可以重用目标最近一次验证计算得到的特征（隐藏状态）——目标已经计算了丰富的表示，而草稿却从零重新推导这些表示。  
2. 草稿输出的是一条线性链。如果草稿能输出一个候选树（每个节点有多个猜测），目标的单次前向可以通过树状注意掩码并行验证多条候选路径，并选择最长被接受的分支。

EAGLE-1 的变动：  
- 草稿输入 = 目标在位置 t 的最终隐藏状态，而非原始标记。  
- 草稿架构 = 1 层 transformer decoder（不是一个独立的小模型）。  
- 输出 = 深度 4–6、每层宽度 4–8 的候选树。

EAGLE-2（2024）添加了动态树拓扑：在草稿不确定的地方树变宽，在自信的地方保持窄，从而在不增加验证成本的情况下提高有效 `α`。

EAGLE-3（Li 等，2025，"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）移除了对顶层特征依赖的固定性，并用一种新的“测试时模拟”损失来训练草稿 —— 草稿在训练时看到与测试时分布匹配的输出，而不是教师强制下的分布。接受率从 0.75（EAGLE-2）提升到 0.82，平均每次验证标记数从 3.0 提升到 4.5。

### 树状注意验证

当草稿输出一棵树时，目标模型通过一个**树状注意掩码**在一次前向中验证它 —— 这是一个以树拓扑为基础的因果掩码，而不是纯粹的链式掩码。每个标记只注意它的祖先节点。验证传递仍然是一次前向，一次矩阵乘；拓扑掩码只需额外几个 KV 条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的第一个标记候选，`c, d, e, f` 是第二个标记候选，所有这六个位置都可以在一次前向中被验证。输出是沿任意被接受路径的最长前缀。

### 何时有效、何时无效

胜出场景：  
- 可预测文本的聊天 / 补全（代码、常见英语、有结构的输出）。`α` 很高。  
- 解码期间存在未被使用的 GPU 计算（内存绑定阶段）。树状草稿利用了可用的 FLOPs。

失败 / 无效场景：  
- 高度随机的输出（高温度下的创意写作）。`α` 会降到接近 `1/|vocab|`。  
- 拥有非常高并发的批量服务 —— 批处理已填满 FLOPs，树状验证空间有限。  
- 目标模型非常小，草稿相对并不小。

生产单位通常报告在聊天上 2–3× 的墙钟加速，在代码生成上 3–5×，在创意写作上接近无增益。

```figure
speculative-decoding
```

## 构建实现

`code/main.py`:

- 一个参考实现 `speculative_decode(target, draft, prompt, K, temperature)`，它实现了精确的拒绝规则并验证其保持目标分布（经验 KL < 0.01 与直接目标采样相比）。  
- 一个 EAGLE 风格的树状起草器，用于构建带有 top-p 分支的深度 K 树。  
- 一个树状注意掩码生成器，用来产生验证器所需的因果模式。  
- 一个接受率测试工具，对一个微小 LM 运行（从 GPT-2-medium 目标蒸馏一个 GPT-2-small）进行验证。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """一次投机性解码回合。返回被接受的标记列表。"""
    # 1. 草稿生成 K 个标记
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. 目标在每个被起草的位置 + 1 个额外位置上计算 p
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. 从左到右接受/拒绝
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. 全部 K 个都被接受 → 从目标分布采样一个奖励（bonus）标记
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

（注意：上面代码中的注释已翻译为中文以便中文开发者理解；代码与变量名保持不变。）

## 使用方法

- **vLLM** 和 **SGLang** 提供一流的投机性解码支持。标志：`--speculative_model`、`--num_speculative_tokens`。EAGLE-2/3 可通过 `--spec_decoding_algorithm eagle` 标志启用。  
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。  
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（用于 Qwen3-32B 的草稿）、`meta-llama/Llama-3.2-1B-Instruct-spec`（用于 70B 的草稿）。  
- **Medusa heads**（Cai 等，2024，"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）：不是使用独立草稿模型，而是在目标模型本身添加 K 个并行预测头。部署更简单，但接受率略低于 EAGLE。

## 部署

本课产出 `outputs/skill-speculative-tuning.md` —— 一个技能（skill），用于分析目标模型的工作负载并选择：草稿模型、K（草稿长度）、树宽、温度，以及何时回退到普通解码。

## 练习

1. 实现精确的拒绝规则并经验验证其正确性。运行 10K 次样本，分别通过 `speculative_decode` 和直接目标采样；计算两种输出分布的 TV 距离（总变差距离）。应 < 0.01。  

2. 计算加速公式。对于固定的 `α` 与 `K`，绘制每次目标前向的期望标记数。为 α ∈ {0.5, 0.7, 0.9} 找到最优的 K。  

3. 训练一个超小草稿。以 124M 的 GPT-2 作为目标，在 1 亿标记上用 KL 损失蒸馏出一个 30M 的 GPT-2 草稿。测量在保留文本上的 `α`。预期：0.6–0.7。  

4. 实现 EAGLE 风格的树状起草器。不要使用链式结构，而是在每个深度输出 top-3 分支。构建树状注意掩码。验证目标接受最长正确分支。  

5. 测量失败模式。在 temperature=1.5（高随机性）下运行投机性解码。展示 `α` 崩溃并且由于草稿开销算法比直接解码更慢。

## 关键词

| 术语 | 常说法 | 实际含义 |
|------|--------|---------|
| Target model | "The big model" | 目标模型：你想从中采样的慢且高质量的模型（p 分布） |
| Draft model | "The speculator" | 草稿模型：小且快速的预测器（q 分布）；通常小 5–30× |
| K / draft length | "Look-ahead" | 草稿每次验证前预测的标记数量 |
| α / acceptance rate | "Hit rate" | 每个标记被草稿提议并被目标接受的概率 |
| Exact rejection rule | "The accept test" | r < p/q 比较规则，用以保持目标分布不变 |
| Residual distribution | "Corrected p-q" | (p - q)+ / ||(p - q)+||_1，在拒绝时用来采样替代的分布 |
| Tree drafting | "Branching speculation" | 树状起草：草稿输出一棵候选树，目标用树结构注意掩码一次性验证 |
| Tree attention mask | "Topological mask" | 拓扑掩码：编码树拓扑的因果掩码，使每个节点只注意其祖先 |
| Medusa heads | "Parallel heads" | 并行头：在目标模型上增加 K 个额外预测头；无需独立草稿模型 |
| EAGLE feature reuse | "Hidden-state draft" | 特征重用：草稿的输入是目标的最后隐藏状态，而非原始标记，从而缩小草稿 |
| Test-time simulation loss | "EAGLE-3 training" | 测试时模拟损失：在训练时让草稿见到与测试时分布匹配的输出，而非教师强制分布 |

## 深入阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则与理论加速分析  
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 的并发表的并行投机采样论文  
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 对独立草稿模型的并行头替代方案  
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 关于特征重用和树状起草的论文  
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑的研究  
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时与测试时分布匹配的工作  
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/lookahead 解码，无需投机器的替代方法