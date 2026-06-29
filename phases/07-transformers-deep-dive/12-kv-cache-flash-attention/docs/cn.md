# KV Cache, Flash Attention & 推理优化

> 训练是并行且受 FLOP 限制的。推理是串行且受内存限制的。瓶颈不同，技巧也不同。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 7 · 02（自注意力）, Phase 7 · 05（完整 Transformer）, Phase 7 · 07（GPT）
**Time:** ~75 分钟

## 问题

一个朴素的自回归解码器在生成 N 个 token 时做 O(N²) 的工作：在每一步它都会对完整前缀重新计算 attention。对于 4K token 的回复那是 1600 万次 attention 操作，其中大多数是冗余的。前缀中每个 token 的隐藏态一旦计算出来就是确定的——你只需要用新 token 的 query 去对之前缓存的 keys 和 values 做 attention。

此外，attention 本身会移动大量数据。标准 attention 会物化一个 N×N 的分数矩阵、N×d 的 softmax 输出、N×d 的最终输出 —— 对 HBM 来说读写过多。当 N≥2K 时，attention 在成为 FLOP 限制之前就变成了内存限制。经典的 attention kernel 在现代 GPU 上利用率往往低 4–10×。

两项来自 Dao 等人的优化把前沿推理从“慢”推进到“快”：

1. **KV cache。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的 attention 只需一个 query 去对缓存的 keys 进行计算。推理将每步从 `O(N²)` 降到 `O(N)`。
2. **Flash Attention。** 将 attention 计算切块，使得完整的 N×N 矩阵永远不落到 HBM。所有的 softmax + matmul 都在 SRAM 中完成。在 A100 上有 2–4× 的实测加速；在 H100（使用 FP8）上是 5–10×。

到 2026 年，这两者都已普及。每一个生产推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都把它们当作默认假设。每一个前沿模型都启用了 Flash Attention。

## 概念

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV cache 数学

每层解码器、每个 token、每个 head：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K 和 V
```

对一个 7B 模型，32 层，32 个 heads，d_head=128，fp16：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

对于 Llama 3 70B（80 层，d_head=128，GQA 使用 8 个 KV heads）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

这 10 GB 就是为什么 Llama 3 70B 在 128K 上下文下，在 batch size 1 时，仅 KV cache 就需要接近 40 GB A100 的大部分内存的原因。

**GQA 是 KV-cache 的胜利点。** 具有 64 个 head 的 MHA 会是 32 GB。MLA 还可以进一步压缩。

拖动尺寸看看缓存大小如何变化。把序列长度或 batch 推高，观察它如何快速超出单卡容量：

```figure
kv-cache-sizer
```

### Flash Attention —— 切块技巧

标准 attention：

```
S = Q @ K^T          (HBM 读取, N×N, HBM 写入)
P = softmax(S)       (HBM 读取, HBM 写入)
O = P @ V            (HBM 读取, HBM 写入)
```

三次 HBM 往返。在 H100 上，HBM 带宽约为 3 TB/s；SRAM 为 30 TB/s。每一次 HBM 往返相对于片上存储都会慢一个数量级（约 10×）。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个 tile 只有一次 HBM 往返。总内存占用从 `O(N²)` 降到 `O(N)`。反向传播通过在前向时重算部分值而不是全部存储，从而进一步节省内存。

**数值技巧。** 运行时 softmax 在切块之间维护 (max, sum)，所以最终归一化是精确的。不是近似 —— Flash Attention 与标准 attention 计算出位相同（modulo fp16 的非结合性）。

**版本演进：**

| Version | Year | Key change | Speedup on reference hardware |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Tiled SRAM kernel | 在 A100 上 2× |
| Flash 2 | 2023 | 更好的并行性，因果优先的排序 | 在 A100 上 3× |
| Flash 3 | 2024 | Hopper 异步性，FP8 | 在 H100（~740 TFLOPs FP16）上 1.5–2× |
| Flash 4 | 2026 | Blackwell 五阶段流水线，软件 exp2 | 面向推理（初期仅前向） |

Flash 4 发布时仅支持前向。训练仍使用 Flash 3。Flash 4 的 GQA 和可变长度支持尚在进行中（2026 年中期）。

### 投机性解码 —— 另一个延迟改进

便宜的模型提出 N 个 token。大模型并行验证所有 N 个。如果验证接受了 k 个 token，你就只为 k 个生成付出一次大模型的前向代价。代码和文章通常的 k=3–5。

2026 年的默认做法：
- **EAGLE 2 / Medusa。** 集成的草稿头（draft heads）共享验证器的隐藏态。无质量损失情况下 2–3× 的加速。
- **使用草稿模型的投机性解码。** 在消费级硬件上 2–4× 的加速。
- **前瞻解码（Lookahead decoding）。** Jacobi 迭代；不需要草稿模型。适用场景有限但零额外成本。

### 连续批处理（Continuous batching）

经典的批量推理：等待最慢的序列完成，然后再开始新一批。短响应完成得早会浪费 GPU 资源。

连续批处理（最早在 Orca 中发布，现已在 vLLM、TensorRT-LLM、SGLang 中实现）：当有旧请求完成时立即将新请求交换进批次。对于典型的聊天工作负载能带来 5–10× 的吞吐率提升。

### PagedAttention —— 把 KV cache 当作虚拟内存

vLLM 的头条特性。KV cache 以 16-token 为块分配；页表映射逻辑位置到物理块。允许在并行样本间共享 KV（beam search、并行采样）、为提示词缓存热插前缀、并碎片整理内存。相对于朴素的连续分配，吞吐率提升约 4×。

```figure
flash-attention-memory
```

## 动手构建

见 `code/main.py`。我们实现：

1. 一个朴素的 `O(N²)` 增量解码器。
2. 一个 `O(N)` 的 KV-cached 解码器。
3. 一个模拟 Flash Attention 运行最大值算法的切块 softmax。

### 第 1 步：KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

很简单：在每层、每个 head 的列表中按 token 不断增长 K、V 向量。

### 第 2 步：切块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """类似 Flash-attention 的 softmax(qK^T)V，使用运行时最大值/和的算法。"""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

与一次性计算的 `softmax(qK) V` 输出位相同，但任意时刻的工作集是 `tile × d_head`，而不是完整的 `N × d_head`。

### 第 3 步：在 100-token 生成上比较朴素与缓存解码

统计 attention 操作数。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码会打印两者。

## 使用方法

```python
# HuggingFace transformers 会在仅解码器的 generate() 上自动启用 KV cache。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # 在 Hopper 上使用 FA3
    torch_dtype="bfloat16",
)
# generate() 会自动使用 KV cache
```

vLLM 生产示例：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是 2026 年的一大胜利 —— 相同的系统提示、少样本示例或长文档上下文可以在多次调用间复用 KV。对于重复调用工具提示的 agent 工作负载，前缀缓存常常带来 5× 的吞吐提升。

## 部署建议

见 `outputs/skill-inference-optimizer.md`。该文档为新的推理部署选择 attention 实现、KV cache 策略、量化和投机性解码策略。

## 练习

1. 简单。运行 `code/main.py`。确认朴素解码器和缓存解码器产生相同的输出；注意操作计数差异。
2. 中等。实现前缀缓存：给定提示 P 和多个完成结果，先对 P 做一次前向填充 KV cache，然后对每个完成分支展开。测量相对于对每个完成都重新编码 P 的加速比。
3. 困难。实现一个玩具版的 PagedAttention：KV cache 以固定的 16-token 块分配并维护一个空闲链表。当一个序列结束时，把它的块归还到内存池。模拟 1,000 个不同长度的聊天完成，比较碎片化程度与连续分配的差异。

## 术语要点

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| KV cache | "The trick that makes decoding fast" | 存储每个前缀 token 的 K 和 V；新的 queries 对它们进行 attention 而不是重算。 |
| HBM | "GPU main memory" | 高带宽内存（High Bandwidth Memory）；H100 上 ~80 GB，B200 上 ~192 GB。带宽约 ~3 TB/s。 |
| SRAM | "On-chip memory" | 片上快速内存（每个 SM，大约 H100 上每个 SM ~256 KB）。带宽约 ~30 TB/s。 |
| Flash Attention | "Tiled attention kernel" | 在不把 N×N 矩阵物化到 HBM 的情况下计算 attention 的内核实现。 |
| Continuous batching | "No-wait batching" | 将完成的序列换出并立即换入新的序列，而不清空批次。 |
| PagedAttention | "vLLM's headline" | KV cache 以固定大小块分配并通过页表映射；消除了碎片化问题。 |
| Prefix caching | "Reuse long prompts" | 在请求间缓存共享前缀的 KV；是 agent 场景下的主要成本削减手段。 |
| Speculative decoding | "Draft + verify" | 廉价的草稿模型提出 token；大模型一次性并行验证 k 个 token。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 五阶段流水线和软件 exp2 技巧；阅读仓库 README 了解仅前向发布的注意事项。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机性解码。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2，有关集成草稿方法的论文。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 与 EAGLE 并列提及的 Medusa 方法。
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 关于 16-token 块和页表设计的权威深入解析。