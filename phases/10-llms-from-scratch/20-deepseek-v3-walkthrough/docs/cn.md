# DeepSeek-V3 架构讲解

> Phase 10 · Lesson 14 命名了每个开源模型都会调整的六个架构旋钮。DeepSeek-V3（2024 年 12 月，参数总量 671B，活动参数 37B）不仅调整了这六个旋钮，还增加了四个：Multi-Head Latent Attention、无辅助损失的负载均衡、多标记预测（Multi-Token Prediction）和 DualPipe 训练。本课从上到下解读 DeepSeek-V3 的架构，并从公开配置推导每一项参数计数。课程结束时，你能解释为什么 671B/37B 的比率是合适的押注，以及为什么 MLA + MoE 联合使用在前沿上比任一单独方案更优。

**Type:** 学习  
**Languages:** Python（stdlib，参数计算器）  
**Prerequisites:** Phase 10 · 14（开源模型逐步解读）、Phase 10 · 17（NSA）、Phase 10 · 18（MTP）、Phase 10 · 19（DualPipe）  
**Time:** ~75 分钟

## 学习目标

- 从上到下阅读 DeepSeek-V3 的配置并用「六个 GPT-2 旋钮」加上四个 DeepSeek 特有扩展来解释每个字段。
- 推导总参数量（671B）、活动参数量（37B）以及这两者的组成部分。
- 计算在 128k 上下文下 MLA 的 KV 缓存占用，并与在相同活动参数下采用 GQA 的密集模型做比较。
- 陈述四项 DeepSeek 特有的创新（MLA、MTP、无辅助损失的路由、DualPipe），并指出每项针对架构或训练栈的哪个部分。

## 问题背景

DeepSeek-V3 是首个在架构上与 Llama 系列明显不同的前沿开源模型。Llama 3 405B 是“带有六个旋钮的 GPT-2”。DeepSeek-V3 则是在 GPT-2 的六个旋钮基础上再加四个。阅读 Llama 3 的配置是理解 DeepSeek 配置的热身，但其深层结构——注意力块的形状、路由逻辑、训练时目标——已足够不同，因而需要单独的逐项讲解。

学习它的回报：DeepSeek-V3 的开源权重发布改变了“前沿能力”在开源模型中的含义。它的架构成为许多 2026 年训练任务的蓝图。理解它是任何涉及前沿 LLM 训练或推理角色的入场券。

## 核心不变量（再次确认）

DeepSeek-V3 仍然是自回归模型。它仍然堆叠解码器块。每个块仍有注意力、MLP 和两个 RMSNorm。MLP 仍使用 SwiGLU。仍然使用 RoPE。Pre-norm。权重共享嵌入。与每个 Llama 或 Mistral 的基础相同。

## 转折：用 MLA 代替 GQA

从 Phase 10 · 14 你知道 GQA 通过在 Q 头之间共享 K 和 V 来缩减 KV 缓存。Multi-Head Latent Attention（MLA）更进一步：K 和 V 被压缩到一个共享的低秩潜在表示（`kv_lora_rank`），然后在运行时为每个头解压。KV 缓存仅存储该潜在表示 —— 典型为每个 token 每层 512 个浮点，而非 8 x 128 = 1024 个浮点。

在 128k 上下文下，使用 MLA 的 DeepSeek-V3（每个 token 每层有一个共享潜在 `c^{KV}`；K 和 V 都由该潜在通过上投影导出，并可以合并到后续的 matmul 中）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假设的 GQA 基线（Llama 3 70B 形状，8 个 KV 头，头维 128）将付出：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文下，MLA 比 Llama-3-70B 风格的 GQA 缓存小约 4 倍。

权衡：MLA 在每次注意力计算（每个头）上增加了一个解压步骤。与节省的带宽相比，这部分额外计算很小。对于长上下文推理来说，整体获益明显。

## 路由：无辅助损失的负载均衡

MoE 路由器决定每个 token 由哪几个 top-k 专家处理。简单的路由器会把工作集中到少数专家上，使其他专家闲置。标准修复方法是在损失中加入一个惩罚负载不均衡的辅助项。这会起作用，但会略微降低主任务性能。

DeepSeek-V3 引入了一种无辅助损失的方案。在路由器 logits 上加入每个专家的偏置项（per-expert bias），并在训练中按简单规则进行调整：若某专家 e 过载，则减小 bias_e；若欠载，则增大 bias_e。无需额外的损失项。训练保持“干净”。专家负载保持平衡。

对主损失的影响：可测量值为零。对 MoE 架构的影响：更干净，无需调节辅助损失的超参数。

## MTP：更密集的训练 + 免费草案（draft）

从 Phase 10 · 18 你知道 DeepSeek-V3 增加了 D=1 的 MTP 模块，该模块预测向前两个位置的 token。在推理时，训练好的模块被改造为有 80%+ 接受率的投机性解码草案。在训练时，每个隐藏态受到 D+1 = 2 个目标的监督，提供更密集的信号。

参数：在 671B 主体之外增加 14B。开销：2.1%。

## 训练：DualPipe

从 Phase 10 · 19 你知道 DualPipe 是一种双向流水线，它通过跨节点的 all-to-all 通信来重叠前向和反向的分片。在 DeepSeek-V3 的 2,048-H800 规模下，它大约挽回了 1F1B 管线气泡会损失的 245k GPU 小时。

## 配置字段逐项解释

下列为 DeepSeek-V3 的简化配置：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

解析：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 层使用大小为 18432 的 dense MLP。其余 58 层使用 MoE。
- `num_attention_heads=128`：128 个 query 头。
- `kv_lora_rank=512`：K 和 V 被压缩到这个潜在维度，并在每个头上解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，采用 top-8 路由。
- `shared_experts=1`：在 256 个被路由的专家之外，每个块还有 1 个总是启用的专家。把它当作一个“稠密底座（dense floor）”，确保每个 token 都能获得可靠的处理。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐藏大小。因为有 256 个专家，专家规模比稠密 MLP 小得多。

### 参数核算

完整计算在 `code/main.py` 中。要点如下：

- 嵌入：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个稠密块：使用 MLA 的注意力（约每块 144M）+ 稠密 MLP（约每块 260M）+ 归一化。总计约 1.2B。
- 58 个 MoE 块：使用 MLA 的注意力（约 144M）+ 256 个专家每个（约 30M/个）+ 1 个共享专家（约 30M）+ 归一化。包含所有专家后每块约 7.95B。58 层合计约 461B。
- MTP 模块：14B。

总计：核心架构约 476B + 14B MTP + 公开文档中列出的其他结构化参数（偏置张量、专家特定组件、共享专家缩放等）构成了公开的 671B。我们在计算器中重现的数值与公开值相差大约 3-5% —— 差值来自 DeepSeek 报告在附录第 2 节中记录的细粒度项。

每次前向的活动参数：

- 注意力：每层 144M * 61 = 8.8B（所有层都会激活）。
- MLP 活动：前 3 层为稠密（3 * 260M = 780M），58 个 MoE 层每层激活 8 个路由专家 + 1 个共享专家 + 路由开销。每层激活的 MLP 约 260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- 嵌入 + 归一化：1.2B。
- 合计活动参数：核心约 26B + 14B MTP（训练时存在但推理时不总是运行）≈ 37B。

### 671B / 37B 比率

18x 的稀疏比（活动参数占总参数的 5.5%）。DeepSeek-V3 是迄今为止开源权重中稀疏性最高的前沿 MoE 模型之一。Mixtral 8x7B 在比率 13/47（28%）上密集得多。Llama 4 Maverick 的比率 17B/400B（4.25%）则可比。DeepSeek 的押注是：在前沿规模上，更多的专家与更低的激活比可以在每个活动 FLOP 上带来更好的质量。

### DeepSeek-V3 的位置

| Model | Total | Active | Ratio | Attention | Novel ideas |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

## 后续：R1、V4

DeepSeek-R1（2025）是在 V3 骨干上进行的推理训练运行。R1 使用相同架构。变化在于后训练配方（大规模在可验证任务上的 RL），而非预训练架构。

DeepSeek-V4（若发布）预计会保留 MLA + MoE + MTP，并加入 DSA（DeepSeek Sparse Attention），这是 Phase 10 · 17 中 NSA 的继任者。血统稳定：架构层面的创新会累积；每个版本都会再调整更多旋钮。

```figure
moe-routing
```

## 使用方法

`code/main.py` 是专为 DeepSeek-V3 形状定制的参数计算器。运行它，比较其输出与论文数值，并在假设的变体上使用（256 专家 vs 512、top-8 vs top-16、MLA rank 512 vs 1024）。

关注点：

- 总参数对比已公布的 671B。
- 活动参数对比已公布的 37B。
- 在 128k 上下文下的 KV 缓存 —— MLA vs GQA 的比较。
- 每层的细目分解，看看参数预算实际花在哪些地方。

## 交付成果

本课会产生 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 系列模型（V3、R1 或任何未来变体），它会生成一个逐组件的架构解读，说明配置中的每个字段、按组件推导出的参数计数，并识别该模型使用了哪四项 DeepSeek 特有创新中的哪些。

## 练习

1. 运行 `code/main.py`。将计算器估计的总参数与公开的 671B 比较，并找出差值来自哪里。论文的第 2 节有完整的项目化清单。
2. 将配置修改为 MLA rank 为 256（而非 512）。计算在 128k 上下文下由此得到的 KV 缓存大小。它带来多大百分比的减少？对每头表达能力有什么代价？
3. 比较 DeepSeek-V3（256 专家、top-8）路由与一个假设的（512 专家、top-8）变体。总参数会增长；活动参数保持不变。额外的专家容量在理论上能带来什么？在推理时会产生什么成本？
4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）第 2.1 节关于 MLA 的部分。用三句话解释为什么 K 和 V 的解压矩阵在推理时可以“吸收到”随后的 matmul 中以提高效率。
5. DeepSeek-V3 在大多数操作中使用 FP8 训练。计算将 671B 权重以 FP8 存储相较于 BF16 的内存节省。这个节省如何与 14.8T-token 的训练预算相互作用？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | 将 K 和 V 压缩到共享的低秩潜在（`kv_lora_rank`，典型为 512），在每个头上动态解压；KV 缓存仅存潜在 |
| kv_lora_rank | "MLA compression dim" | 用于 K 和 V 的共享潜在维度；DeepSeek-V3 使用 512 |
| First k dense layers | "Early layers stay dense" | 前几层跳过 MoE 路由，运行稠密 MLP 以保证稳定性 |
| num_experts_per_tok | "Top-k routing" | 每个 token 触发多少个路由专家；DeepSeek-V3 使用 8 |
| Shared experts | "Always-on experts" | 无论路由如何，都会处理每个 token 的专家；DeepSeek-V3 使用 1 |
| Auxiliary-loss-free routing | "Bias-adjusted load balance" | 在路由 logits 上使用每专家偏置并通过训练中调整偏置来维持负载平衡，而不增加损失项 |
| MTP module | "Extra prediction head" | 从 h^(1) 和 E(t+1) 预测 t+2 的 Transformer 块；更密集训练，提供免费投机解码草案 |
| DualPipe | "Bidirectional pipeline" | 将前向/反向计算与跨节点 all-to-all 重叠的训练调度 |
| Active parameter ratio | "Sparsity" | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 training | "8-bit training" | 在 FP8 中存储并在许多计算中使用 FP8；相较 BF16 大致减半内存，但带来小幅质量成本 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的架构、训练与结果文档  
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件与部署说明  
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前作  
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在 V3 架构上进行推理训练的后继工作  
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 系列注意力的未来方向  
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — 训练调度参考