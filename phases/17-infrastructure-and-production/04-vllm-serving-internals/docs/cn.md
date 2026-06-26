# vLLM Serving Internals: PagedAttention, Continuous Batching, Chunked Prefill

> vLLM 在 2026 年的领先并非单一技巧，而是三项复合默认设置共同作用。PagedAttention 永远开启。Continuous batching 在每次解码迭代之间将新请求注入活动批次。Chunked prefill 将长提示切片，这样解码令牌永远不会饿死。把这三项都打开后，单张 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 并发时能推动 2,200–2,400 tok/s —— 大约比 vLLM 自身的默认配置高 25%，比天真的 PyTorch 循环快 3–4 倍。本课会在可绘制级别阅读调度器和注意力内核，并以 `code/main.py` 中的一个玩具连续批处理器结束，该文件以 vLLM 的方式调度 prefill 和 decode。

**Type:** 学习  
**Languages:** Python（stdlib，玩具版连续批处理调度器）  
**Prerequisites:** 阶段 17 · 01（模型服务），阶段 11（LLM 工程）  
**Time:** ~75 分钟

## 学习目标

- 将 PagedAttention 解释为一个 KV 缓存分配器：块、块表，以及为什么在生产负载下碎片率保持在 4% 以下。  
- 在迭代级别上绘制 continuous batching 的图：已完成序列如何离开批次，新序列如何在不清空批次的情况下加入。  
- 用一句话描述 chunked prefill，并指出它保护的是哪个延迟指标（提示：是 TTFT 的尾部，而不是平均吞吐量）。  
- 说出 2026 年 vLLM v0.18.0 的那个陷阱：启用所有优化时会让团队栽跟头的设置。

## 问题

一个天真的 PyTorch 服务循环一次只处理一个请求：分词、prefill、解码直到 EOS、返回。对单个用户这是可以的。对一百个用户则成了排队等待的耐心人群。显而易见的修正 —— 静态批处理 —— 会把窗口中每个请求填充到最长提示，把每次解码填充到最长预期输出，并在最慢的序列上阻塞整个批次。你为从未使用的填充买单，快速请求要等慢请求。

vLLM 同时解决了三类问题。PagedAttention 阻止 KV 缓存碎片像经典的连续分配那样吞掉 60–80% 的 GPU 内存。Continuous batching 允许请求在每次解码迭代之间加入和离开批次，因此批次始终充满真实工作。Chunked prefill 将 32k 令牌的长提示拆成 ~512 令牌的切片，与解码交错，这样长提示不会冻结每一个解码令牌在 GPU 上的执行。

2026 年的生产默认就是把这三项都打开。你需要理解每一项在做什么，因为失败模式都在调度器上，而不是模型上。

## 概念

### 把 PagedAttention 当作虚拟内存系统

一个 KV 缓存对于每个序列是 num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element 的规模。对于在 8192 令牌下的 Llama 3.3 70B，使用 BF16 大约是每个序列 1.25 GB。如果你为每个请求预留 8192 插槽，但平均请求只使用 1500 令牌，你会浪费大约 82% 的已预留 HBM。经典批处理要为这种浪费买单。

PagedAttention 借用了操作系统虚拟内存的想法。KV 缓存不是每个序列的连续空间，而是按固定大小块分配（默认 16 令牌）。每个序列有一张块表，将其逻辑令牌位置映射到物理块 ID。当序列超出已分配块时，增加一个块。当序列结束时，它的块返回到池中。

碎片率从经典方法的 60–80% 降到 PagedAttention 下的 <4%。你不会通过一个开关来启用 PagedAttention —— 这是 vLLM 出货时唯一的分配器。可调节的是 `--gpu-memory-utilization`（默认 0.9），它告诉 vLLM 在加载权重和激活后为 KV 块保留多少 HBM。

### 迭代级别的 Continuous batching

旧的“动态批处理”会等待一个窗口（比如 10 ms）来填充批次，然后运行 prefill + decode + decode + decode，直到每个序列都完成。快速序列会提前离开并处于空闲状态，而 GPU 仍在为慢序列收尾。

Continuous batching 在每次解码步骤之间运行。把正在运行的序列集合称为 `RUNNING` 列表。在每次迭代中：

1. 从 `RUNNING` 中移除任何刚到达 EOS 或达到最大令牌数的序列。  
2. 调度器查看等待队列。如果有空闲的 KV 块，会准入新的序列（prefill 或恢复）。  
3. 前向推理在当前 `RUNNING` 中的所有序列上运行，为每个序列发出一个新令牌。

批次大小从不被填充到固定数目。处于不同输出位置的序列共享一次融合的前向。在 2026 年的 vLLM 中，这被称为 `V1 scheduler`。关键不变量是：调度器每次解码迭代运行一次，而不是每个请求运行一次。

### Chunked prefill 保护 TTFT 尾部

Prefill 是计算受限的。在一张 H100 上，Llama 3.3 70B 的 32k 令牌提示纯 prefill 大约需要 ~800 ms。当 prefill 运行时，批次中每个其他序列的解码令牌都在等待。在服务循环中，一个长提示的首个令牌延迟（TTFT）会成为数十个其他用户的互令牌延迟（ITL）峰值。

Chunked prefill 把 prefill 切成固定大小的块（默认 512 令牌），并把每个块作为一个单位调度。在块之间，调度器可以推进解码序列一个令牌。你用每个块带来几个毫秒的绝对 prefill 延迟开销，换取更低的解码时抖动。在已发布的基准中，混合负载下的 P99 ITL 从 ~50 ms 降到 ~15 ms。

### 三项默认如何相互作用

这三项功能彼此依赖。PagedAttention 给调度器提供了细粒度的 KV 资源以进行权衡。Continuous batching 需要这种细粒度资源，以便准入新序列不会迫使全局重排。Chunked prefill 是调度器在同一 `RUNNING` 列表上做的一个决策 —— 它只是又一项调度策略，而不是独立系统。

你不需要知道每一个标志。你需要知道调度器优化的目标：在 KV 块预算下的良好吞吐（goodput），并受到分块 prefill 切片策略的约束。

### 2026 年 v0.18.0 的陷阱

在 vLLM v0.18.0 中，你不能将 `--enable-chunked-prefill` 与草案模型的 speculative decoding（`--speculative-model`）结合使用。文档中例外是 V1 scheduler 中的 N-gram GPU speculative decoding。那些在没看发行说明就把每个标志都打开的团队，会在启动时遇到运行时错误，而不是软回归。如果你的 speculative 带来的收益值得为之启用 chunked prefill，请重新评估 —— 在 2026 年，正确答案往往是使用不带 chunked prefill 的 EAGLE-3，而不是无法编译的草案模型 + chunked prefill。

### 你应该记住的数据

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三项都开：2,200–2,400 tok/s。  
- 同模型，vLLM 默认（无 chunked prefill）：~1,800 tok/s。  
- 同模型，天真的 PyTorch 前向循环：~600 tok/s。  
- PagedAttention 在生产负载下的 KV 碎片浪费：<4%。  
- 混合负载下的 P99 ITL：有 chunked prefill 时 ~15 ms，没时 ~50 ms。

### 调度器长什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 正是用标准库 Python 写的这个循环，使用假的令牌计数和假的前向延迟。运行它可以展示分块 prefill 如何在长 prefill 期间保持解码序列存活。

```figure
tensor-parallel
```

## 使用方法

`code/main.py` 模拟了类似 vLLM 的调度器，支持可切换的功能。运行它可以看到：

- `NAIVE` 模式：一次处理一个请求，不做批处理。  
- `STATIC` 模式：填充并等待，经典批处理。  
- `CONTINUOUS` 模式：迭代级别的准入与释放。  
- `CONTINUOUS + CHUNKED` 模式：预填充切片与解码交错。

输出显示总吞吐量（每秒虚拟令牌数）、TTFT 平均值和 P99 ITL。`CONTINUOUS + CHUNKED` 行在混合流量下应当占优。

## 交付

本课会生成 `outputs/skill-vllm-scheduler-reader.md`。在给定一个服务配置（批次大小、KV 内存利用率、chunked prefill 大小、speculative 配置）后，它会输出一个调度器诊断，指出三项默认中哪一项成为瓶颈以及该如何调优。

## 练习

1. 运行 `code/main.py`。在混合短/长请求的工作负载下比较 `STATIC` 与 `CONTINUOUS`。吞吐差距来自哪里 —— prefill 效率、解码效率，还是尾部延迟？  
2. 修改玩具调度器以添加 `--max-num-batched-tokens`。对于运行 Llama 3.3 70B FP8 的 H100，合适的值是多少？（提示：它是 KV 块大小和空闲块数量的函数，而不是原始 HBM。）  
3. 重新阅读 vLLM v0.18.0 的发行说明。哪些标志组合是互斥的？列出来。  
4. 计算以下情况下 1,000 个请求的 KV 缓存碎片浪费：平均 1,500 输出令牌，标准差 600 令牌，分别在 (a) 连续的每请求分配（最大为 8192），(b) 使用 PagedAttention 且块大小为 16 令牌。  
5. 用一段话解释为什么 chunked prefill 能降低 P99 ITL，但单独并不会提高吞吐率。在实践中吞吐率的提升来自哪里？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| PagedAttention | "the KV trick" | 用于 KV 缓存的固定大小块分配器；碎片率 <4% |
| Block table | "the page table" | 每序列的映射表，将逻辑令牌位置映射到物理 KV 块 |
| Continuous batching | "dynamic batching, but right" | 在每次解码迭代做准入/释放决策 |
| Chunked prefill | "prefill splitting" | 将长 prefill 拆成 512 令牌的切片并与解码交错 |
| TTFT | "first token time" | Prefill + 排队 + 网络；长提示时以 prefill 为主导 |
| ITL | "inter-token latency" | 连续解码令牌之间的时间；受批次大小主导 |
| Goodput | "throughput that meets SLO" | 在满足 TTFT 和 ITL 目标的前提下的令牌/sec |
| V1 scheduler | "the new scheduler" | vLLM 在 2026 年的调度器；N-gram 规格解码与 chunked prefill 兼容 |
| `--gpu-memory-utilization` | "the memory knob" | 在权重和激活之后为 KV 块保留的 HBM 比例 |

## 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于 chunked-prefill 和 speculative-decoding 兼容性的官方来源。  
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 年的发布节奏和版本特定行为。  
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 最初的文章，仍定义如何思考该分配器。  
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — 碎片率分析与调度器设计。  
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — 详细的 V1 scheduler 逐步讲解与火焰图。