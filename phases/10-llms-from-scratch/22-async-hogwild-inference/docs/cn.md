# Async and Hogwild! Inference

> Speculative decoding (Phase 10 · 15) parallelizes tokens within one sequence. Multi-agent frameworks parallelize across whole sequences but force explicit coordination (voting, sub-task splitting). Hogwild! Inference (Rodionov et al., arXiv:2504.06261) does something else: run N instances of the same LLM in parallel against a SHARED key-value cache. Each worker sees every other worker's generated tokens instantly. Modern reasoning models — QwQ, DeepSeek-R1 — can self-coordinate through that shared cache without any fine-tuning. The approach is experimental but it opens an entirely new axis of inference parallelism that sits orthogonal to spec decode. This lesson implements a two-worker Hogwild! simulator in stdlib Python and explains why the shared-cache collaboration emerges from the existing model's reasoning abilities.

**Type:** 构建  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 10 · 12（推理优化），Phase 10 · 15（投机性解码）  
**Time:** ~60 分钟

## Learning Objectives

- 描述三种常见的并行 LLM 拓扑结构（voting、sub-task、Hogwild!）并说明每种方法针对的问题域。
- 陈述 Hogwild! 的核心设置：多个 worker、一个共享的 KV cache、通过自我提示实现的涌现式协调。
- 推导 Hogwild! 的壁钟时间加速，作为 worker 数 `N`、任务级并行度 `p` 和协调开销 `c` 的函数。
- 用玩具问题在 stdlib Python 中实现一个双 worker Hogwild! 模拟器并观察涌现的任务划分。

## The Problem

Modern LLMs solve hard problems by producing long chains of reasoning — 5000 tokens of step-by-step logic is common, tens of thousands of tokens happens on deep math problems. At 35 tokens/sec decode on a 70B model, 50k tokens is 24 minutes. Interactive the model is not.

投机性解码（Phase 10 · 15）通过在单序列内并行化 token 可以带来 3–5 倍的加速。超过这个点，自回归解码的顺序依赖成为硬上限：每个新 token 都依赖于所有先前的 token。

显然的问题是：能否跨序列并行化？运行多个相同模型的副本解决同一问题，让它们协作并分担工作？

已有方法：投票集成（运行 N 个模型，取多数答案）、tree-of-thought（分支推理路径并重组）、多智能体框架（分配每个 agent 子任务，使用协调器）。这些方法在特定任务域中有效，但都引入了显式协调机制——投票规则、分支与剪枝逻辑、agent 间消息协议。

Hogwild! Inference 采取了不同的方法。N 个 worker 共享一个 KV cache。每个 worker 能够立即看到其他 worker 已生成的所有 token，就好像那些 token 是它自身的上下文。Workers 在没有任何训练或微调的情况下，会学会如何划分工作。现代的推理模型（QwQ、DeepSeek-R1、Claude 系列的推理模式）能读取共享缓存并说“我看到 worker 2 已经处理了基础情况，那我来做归纳步骤。”

截至 2026 年 4 月，速度提升依赖于工作负载且仍处于实验阶段。但这个思路值得了解，因为它打开了与投机性解码正交的、全新的推理并行维度。

## The Concept

### The setup

初始化 N 个 worker 进程，它们都运行相同的 LLM。不是为每个 worker 维护独立的 KV cache，而是维护一个共享的 cache。当 worker `i` 生成 token `t_j` 时，该 token 被写入共享缓存的下一个位置。当 worker `k` 执行下一步时，它会读取当前缓存状态（包含所有 N 个 worker 到目前为止生成的内容）。

在步进时间上，workers 争抢写入 token。没有每-worker 的位置索引——缓存是一个单一的增长序列。顺序由写入到达时间决定。

### Why coordination emerges

workers 共享一个提示（prompt）。典型示例是：“你是 N 个实例中的一个，一起协作解决这个问题。每个实例读取共享内存并能看到其他实例写了什么。避免重复工作。”提示词加上共享缓存就足够了。推理模型读取缓存，注意到问题的哪些部分已经被尝试过，然后（常常但不总是）转向未探索的部分。

Hogwild! 论文（Rodionov 等，2025）报告了一些观察：

- Workers 制定计划并通过缓存向其他 worker 传达。
- Workers 发现其他 worker 的推理错误并指出来。
- Workers 在计划失败时做出调整并提出替代方案。
- 当被提示检查冗余时，workers 能检测到并转向别的方向。

这些都不需要微调。涌现行为来自模型已有的推理能力。

### The naming

论文名借鉴了 Hogwild! SGD（Recht 等，2011），一种异步更新的优化器。类比是：SGD 的异步 worker 都写入共享参数向量；Hogwild! Inference 的 workers 都写入共享的 KV cache。两者都依赖经验收敛而非同步保证。

### RoPE makes this tractable

Rotary Position Embeddings（RoPE，Su 等，2021）通过在 Q 和 K 向量中用旋转编码位置信息。因为位置是旋转而不是内置偏移，token 的位置移动时不需要重新计算 KV cache 条目。当 worker `i` 在位置 `p` 向共享缓存写入时，其他读取该位置的 worker 可以直接使用缓存条目——无需重新旋转。

在学习位置或绝对位置模型中，Hogwild! 将需要在每次并发写入时使缓存失效。RoPE 让缓存保持稳定。

### Wall-time math

设 `T_serial` 为单个 worker 独立解决该问题所需时间。令 `p` 为任务级可并行部分的比率。令 `c` 为每步的协调开销（读取扩展缓存并决定写什么）。

单 worker 时间：`T_serial`。  
若协调为零，则 N worker Hogwild! 时间为：`T_serial * ((1 - p) + p / N)`。这是经典的 Amdahl。  
考虑协调开销后：`T_serial * ((1 - p) + p / N) + c * steps_per_worker`。

要使 worker 高效，`c` 必须相对于每步解码时间足够小。对生成 5k+ token 的推理模型而言，workers 可以承受数百 token 的协调开销仍能取得收益。对短交互式对话任务，协调成本占主导，Hogwild! 反而比串行更差。

### Concrete example

推理问题：1 万 token 的思维链。假设问题有 `p = 0.7` 的可并行内容（不同证明策略，不同情形分析），并且每个 worker 的协调开销为 `c = 200` token。取 `N = 4` worker：

- 串行时间：10000 步解码。
- Hogwild! 时间：10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 解码步。
- 加速比：10000 / 5550 = 1.8x。

这相当温和。但在更长的推理问题（5 万 token）上，协调开销被摊薄，加速会接近 2.5–3x。Hogwild! 在推理中的地位类似于线程级并行：在允许自然写多线程代码的语言里实现多线程并行。

### When to reach for Hogwild!

- 长推理问题（数千 token），任务可以在独立子目标间并行化。
- 被训练为逐步思考的推理模型。非推理模型不会良好自我协调。
- 单节点部署且有足够 VRAM 容纳共享缓存和 N 个 worker 进程。缓存是共享的，但每个 worker 有自己的激活内存。

### When not to

- 短交互式聊天。协调开销占主导。
- 无法并行化的任务（单线性证明、单次编译）。N=1 是上限。
- 非推理模型。不会出现协调行为。
- 多节点部署。共享缓存需要非常快的跨 worker 同步。节点内（intra-node）可行；跨节点（cross-node）会因延迟而惨烈。

### The experimental status

截至 2026 年 4 月，Hogwild! 仍是研究方法，并有一个开源的 PyTorch 实现。尚未进入生产采纳。三个阻碍因素：

1. 并发进程间共享 KV cache 的管理是复杂的工程问题。  
2. 涌现式协调依赖任务；基准测试仍在构建中。  
3. 相比投机性解码带来的加速，Hogwild! 的提升较温和，两者可以组合但组合起来的工程复杂度更高。

值得了解，值得实验，但尚不足以在产品上孤注一掷。

```figure
continuous-batching
```

## Build It

`code/main.py` 实现了一个玩具的 Hogwild! 模拟器：

- 两个 worker 进程，每个是一个确定性的“LLM”，生成几类 token（work-token, observe-token, coordinate-token），且有已知概率分布。
- 一个共享缓存（仅一个 token 列表）供两个 worker 读写。
- 一个简单的协调逻辑：当 worker 看到另一个 worker 在某类上已生成足够的 work-token 时，它会选择不同的类别。

模拟器在固定的步预算内运行并报告：

- 产生的 work-token 总数。  
- 总壁钟时间（以 worker 步数计）。  
- 相对于单 worker 的有效加速。  
- 哪个 worker 写了哪个 token 的跟踪记录。

### Step 1: the shared cache

一个列表，供两个 worker append。真实实现中使用简单锁（Python 的 `threading.Lock`）；此处用一个计数器来模拟。

### Step 2: the worker loop

每个 worker 在每个步骤中：

- 读取当前共享缓存。  
- 根据现有内容决定要写入哪类 token。  
- 写入一个 token。

### Step 3: the coordination heuristic

如果类别 X 在缓存中已有 K 个 token，而 worker 原本想写类别 X，则 worker 切换到类别 Y。这个玩具启发式是模型行为的简化替代：模型“注意到这部分已经被覆盖，就去做别的”。

### Step 4: measured speedup

在相同的总步预算下，分别以 N=1 和 N=2 运行模拟器。统计产生的 work-token。N=2 在协调良好时应产生大约 1.5–1.8x 的 work-token。

### Step 5: stress the coordination

降低协调启发式的敏感度。重跑。观察在缺乏协调时 N=2 会重复产生相同 token，加速降到小于 1。这与论文观察一致：只有当 workers 拥有自我协调的推理能力时方法才有效。

## Use It

截至 2026 年 4 月，Hogwild! 在生产中的集成仍属研究级别。来自 Yandex/HSE/IST 的参考实现基于 PyTorch，目标是单节点多进程部署，针对 DeepSeek-R1 和 QwQ 模型。

务实的采用路径：

1. 对你的推理任务做剖面分析。衡量探索性 token（多策略、情形分析、搜索）占比与线性 token 的比例。  
2. 如果探索占主导，先运行一个双 worker 的 Hogwild! 实验。测量壁钟时间改善。  
3. 如果提升低于 1.3x，你处于协调占主导的区域，应回退到单 worker。  
4. 如果提升超过 1.5x，尝试扩展到 N=4 并再次测量。通常边际效应在 N=4–8 时开始递减。

可与投机性解码配合：每个 Hogwild! worker 可以独立使用 spec decode。两个加速大致相乘：3x 的 spec decode 与 1.8x 的 Hogwild! 可合成约 5.4x 的整体加速（相对于朴素单 worker 解码）。

## Ship It

本课输出 `outputs/skill-parallel-inference-router.md`。给定一个推理工作负载的概况（token 预算、任务并行剖面、模型家族、部署目标），该路由器在 voting、tree-of-thought、multi-agent、Hogwild! 和 speculative decoding 策略间做出推荐。

## Exercises

1. 运行 `code/main.py` 的默认设置。确认在相同壁钟时间内 N=2 Hogwild! 配置产生的 work-token 比 N=1 基线多。  

2. 将协调启发式的强度降低（设置 `coordination_weight=0.1`）。重新运行。显示加速崩塌。解释原因：当 workers 无法协调时会重复劳动。  

3. 计算一个 50k token 推理任务在 `p=0.8, c=500`、N=4 的预期 Hogwild! 加速。对比一个 1k token 聊天任务在 `p=0.3, c=200`、N=4 的情况。为什么前者是收益而后者是损失？  

4. 阅读 Hogwild! 论文的第 4 节（初步评估）。识别作者报告的两种失败模式。描述如何通过更好的协调提示词来缓解每种失败模式。  

5. 在玩具模拟器中将 Hogwild! 与投机性解码结合：每个 worker 在内部使用 2-token 的 spec-decode。报告乘法加速比。当两个 worker 都想扩展相同的共享缓存前缀时，会出现什么账本（bookkeeping）问题？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Hogwild! | "Parallel workers, shared cache" | N instances of the same LLM running concurrently with one shared KV cache; emergent coordination via self-prompting |
| Shared KV cache | "The coordination medium" | A single growing KV buffer that all workers read and write; enables instant token visibility across workers |
| Emergent coordination | "No training needed" | Reasoning-capable LLMs can read the shared cache and divide work without any fine-tuning or explicit protocol |
| Coordination overhead (c) | "Tokens spent orienting" | The per-worker cost of reading the extended cache and deciding what to do; must stay small vs total decode time |
| Parallelizable fraction (p) | "What can run in parallel" | Task-level parallelism: the fraction of the total work that is not intrinsically sequential |
| RoPE enables Hogwild! | "Rotary positions are shift-invariant" | Because positions are rotations, writing into a shared cache does not require recomputing prior tokens |
| Voting ensemble | "Run N, pick the majority" | The simplest parallel inference topology; useful for classification, less for long-form reasoning |
| Tree of thought | "Branch and prune" | Reasoning strategy that explores multiple branches and prunes; explicit coordination logic |
| Multi-agent framework | "Assign sub-tasks" | Each agent gets a role; a coordinator orchestrates; heavy protocol overhead |

## Further Reading

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — the Hogwild! paper, preliminary evaluation on QwQ and DeepSeek-R1  
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — the original Hogwild!, the naming origin  
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE, the property that makes shared-cache inference tractable  
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — the tree-of-thought reasoning strategy Hogwild! sits orthogonal to  
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — speculative decoding, the within-sequence parallelism Hogwild! composes with  
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm) — the single source of truth for the paper's experiments