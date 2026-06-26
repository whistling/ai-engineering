# EAGLE-3 投机性解码在生产环境中

> 投机性解码将一个快速的草稿模型与目标模型配对。草稿生成 K 个候选标记；目标在一次前向中验证这些标记；被接受的标记是“免费”的。在 2026 年，EAGLE-3 是生产级变体 —— 它在目标模型的隐藏状态上训练一个草稿头，而不是在原始标记上，从而将接受率 alpha 推高到通用聊天中 0.6–0.8 的区间。正确的问题不是“草稿有多快”，而是“我流量下的 alpha 是多少？”如果 alpha 低于约 ~0.55，在高并发下投机性解码反而会造成净损失，因为每个被拒绝的草稿都会产生一次额外的目标前向成本。这个课程教你先测量 alpha，再决定是否打开开关。

**Type:** 学习  
**Languages:** Python（stdlib、验收率玩具模拟器）  
**Prerequisites:** Phase 17 · 04（vLLM Serving Internals），Phase 10 · 18（多标记预测）  
**Time:** ~60 分钟

## 学习目标

- 说出投机性解码的三代并解释 EAGLE-3 相比 EAGLE-2 和经典草稿模型做了哪些改变。
- 定义接受率 alpha，计算由 alpha 和 K（草稿长度）带来的期望加速，并识别在目标并发下的收支平衡 alpha。
- 解释为什么在 vLLM 2026 中投机性解码是需要显式选择的（非默认），以及为何在没有测量 alpha 的情况下直接开启它是一个生产反模式。
- 制定一个测量计划：使用哪个基准、哪个提示分布、在哪个并发点测量、以及以哪个指标作为门控。

## 问题背景

解码是内存带宽受限的。在 H100 上运行 Llama 3.3 70B FP8 时，每解码一个标记约需读取 ~140 GB/s 的权重并输出一个标记。解码过程中 GPU 计算几乎空闲 —— 瓶颈是 HBM 带宽，而不是矩阵乘法吞吐。

投机性解码利用了这个差距。用一个廉价的草稿模型生成 K 个候选标记，然后让目标模型在一次前向中验证全部 K 个。每个被验证通过的标记实际上是“免费”的（摊销到目标本来需要做的一次 K 批量前向中）。

经典的草稿模型方法使用同族的更小模型（例如用 Llama 3.2 1B 为 Llama 3.3 70B 起草）。这种方式可行，但接受率平平 —— 更小模型的分布会偏离目标分布。EAGLE、EAGLE-2、再到 EAGLE-3 都是在目标模型的内部状态上直接训练一个轻量草稿头，因此草稿分布与目标更紧密对齐。这就是为什么从草稿模型的 alpha ~0.4 到 EAGLE-3 的 0.6–0.8。

注意事项：EAGLE-3 在 vLLM 2026 中是需要显式开启的。必须显式设置 `speculative_config`。没有这个标记，就没有加速。那些在没有在真实流量上测量 alpha 的情况下直接开启的人，常看到尾延迟（tail latency）变糟而不是变好。

## 概念

### 投机性解码实际带来的收益

没有投机性解码时，每个标记的成本是一次目标前向。使用草稿长度为 K 的投机性解码且接受率为 alpha 时，每次目标前向的期望产出标记数是 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证的额外开销。对于 K=5、alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。现实世界的数字通常集中在 2–3x，因为在生产流量上 alpha 很少那么高，而且在大批量时 epsilon 会增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的标记不会消失 —— 它们会导致第一个被拒绝标记触发第二次目标前向。在 alpha 降到 0.4 的负载下，你需要支付草稿开销、验证开销以及重试开销。在高并发（例如 256 并发）时，解码批次已经足够大，使得“仅目标”与“目标+验证”之间内存带宽的差距减小。在大多数 2026 年硬件上，当 alpha 低于 0.55 时，投机性解码是净负的。

Alpha 随工作负载而变化。在类似 ShareGPT 的通用聊天上，用 ShareGPT 训练的 EAGLE-3 可达到 alpha 0.6–0.8。在领域特定流量（代码、医学、法律）上，基于通用数据训练的草稿头会降到 0.4–0.6。训练一个领域特定的草稿头可以恢复 alpha —— 相较于目标模型的微调，这只是一次轻量且快速的训练作业。

### EAGLE 各代一览

- **经典草稿模型**：同族的小模型。Alpha 0.3–0.5。基础设施简单 —— 两个模型都要加载，草稿每个目标前向运行 K 次。
- **EAGLE-1（2024）**：在目标隐藏状态（最后一层）上训练的单个草稿头。Alpha ≈ 0.5–0.6。对目标仅增加小量参数开销。
- **EAGLE-2（2025）**：自适应草稿长度和基于树的草稿（一次目标前向验证多条分支）。Alpha ≈ 0.6–0.7。草稿调度器更复杂。
- **EAGLE-3（2025–2026）**：在多个目标层上训练草稿头（不仅仅是最后一层），对齐更好。通用聊天上 alpha ≈ 0.6–0.8。

### 2026 年的生产配方

1. 先上线纯目标模型。测量基线 TTFT、ITL、以及在目标并发下的吞吐。
2. 通过 vLLM 的 `speculative_config` 启用 EAGLE-3 草稿。重新运行基准。
3. 记录接受率 alpha。vLLM V1 将其报告为 `spec_decode_metrics.accepted_tokens_per_request`。将其除以请求的草稿长度即可得到 alpha。
4. 如果在生产流量分布上 alpha < 0.55，则禁用投机性解码或训练一个领域特定的 EAGLE-3 草稿。
5. 在生产并发下重跑。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾延迟

平均 ITL 会随着投机性解码下降。若不做调优，P99 可能会变差。被拒绝的草稿会触发两次序列（草稿 + 验证失败 + 重试）。在满批下，这两次前向会串行化。关注 P99 ITL，而不是 P50。

### EAGLE-3 已部署的场景

Google 在 2025 年把投机性解码部署到 AI Overviews（相同质量，更快响应）。vLLM V1 将 `speculative_config` 作为有文档的接口；V1 中兼容 chunked prefill 的变体是 GPU 上的 N-gram 投机性解码。SGLang 推荐在前缀占比较高的工作负载中使用 EAGLE-3 作为首选草稿路径。

### 一行收支平衡数学

期望加速：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 可解出 alpha：`alpha_breakeven = verify_overhead / K`。对于典型的 verify_overhead ≈ 0.15 和 K=5：`alpha_breakeven = 0.03`。但这是原始解码数学。在高并发时，verify_overhead 会增加且解码批次已将内存读取跨序列摊销，因此实际的 alpha_breakeven 会升到大约 0.45–0.55。

### 何时不使用投机性解码

- 批次为 1 的离线生成且延迟不重要时。使用纯目标模型即可。
- 非常短的输出（少于 50 个标记）。草稿开销与验证成本会占主导。
- 没有领域训练草稿头的专门领域。alpha 太低。
- vLLM v0.18.0 加上草稿模型的投机性解码并且使用 `--enable-chunked-prefill`。这个组合无法编译。文档中列出的例外是 V1 中的 N-gram GPU 投机性解码。

## 使用方式

`code/main.py` 模拟了一个带或不带投机性解码的解码循环，跨越不同的 alpha 值和草稿长度 K。它会打印收支平衡 alpha、测得的加速比以及尾部行为。在若干（alpha, K）组合上运行，能准确看到投机性解码在何处不再划算。

## 部署方案

本课会产出 `outputs/skill-eagle3-rollout.md`。给定一个目标模型、流量分布描述和并发目标，它会生成分阶段的 EAGLE-3 推出计划 —— 基准基线、开启配置、测量 alpha、以 alpha >= 0.55 作为门控、并关注 P99 ITL。

## 练习

1. 运行 `code/main.py`。在 K=5 时，要达到 2x 加速需要什么 alpha？要达到 3x 加速需要什么 alpha？对 verify_overhead 这一参数的敏感性如何？
2. 假设生产流量为 70% 通用聊天、30% 代码。通用聊天在基于 ShareGPT 训练的 EAGLE-3 上的 alpha 为 0.7；代码的 alpha 为 0.4。混合 alpha 为多少？投机性解码在该混合流量上是否净收益？
3. 阅读 vLLM 的 `speculative_config` 文档。列出三种模式（草稿模型、EAGLE、N-gram）以及哪一种与 chunked prefill 兼容。
4. 在启用 EAGLE-3 后你看到平均 ITL 下降 25%，但 P99 ITL 上升 15%。诊断原因并提出缓解方案。
5. 计算 Llama 3.3 70B 的 EAGLE-3 草稿头的内存开销。与运行 Llama 3.2 1B 作为经典草稿模型相比如何？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 投机性解码 | "draft plus verify" | 使用廉价模型提出 K 个标记，目标在一次前向中验证所有 K 个 |
| Acceptance rate alpha | "spec accept rate" | 草稿被目标接受的标记占比；唯一重要的指标 |
| Draft length K | "spec k" | 草稿每次为目标前向提出多少个标记；典型为 4–8 |
| Verify overhead epsilon | "spec overhead" | 相比纯目标前向，验证并重试的额外成本；随批量增长 |
| EAGLE-3 | "latest EAGLE" | 2025–2026 年的变体；在多个目标层上训练草稿头；通用聊天上 alpha 0.6–0.8 |
| `speculative_config` | "vLLM spec config" | vLLM V1 中的显式 opt-in；没有默认开启即没有加速 |
| N-gram spec decode | "N-gram draft" | 在 GPU 端使用提示中的 N-gram 查找来做草稿；与 chunked-prefill 兼容 |
| Break-even alpha | "no-op alpha" | 在该 alpha 下投机性解码带来零加速；要在生产并发下关注此值 |
| Rejected-draft two-pass | "reroll cost" | 草稿被拒绝时需要两次目标前向；推动 P99 尾延迟上升 |

## 深入阅读

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于 `speculative_config` 和 V1 中 chunked-prefill 兼容性的权威来源。  
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 精确的字段集合。  
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — 原始的 EAGLE 草稿头表述。  
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 自适应草稿与树形草稿。  
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 带投机性解码的高效 LLM 系统。  
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产化部署清单。