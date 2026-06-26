# vLLM Production Stack with LMCache KV Offloading

> vLLM 的生产堆栈是参考的 Kubernetes 部署 —— 路由器、引擎和可观测性联动在一起。LMCache 是将 KV 缓存从 GPU 内存中抽取并跨查询与引擎复用的 KV 卸载层（先到 CPU DRAM，然后到磁盘/Ceph）。vLLM 0.11.0 的 KV Offloading Connector（2026 年 1 月）通过 Connector API（v0.9.0+）使其异步化且可插拔。卸载延迟不是用户感知的。即使没有共享前缀，LMCache 也很有价值 —— 当某张 GPU 的 KV 槽耗尽时，被抢占的请求可以从 CPU 恢复，而不是重新计算 prefill。已发布的基准（16x H100，80GB HBM，分布在 4 台 a3-highgpu-4g 上）表明：当 KV 缓存超过 HBM 时，本地 CPU 卸载和 LMCache 都显著提升吞吐；在低 KV 占用时，所有配置与基线匹配，仅有小幅开销。

**Type:** 学习  
**Languages:** Python（stdlib，玩具 KV 溢出模拟器）  
**Prerequisites:** Phase 17 · 04 (vLLM 服务内部原理)，Phase 17 · 06 (SGLang/RadixAttention)  
**Time:** ~60 分钟

## 学习目标

- 绘制 vLLM 生产堆栈层次图：路由器、引擎、KV 卸载、可观测性。
- 解释 KV Offloading Connector API（v0.9.0+）以及 0.11.0 如何通过异步路径隐藏卸载延迟。
- 量化何时 LMCache 在 CPU-DRAM 上有益（KV > HBM），何时会增加开销（KV 足够小可放入 HBM）。
- 根据部署约束在本地 vLLM CPU 卸载和 LMCache 连接器之间做出选择。

## 问题描述

你的 vLLM 服务在并发上升时 HBM 使用率达 100% 并有抢占事件。请求被逐出、重新排队，你在一分钟内对相同的 2K-token 提示进行了四次重复 prefill。GPU 计算被浪费在冗余的 prefill 上；有效产出远低于原始吞吐。

增加更多 GPU 成本线性增长。增加 HBM 不可行。但 CPU DRAM 很便宜 —— 单插槽可达 512 GB+，延迟比 HBM 大几个数量级，但对于“临时热”KV 缓存来说是可接受的。

LMCache 将 KV 缓存抽取到 CPU DRAM，使被抢占的请求能够快速恢复，并且不同引擎之间重复的前缀可以共享缓存，而无需各自重新 prefill。

## 概念

### vLLM 生产堆栈

`github.com/vllm-project/production-stack` 是参考的 Kubernetes 部署：

- **Router** — 支持缓存感知（Phase 17 · 11）。消费 KV 事件。
- **Engines** — vLLM 工作进程。每个 GPU 或每个 TP/PP 组一个实例。
- **KV cache offload** — LMCache 部署或本地 connector。
- **Observability** — Prometheus 抓取、Grafana 仪表盘、OTel traces。
- **Control plane** — 服务发现、配置、滚动更新。

以 Helm chart + operator 的形式发布。

### KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入了用于可插拔 KV 缓存后端的 Connector API。你的引擎将 block 卸载到 connector；connector 负责存储（内存、磁盘、对象存储、LMCache）。请求需要某个 block 时，connector 会将其加载回来。

vLLM 0.11.0（2026 年 1 月）新增了异步卸载路径 —— 卸载可以在后台进行，因此在常见情况下引擎不会被阻塞。端到端延迟和吞吐仍取决于工作负载形态、KV 缓存命中率和系统压力；vLLM 自身的说明指出：自定义内核的卸载在低命中率时可能会降低吞吐，并且异步调度与投机性解码存在已知的交互问题。

### 本地 CPU 卸载 与 LMCache

**本地 vLLM CPU 卸载**：引擎本地。将 KV block 存储在宿主机内存中。实现快速、没有网络跳。不能在引擎之间共享。

**LMCache connector**：集群级别。将 block 存储在共享的 LMCache 服务器（CPU DRAM + Ceph/S3 分层）。任何引擎都可以访问这些 block。已有 16x H100 的基准发布。

当单个引擎遇到 HBM 压力时选择本地卸载；当多个引擎共享前缀（例如具有共同系统提示的 RAG、多租户共享模板）时选择 LMCache。

### 基准行为

在那次 16x H100（80 GB HBM）分布到 4 台 a3-highgpu-4g 的测试中：

- 低 KV 占用（短提示，低并发）：所有配置与基线匹配，LMCache 增加约 3-5% 的开销。
- 中等占用：LMCache 在跨引擎前缀重用上开始展现优势。
- KV 超过 HBM：本地 CPU 卸载和 LMCache 均显著提升吞吐；LMCache 的收益更大，因为它支持跨引擎共享。

### LMCache 决定性适用场景

- 多租户服务场景，系统提示在租户间共享。
- RAG 场景中，文档片段在查询间重复出现。
- 在相同基模型上微调的多个变体（如 LoRA），基模型的 KV 重用可减少冗余工作。
- 抢占频繁的工作负载：从 CPU 恢复比重新 prefill 更便宜。

### 不建议启用的场景

- 只有小规模 HBM 压力 —— 你将承担开销但得不到收益。
- 短上下文（<1K token）—— 传输时间大于重新 prefill 的时间。
- 单租户且单提示工作负载 —— 无法捕获重用。

### 与分离式（disaggregated）服务的集成

Phase 17 · 17 的分离式服务 + LMCache 可以叠加：来自 prefill 池到 decode 池的 KV 传输如果未被使用会落到 LMCache；后续查询会从 LMCache 拉取。Phase 17 · 11 的缓存感知路由器可以路由到本地或 LMCache 共享缓存匹配的引擎。

### 需要记住的数字

- vLLM 0.9.0：发布 Connector API。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响依赖于工作负载、KV 命中率和系统压力（并非绝对保证）。
- 16x H100 基准：当 KV 占用超过 HBM 时 LMCache 有帮助。
- 小规模 HBM 压力：约 3-5% 的开销而无收益。

```figure
zero-sharding
```

## 使用方法

`code/main.py` 模拟了带和不带 LMCache 的抢占密集型工作负载。会报告避免的重新 prefill 次数、吞吐提升和收支平衡的 HBM 利用率。

## 交付物

本课输出 `outputs/skill-vllm-stack-decider.md`。根据工作负载形态和 vLLM 部署，决定使用本地、LMCache 或都不使用。

## 练习

1. 运行 `code/main.py`。LMCache 在多少 HBM 利用率开始带来净收益？  
2. 一个租户在 200 次/小时 的查询中共享一个 6K-token 的系统提示。计算每个租户预期的 LMCache 节省量。  
3. LMCache 服务器是单点故障。设计高可用策略（副本、回退到本地）。  
4. LMCache 将数据写到基于旋转盘的 Ceph。对于一个 4K-token 的 KV（70B FP8，约 500 MB），其读取时间与重新 prefill 比较如何？  
5. 论证 vLLM 0.11.0 的异步路径是否“免费”——开销隐藏在哪里？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Production-stack | "the reference deployment" | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | "KV backend interface" | vLLM 0.9.0+ 的可插拔 KV 存储接口 |
| Native CPU offload | "engine-local spill" | 将 KV 存储在同一引擎所在宿主机的内存中 |
| LMCache | "cluster KV cache" | 基于 CPU DRAM + 磁盘的跨引擎 KV 缓存服务器 |
| 0.11.0 async | "non-blocking offload" | 将卸载隐藏在引擎流背后（非阻塞） |
| Preemption | "evict to make room" | HBM 满时的 KV 缓存置换 |
| Prefix reuse | "same system prompt" | 多次查询共享前缀；缓存命中 |
| Ceph tier | "disk tier" | 位于 DRAM 之下的持久化存储层 |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)  
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。  
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)  
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector 实现。  
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — 异步路径详情。