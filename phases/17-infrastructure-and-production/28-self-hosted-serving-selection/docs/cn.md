# Self-Hosted Serving Selection — llama.cpp, Ollama, TGI, vLLM, SGLang

> 四个引擎在 2026 年主导自托管推理。根据硬件、规模和生态选择：**llama.cpp** 在 CPU 上最快——支持最广泛的模型，对量化和线程控制有完全掌控。**Ollama** 是适合开发笔记本的一键安装方案，速度比 llama.cpp 慢约 15–30%（Go + CGo + HTTP 序列化），在接近生产负载下吞吐率差距约 3 倍。**TGI 于 2025 年 12 月 11 日进入维护模式**——仅修复 bug，原始吞吐率比 vLLM 慢约 10%，但在可观测性和 HF 生态集成方面历史上最优秀。该维护状态使其成为长期项目的高风险选择——对于新项目，SGLang 或 vLLM 是更安全的默认。**vLLM** 是通用生产默认——v0.15.1（2026 年 2 月）加入了 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是具代理性的多轮/前缀密集型专家——在生产中部署于 400,000+ GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：仅 CPU → 仅能选 llama.cpp。AMD / 非 NVIDIA → 仅能选 vLLM（TRT-LLM 被 NVIDIA 锁定）。2026 年的流水线模式：开发 = Ollama，暂存 = llama.cpp，生产 = vLLM 或 SGLang。全程使用相同的 GGUF/HF 权重。

**Type:** 学习  
**Languages:** Python (stdlib, engine-decision tree walker)  
**Prerequisites:** 所有第 17 阶段涵盖引擎的课程（04、06、07、09、18）  
**Time:** ~45 分钟

## 学习目标

- 根据硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和工作负载（通用聊天 / 代理 / 长上下文）选择合适的引擎。  
- 说明 2026 年为何要注意 TGI 的维护模式状态（2025 年 12 月 11 日），以及该事件为何将新项目偏向 vLLM 或 SGLang。  
- 描述使用相同 GGUF 或 HF 权重的 dev/staging/prod 流水线。  
- 解释为何“仅 CPU” 强制使用 llama.cpp，以及为何“AMD” 排除 TRT-LLM。

## 问题情境

你的团队启动一个新的自托管 LLM 项目。一位工程师说用 Ollama，另一位说用 vLLM，第三位说“不是 TGI 开箱就能运行吗？”。三者在不同场景下都有道理，但没有一种方案适用于所有情况。

在 2026 年，选择树很重要：先看硬件，其次看规模，第三看工作负载。还有一个具体的 2025 年事件——TGI 于 12 月 11 日进入维护模式——改变了新项目的默认选择。

## 概念

### 五大引擎

| Engine | 最适合 | 说明 |
|--------|--------|------|
| **llama.cpp** | CPU / 边缘 / 最少依赖 / 最广模型支持 | 在 CPU 上最快，完全可控 |
| **Ollama** | 开发笔记本、单用户、一键安装 | 比 llama.cpp 慢 15–30%；生产负载下吞吐率差距 3 倍 |
| **TGI** | HF 生态、受监管行业 | **自 2025-12-11 起进入维护模式** |
| **vLLM** | 通用生产，100+ 用户 | 广泛的生产默认；v0.15.1（2026-02） |
| **SGLang** | 具代理性的多轮、前缀密集型工作负载 | 在生产中部署于 400,000+ GPU |

### 以硬件为首的决策

- **仅 CPU** → llama.cpp。Ollama 也能用，但更慢。没有其它引擎在 CPU 上有竞争力。  
- **AMD GPU** → vLLM（支持 AMD ROCm）。SGLang 也可行。TRT-LLM 被 NVIDIA 锁定，因此不可用。  
- **NVIDIA Hopper（H100 / H200）** → vLLM、SGLang 或 TRT-LLM。三者均为一线选择。  
- **NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 在吞吐率上领先（Phase 17 · 07）。vLLM 和 SGLang 紧随其后。  
- **Apple Silicon（M 系列）** → llama.cpp（Metal）。Ollama 对此有封装支持。

### 以规模为次的决策

- **1 用户 / 本地开发** → Ollama。一键安装，首 token 响应秒级。  
- **10–100 用户 / 小团队** → vLLM 单 GPU。  
- **100–10k 用户 / 生产环境** → vLLM production-stack（Phase 17 · 18）或 SGLang。  
- **10k+ 用户 / 企业级** → vLLM production-stack + 去耦合（Phase 17 · 17） + LMCache（Phase 17 · 18）。

### 以工作负载为三的决策

- **通用聊天 / 问答** → vLLM 在广泛默认场景上胜出。  
- **具代理性的多轮（工具、规划、记忆）** → SGLang 的 RadixAttention（Phase 17 · 06）占优。  
- **RAG 且大量前缀重用** → SGLang。  
- **代码生成** → vLLM 表现良好；SGLang 在缓存利用上稍有优势。  
- **长上下文（128K+）** → vLLM + 分块 prefill；SGLang + 分层 KV（键值）策略。

### TGI 的维护陷阱

Hugging Face 的 TGI 自 2025 年 12 月 11 日进入维护模式——仅继续修复 bug。历史上它在可观测性和 HF 生态（模型卡、安全工具）集成方面位列前茅，但在原始吞吐率上略落后于 vLLM。

对 2026 年的新项目：应默认回避 TGI。现有的 TGI 部署可以继续运行，但应规划逐步迁移。SGLang 和 vLLM 是更稳妥的默认选择。

### 流水线模式

开发（Ollama）→ 暂存（llama.cpp）→ 生产（vLLM）。全程使用相同的 GGUF 或 HF 权重。工程师可在笔记本上快速迭代；暂存环境镜像生产的量化；生产则是最终的服务目标。

### Ollama 的注意点

Ollama 非常适合开发场景。但对于共享生产并不理想：Go 的 HTTP 序列化带来开销，并发管理不如 vLLM 复杂，OpenTelemetry 支持落后。将 Ollama 用于其擅长的场景——单用户、一键安装——然后在共享场景下切换到 vLLM。

### 自托管 vs 托管是另一个独立决策

Phase 17 · 01（托管超大厂）、·02（推理平台）涵盖托管方案。本课假设你已决定自托管。选择自托管的原因：数据驻留、定制微调、规模化时的总体拥有成本、托管平台上没有可用的领域模型。

### 需要记住的关键数据

- TGI 进入维护模式：2025 年 12 月 11 日。  
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。  
- SGLang 生产部署规模：400,000+ GPU。  
- Ollama 相对于 llama.cpp 的吞吐率差距：慢 15–30%；生产负载下 3 倍差距。

```figure
data-parallel
```

## 使用方法

`code/main.py` 是一个决策树遍历器：给定硬件 + 规模 + 工作负载，选择一个引擎并解释原因。

## 部署输出

本课产出 `outputs/skill-engine-picker.md`。在给定约束下，选择一个引擎并写出迁移计划。

## 练习

1. 使用你的硬件 / 规模 / 工作负载运行 `code/main.py`。输出是否符合你的直觉？  
2. 你的基础设施有 12 块 H100 和 8 块 MI300X（AMD）。选哪个引擎？为什么 TRT-LLM 不在考量范围？  
3. 一个团队在 2026 年仍想使用 TGI，因为“我们熟悉它”。论证迁移的必要性。  
4. Ollama 开发到 vLLM 生产：在量化、配置和可观测性上需要做哪些变更？  
5. 一个 RAG 产品，P99 前缀长度 8K，且在租户间高度重用前缀。选择一个引擎并将其与 Phase 17 · 11 + 18 组合成堆栈。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| llama.cpp | “那个 CPU 的” | 模型支持最广、在 CPU 上最快 |
| Ollama | “笔记本那款” | 一键安装，适合开发级别的吞吐率 |
| TGI | “HF 的 serving” | 自 2025 年 12 月起进入维护模式 |
| vLLM | “默认” | 2026 年的广泛生产基线 |
| SGLang | “具代理性的那个” | 前缀密集型，RadixAttention |
| TRT-LLM | “NVIDIA 锁定” | Blackwell 吞吐率领先，仅限 NVIDIA |
| GGUF | “llama.cpp 格式” | 打包的 K-量化变体 |
| Production-stack | “vLLM K8s” | Phase 17 · 18 的参考部署 |
| Pipeline pattern | “dev→stage→prod” | Ollama → llama.cpp → vLLM，使用相同权重 |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)  
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)  
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)  
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)  
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — 发布说明。  
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)