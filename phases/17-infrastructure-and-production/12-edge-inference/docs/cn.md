# Edge Inference — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> 核心的边缘约束是内存带宽，而非算力。移动端 DRAM 仅为 50–90 GB/s；数据中心的 HBM3 可达 2–3 TB/s — 差距在 30–50 倍之间。解码受到内存带宽限制，因此这一区别决定了性能。在 2026 年，边缘场景分为四类。Apple M4/A18 Neural Engine 峰值 38 TOPS，且使用统一内存（无 CPU↔NPU 拷贝开销）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上运行 Llama 3.1 8B（Q4）约为 ~41 tok/s（大约为本地性能的 70–80%）；WebLLM 在 GitHub 上有 17.6k 星，提供与 OpenAI 兼容的 API，覆盖率约为 70–75% 的移动设备。NVIDIA Jetson Orin Nano Super（8GB）可部署 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 在 gpt-oss-20b 上运行约 ~40 tok/s；Jetson T4000（JetPack 7.1）性能为 AGX Orin 的 2 倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、分块预填充（chunked prefill）——已在 2026 年 CES 被 Bosch、ThunderSoft、MediaTek 展示。

**Type:** 学习  
**Languages:** Python（stdlib，带有带宽受限解码的示例模拟器）  
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)，Phase 17 · 09 (Production Quantization)  
**Time:** ~60 分钟

## 学习目标

- 解释为什么移动端 LLM 推理受内存带宽限制，算力是次要因素。  
- 列举四个边缘目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson），并为每个目标匹配适用的用例。  
- 说明 2026 年 WebGPU 的覆盖差距（Firefox Android 正在追赶）以及 Safari iOS 26 的发布。  
- 为每个目标选择合适的量化格式（Apple ANE 使用 Core ML INT4 + FP16，Hexagon 使用 QNN INT8/INT4，浏览器使用 WebGPU Q4，Jetson 使用 NVFP4）。

## 问题

客户希望实现一个在设备端运行的聊天机器人：以语音为主、默认隐私、本地离线可用。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 运行约 ~55 tok/s —— 足够。在 iPhone 16 Pro 上，同一模型运行仅为 3 tok/s —— 不足。在一台中档 Android（Snapdragon 8 Gen 3）上为 7 tok/s。通过 Chrome Android 的 WebGPU 在浏览器中运行时，根据设备不同为 4–8 tok/s。

吞吐量差异不是移植问题，而是带宽差距乘以量化格式再乘以 NPU 是否可由用户态访问的结果。到 2026 年，边缘推理分为四种不同的问题，需要四种不同的解决方案。

## 概念

### 带宽是真正的天花板

解码（decode）在每个令牌都会读取完整权重集。一个 7B 模型在 Q4 下约为 3.5 GB。以 50 GB/s 的带宽读取 3.5 GB 需要 70 ms —— 理论上限约为 ~14 令牌/秒。在 90 GB/s（高端移动 DRAM）下，上限移动到约 ~25 令牌/秒。无论算力再强也无法突破这个限制。

数据中心的 HBM3 在 3 TB/s 时能在 1.2 ms 内读取相同的 3.5 GB —— 上限为 830 令牌/秒。同样的模型、相同的权重，不同的内存子系统。

### Apple Neural Engine (M4 / A18)

- 峰值可达 38 TOPS。使用统一内存（CPU 和 ANE 共享同一内存池）——没有拷贝开销。  
- 通过 Core ML + `.mlmodel` 编译模型访问，或通过 PyTorch 使用 Metal Performance Shaders (MPS)。  
- Llama.cpp 的 Metal 后端使用的是 MPS，而非直接调用 ANE；原生 ANE 需要 Core ML 转换。  
- 对于 2026 年的 iOS 应用，最实用的路径：Core ML + INT4 权重 + FP16 激活。

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- 峰值可达 45 TOPS。与 SoC 中的 CPU 和 GPU 集成，但属于独立的内存域。  
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换。  
- 聊天模板、Llama 3.2、Phi-3 等都作为 AI Hub 上的一等工件发布。

### Intel / AMD NPUs (Lunar Lake, Ryzen AI 300)

- 40–50 TOPS。软件生态落后于 Apple/Qualcomm；OpenVINO 正在改进，但仍属小众。  
- 最适合 Windows ARM 的 Copilot 类应用；在 AMD/Intel 桌面上用于本地优先部署。

### WebGPU + WebLLM

- 通过 WebGPU 的计算着色器在浏览器中运行模型；无需安装。  
- 在 M3 Max 上，Llama 3.1 8B Q4 约为 ~41 tok/s —— 大约是相同后端本地性能的 70–80%。  
- WebLLM 在 GitHub 上有 17.6k 星；提供与 OpenAI 兼容的 JS API；Apache 2.0 许可。  
- 2026 年覆盖：Chrome Android v121+、Safari iOS 26 GA，Firefox Android 仍在追赶。总体移动覆盖率约为 70–75%。

### NVIDIA Jetson 系列

- Orin Nano Super（8GB）：可容纳 Llama 3.2 3B、Phi-3 并实现良好令牌速率。  
- AGX Orin：通过 vLLM 在 gpt-oss-20b 上运行约 ~40 tok/s。  
- Thor / T4000（JetPack 7.1）：性能为 AGX Orin 的 2 倍，支持 EAGLE-3 和 NVFP4。  
- TensorRT Edge-LLM（2026）支持 EAGLE-3 的投机性解码（speculative decoding）、NVFP4 权重和分块预填充——将数据中心的优化移植到边缘。

### 每个目标的量化选择

| 目标 | 格式 | 备注 |
|------|------|------|
| Apple ANE | INT4 权重 + FP16 激活 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | 使用 `mlc_llm convert_weight` + 编译后的 `.wasm`；GGUF 不受支持 |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 受内存带宽限制 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘上的长上下文陷阱

Llama 3.1 的 128K 上下文是数据中心的特性。在一台只有 8 GB 内存的手机上，4 GB 模型 + 2 GB 的 KV 缓存用于 32K 令牌 + 操作系统开销 = 会导致内存溢出（OOM）。边缘部署通常将上下文限制在 4K–8K，除非接受对 KV 进行激进量化（Q4 KV）。

### 语音是杀手级应用

语音代理对延迟非常敏感（首令牌 < 500 ms）。本地推理可以完全消除网络延迟。结合语音转文本（Whisper Turbo 等变体可在边缘运行），边缘推理成为生产级别的语音回路。

### 需要记住的数据

- Apple M4 / A18 ANE：38 TOPS。  
- Qualcomm Hexagon SD X Elite：45 TOPS。  
- WebLLM 在 M3 Max 上：Llama 3.1 8B Q4 约为 ~41 tok/s。  
- AGX Orin：通过 vLLM 在 gpt-oss-20b 上约为 ~40 tok/s。  
- 数据中心与边缘的带宽差距：30–50x。  
- WebGPU 移动覆盖率：约 70–75%（Firefox Android 落后）。

## 使用方式

`code/main.py` 计算基于带宽受限数学的理论解码吞吐上限，涵盖各个边缘目标。与观测到的基准进行比较，并指出何处是带宽而非算力成为瓶颈。

## 交付

本节课会生成 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/浏览器/Jetson）、模型以及延迟/内存预算，选择量化格式和转换流水线。

## 练习

1. 运行 `code/main.py`。对一个 7B 模型在 Q4 下，在 Snapdragon 8 Gen 3（约 77 GB/s 带宽）上计算解码上限。将该理论值与观测到的 6–8 tok/s 做比较——运行时是否高效？  
2. Android 上的 WebGPU 需要 Chrome v121+。为较旧浏览器设计一个回退方案——通过相同的与 OpenAI 兼容的 API 在服务器端执行。  
3. 你的 iOS 应用需要 4K 上下文流式处理。哪种模型/格式组合能让你在 iPhone 16 上保持活跃内存低于 4 GB？  
4. Jetson AGX Orin 在 gpt-oss-20b 上运行约 40 tok/s。Jetson Nano 仅能容纳 3B。如果你的产品同时面向两者，如何统一推理栈？  
5. 论证 “WebLLM 在 2026 年是否可用于生产环境”。引用覆盖率、性能以及 Firefox Android 的差距。

## 术语表

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| ANE | "Apple neural engine" | M 系和 A 系中的设备端 NPU；统一内存 |
| Hexagon | "Qualcomm NPU" | Snapdragon 的 NPU；通过 QNN SDK 访问 |
| WebGPU | "browser GPU" | W3C 标准化的浏览器 GPU API；Chrome/Safari 在 2026 年支持 |
| WebLLM | "browser LLM runtime" | MLC-LLM 项目；Apache 2.0；与 OpenAI 兼容的 JS 接口 |
| Jetson | "NVIDIA edge" | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | "edge TensorRT" | 2026 年的 TensorRT-LLM 边缘移植；支持 EAGLE-3 + NVFP4 |
| Unified memory | "shared pool" | CPU 与 NPU 共享的内存池；无拷贝开销 |
| Bandwidth-bound | "memory limited" | 解码由读取权重的字节/秒速率限制 |
| Core ML | "Apple conversion" | Apple 的 ANE 原生模型转换框架 |
| QNN | "Qualcomm stack" | Qualcomm Neural Network SDK |

## 进一步阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 生态概览与基准测试。  
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。  
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 年的边缘移植发布。  
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 设计与基准。  
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 原生转换文档。  
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — 为 Hexagon 提供的预转换模型。