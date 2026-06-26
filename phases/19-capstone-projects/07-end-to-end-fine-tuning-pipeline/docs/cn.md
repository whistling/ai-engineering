# Capstone 07 — 端到端微调流水线（从数据到 SFT 到 DPO 到 服务）

> 一个在你自己的数据上训练的 8B 模型，基于你的偏好通过 DPO 对齐，量化、投机性解码并以可度量的 $/1M tokens 提供服务。到 2026 年的开放栈是 Axolotl v0.8、TRL 0.15、Unsloth 用于迭代，GPTQ/AWQ/GGUF 用于量化，vLLM 0.7 搭配 EAGLE-3 用于服务。该结业项目的目标是可复现地运行整个流水线 —— 输入 YAML，输出可用的服务端点 —— 并根据 2026 年模型开放性框架发布模型卡。

**Type:** 结业项目  
**Languages:** Python（流水线）、YAML（配置）、Bash（脚本）  
**Prerequisites:** Phase 2（机器学习）、Phase 3（深度学习）、Phase 7（transformers）、Phase 10（从零构建 LLM）、Phase 11（LLM 工程）、Phase 17（基础设施）、Phase 18（安全）  
**Phases exercised:** P2 · P3 · P7 · P10 · P11 · P17 · P18  
**Time:** 35 小时

## 问题

到 2026 年，每个严肃的 AI 团队都会维护一套微调流水线。这并不是因为他们发布了前沿的基础模型，而是因为下游适配 —— 域内 SFT、对带标签偏好的 DPO、用于投机性解码的蒸馏草稿、以及配合 EAGLE-3 的服务化 —— 才是可量化获益的来源。Axolotl v0.8 支持多 GPU 的 SFT 配置。TRL 0.15 支持 DPO 和 GRPO。Unsloth 提供快速的单 GPU 迭代。vLLM 0.7 配合 EAGLE-3 在无损质量的情况下将解码吞吐推高 2–3 倍。工具链可用；工艺在于 YAML、数据卫生和评估纪律。

你将用一个 8B 基础模型（Llama 3.3、Qwen3 或 Gemma 3）依次进行 SFT 和 DPO，使用任务特定数据进行训练，量化以便部署，并针对 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 测量增益。你还将依据 2026 年模型开放性框架撰写模型卡。重点是可复现性 —— 一条命令可重新运行整条流水线。

## 概念

流水线有五个阶段。  
- Data：去重（MinHash / Datatrove）、质量过滤（类似 Nemotron-CC 的分类器）、PII 清理、针对公开基准的 split-hygiene 污染检测。  
- SFT：Axolotl YAML，8xH100 上的 ZeRO-3，余弦学习率调度，序列打包，2–3 轮微调。  
- DPO 或 GRPO：TRL 配置，1 轮，偏好对来自人工标注或模型判定，beta 调参。  
- 量化：GPTQ + AWQ + GGUF，保证部署灵活性。  
- 服务：vLLM 0.7 配合 EAGLE-3 的投机性草稿（或使用 SGLang + SpecForge），Kubernetes 部署，基于 queue-wait 的 HPA。

交付产物要包含消融对比：在三个任务特定基准上比较 SFT-only、SFT+DPO、SFT+GRPO。服务指标包括：batch 1/8/32 下的 tokens/s、EAGLE-3 接受率、每 1M tokens 的 $ 成本。安全评估：Llama Guard 4 的通过率。模型卡需包含偏差评估、可复现性随机种子、数据许可信息。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- 数据：Datatrove 用于去重，Nemotron-CC 分类器用于质量过滤，Presidio 用于 PII 清理  
- 基础模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B  
- SFT：Axolotl v0.8，配合 ZeRO-3、FlashAttention 3、序列打包  
- 偏好调优：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 快速迭代  
- 量化：GPTQ（Marlin）、AWQ、以及通过 llama.cpp 的 GGUF  
- 服务：vLLM 0.7 配合 EAGLE-3 的投机性解码（或 SGLang 0.4 + SpecForge）  
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro  
- 安全评估：Llama Guard 4、ShieldGemma-2  
- 基础设施：Kubernetes + NVIDIA 设备插件，基于 queue-wait 指标的 HPA  
- 可观测性：训练使用 W&B，推理使用 Langfuse

## 构建流程

1. Data pipeline。对原始语料运行 Datatrove 去重。应用类似 Nemotron-CC 的质量分类器。用 Presidio 清理 PII。以明确的随机种子写入训练/验证切分。  
2. 污染检测。对每个验证拆分，使用 MinHash 与 MMLU-Pro、MT-Bench-v2、RewardBench-2 的测试集计算相似度并拒绝任何重叠样本。  
3. Axolotl SFT。编写包含 ZeRO-3、FA3、序列打包的 YAML。在 8xH100 上训练 2–3 轮。日志记录到 W&B。  
4. TRL DPO / GRPO。从 SFT 检查点开始，针对偏好对运行 1 轮 DPO（或对数学/代码具有可验证 reward 的 GRPO）。进行 beta 扫描。  
5. 量化。产出三种量化模型：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M（用于 llama.cpp）。记录大小和标称吞吐率。  
6. 使用投机性解码进行服务。配置 vLLM 0.7，加载由 Red Hat Speculators 训练的 EAGLE-3 草稿头。测量接受率和在 batch 1 / 8 / 32 下的尾延迟。与 Anthropic / OpenAI 在相同评估上的 $/1M tokens 做对比。  
7. 评估矩阵。在 base、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成对比表。  
8. 安全评估。在开发集上测量 Llama Guard 4 的通过率。使用 ShieldGemma-2 进行输出过滤测试。  
9. 模型卡。按照 MOF 2026 模板撰写：数据、训练、评估、安全、许可、可复现性（含 YAML 和提交 SHA）。

## 使用示例

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 上线交付

`outputs/skill-finetuning-pipeline.md` 描述了交付物。用一条命令即可将数据贯穿 SFT、DPO、量化、服务化和评估，并生成模型卡和可访问的服务端点。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Eval delta vs base | 在目标任务（MMLU-Pro、MT-Bench-v2、任务特定基准）上的测量增益 |
| 20 | Pipeline reproducibility | 一条命令可在相同随机种子下端到端复现 |
| 20 | Data hygiene | 去重率、PII 清理覆盖率、污染检测为通过（green） |
| 20 | Serving efficiency | 在 bs=1/8/32 下的 tokens/s、EAGLE-3 接受率、每 1M tokens 的 $ 成本 |
| 15 | Model card + safety eval | 遵循 2026 MOF 的完整性 + Llama Guard 4 的通过率 |
| **100** | | |

## 练习

1. 对同一任务特定基准运行 SFT-only、SFT+DPO、SFT+GRPO。报告哪种偏好方法胜出以及幅度。  
2. 将 Llama 3.3 8B 替换为 Qwen3 14B。在质量匹配的情况下测量每 1M tokens 的 $ 成本。  
3. 测量在域内数据与通用 ShareGPT 上的 EAGLE-3 接受率。报告差值及其对延迟预算的含义。  
4. 在训练数据中注入 1% 的污染（泄露 MMLU-Pro 答案），重新运行评估。观察 MMLU-Pro 精度的非真实上升。为此构建能抓住这种情况的污染检测 CI 门控。  
5. 将 LoRA SFT 作为全量微调的替代方案。以 10x 更低内存测量质量差距。

## 术语速查

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Axolotl | "SFT trainer" | 用于 SFT、DPO 与蒸馏的统一 YAML 驱动训练器 |
| TRL | "Preference tuner" | Hugging Face 提供的用于 LLM 的 DPO、GRPO、PPO 实现库 |
| GRPO | "Group-relative policy optimization" | DeepSeek R1 的 RL 配方，带有可验证的 reward |
| EAGLE-3 | "Speculative decoding draft" | 预测未来 N 个 token 的草稿头；vLLM 用目标模型验证草稿 |
| MOF | "Model Openness Framework" | 2026 年对模型发布（数据、代码、许可）评分的标准 |
| Contamination check | "Split hygiene" | 基于 MinHash 的训练/测试集泄露检测 |
| Acceptance rate | "EAGLE / MTP metric" | 草稿被目标模型接受的 token 比率 |

（注：表中保留了常用术语缩写如 SFT、DPO、GRPO、EAGLE-3、LOra 等以便工程上下文一致。）

## 延伸阅读

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — SFT / DPO 训练器参考文档  
- [TRL documentation](https://huggingface.co/docs/trl) — DPO 与 GRPO 的参考实现  
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 迭代参考实现  
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 方法学  
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — 服务栈参考文档  
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 可替代的投机性解码训练器  
- [Model Openness Framework 2026](https://isocpp.org/) — 开放发布评分标准（MOF 2026）  
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 规范化评估运行器