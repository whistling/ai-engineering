# Open-Weight VLM Recipes: What Actually Matters

> 2024–2026 年的开源权重 VLM 文献是一片消融表的森林。Apple 的 MM1 测试了 13 种图像编码器、连接器和数据混合的组合。Allen AI 的 Molmo 证明详尽的人类描述优于 GPT-4V 蒸馏。Cambrian-1 对 20+ 个编码器做了比较。Idefics2 将设计空间形式化为五个轴。Prismatic VLMs 在受控基准上比较了 27 种训练配方。在这些噪声中，有一小部分跨论文成立的结论：图像编码器比连接器架构更重要，数据混合比这两者都更重要，详尽的人类描述在相同 token 数下胜过蒸馏合成数据。本文解读那些表格，免去你逐篇阅读的负担。

**Type:** 学习 + 实验  
**Languages:** Python（stdlib、消融表解析器 + 配方选择器）  
**Prerequisites:** Phase 12 · 05（LLaVA 基线）  
**Time:** ~180 分钟

## 学习目标

- 说出五轴 VLM 设计空间：图像编码器、连接器、LLM、数据混合、分辨率调度（resolution schedule）。
- 阅读 MM1 / Idefics2 / Cambrian-1 的消融表并预测某个旋钮会如何影响给定基准。
- 在给定计算预算和任务混合的情况下为新的 VLM 选择配方（编码器、连接器、数据、分辨率）。
- 解释为什么详尽的人类描述在相同 token 数下胜过 GPT-4V 蒸馏。

## 问题陈述

存在数百个开源权重的 VLM。大多数“良好”与“最先进”之间的差距并非来自架构，而是来自数据、分辨率调度和编码器选择。知道在模型表现不佳时应先调整哪个旋钮，可以帮你避免价值数千万 GPU 小时的错误。

2023 年浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）基于 caption-pair 预训练 + LLaVA-Instruct-150k。是个不错的基线，MMMU 顶点约 35%。

2024 年浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）做了详尽的消融。结果既令人意外又实用。

## 概念

### 五轴设计空间

Idefics2（Laurençon 等，2024）给出了轴：

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch 大小、输入分辨率和预训练目标上不同。
2. 连接器（Connector）。MLP（2–4 层）、Q-Former（32 个 queries + cross-attn）、Perceiver Resampler（64 queries）、C-Abstractor（卷积 + 双线性池化）。
3. 语言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 大小是参数成本的主导因素。
4. 训练数据。Caption pairs（CC3M、LAION）、interleaved（OBELICS、MMC4）、instruction（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率调度（Resolution schedule）。固定 224/336/448、AnyRes、原生动态。在训练中分阶段提高或保持恒定。

每个生产级 VLM 都要在每个轴上做出选择。MMMU 得分的大部分方差由轴 1、4 和 5 解释——而不是你选的连接器。

### 轴 1：编码器比连接器更重要

MM1 第 3.2 节显示：将 CLIP ViT-L/14 替换为 SigLIP SO400m/14 会增加 3+ 点 MMMU。将连接器从 MLP 换成 Perceiver Resampler 增益不到 1 点。Idefics2 复现了这一点：SigLIP > CLIP，Q-Former ≈ MLP ≈ Perceiver（在相同 token 数下）。

Cambrian-1 的 “Cambrian Vision Encoders Match-Up”（Tong 等，2024）在一个视觉为中心的基准（CV-Bench）上测试了 20+ 个编码器。排行榜前列是 DINOv2 和 SigLIP 的混合；CLIP 在中游；ImageBind 和 ViT-MAE 在下游。从 CLIP ViT-L 到 DINOv2 ViT-g/14 在 CV-Bench 上的差距约为 5–7 个点。

到 2026 年，开源 VLM 的默认编码器是 SigLIP 2 SO400m/14（用于语义和稠密特征），在需要分割/定位时有时会与 DINOv2 ViT-g/14 特征拼接（Cambrian 的 “Spatial Vision Aggregator” 有这类做法）。

### 轴 2：连接器设计差异微乎其微

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出相同结论：在固定视觉 token 数下，连接器架构几乎无差异。对 patch 做均值池化后再用 2 层 MLP，其表现与在相同 token 预算下的 32-query Q-Former 相差在 1 点以内。

真正重要的是视觉 token 的数量。更多的视觉 tokens = 更多的 LLM 计算 = 更好表现，直到边际效应递减。每图 64 tokens 对 OCR 来说太少。对于大多数开源 VLM，576–1024 tokens 是甜区。2048+ 只有在处理文档和图表时才有帮助。

Q-Former vs MLP 本质上是成本问题，不是质量问题：Q-Former 把 token 数限制在 32–64，不受图像分辨率影响；MLP 会输出所有 patch token。对于高分辨率输入，Q-Former 可以为 LLM 节省上下文；对于低分辨率，差异是噪声。

### 轴 3：LLM 大小决定上限

把 LLM 从 7B 翻倍到 13B，会在几乎所有 VLM 论文中可靠地增加 2–4 点 MMMU。到 70B 时大多数基准会达到饱和。VLM 的多模态推理上限就是 LLM 的文本推理上限——视觉编码器只能喂信息，不能替代语言模型的推理能力。

这也是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro 和 ScreenSpot-Pro 上表现抢眼：语言“脑”足够庞大。一个 7B 的 VLM 无法通过巧妙的连接器设计来替代 70B 的 VLM。

### 轴 4：数据 —— 详尽的人类描述胜过蒸馏

Molmo + PixMo（Deitke 等，2024）是每个人都应该读的 2024 年结果。Allen AI 让人工标注者用 1–3 分钟的密集语音转文本流程来描述图像，得到 712K 张密集标注图像。训练数据中没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11/11 个基准上击败了 Llama-3.2-90B-Vision。差异不是架构，而是描述质量。详尽的人类描述每张图包含的有用信息是网页短描述的 5–10 倍，且在事实性上比 GPT-4V 蒸馏更可靠（后者会产生幻觉）。

ShareGPT4V（Chen 等，2023）和 Cauldron（Idefics2）也采用了混合的人类 + GPT-4V 描述策略。趋势很明确：到 2026 年的前沿，描述密度 > 描述数量 > 蒸馏的便利性。

### 轴 5：分辨率及其调度

Idefics2 的消融：384 -> 448 添加 1–2 点。448 -> 980（通过图像拆分，AnyRes）在 OCR 基准上再增加 3–5 点。固定分辨率训练会在中等精度处平台化；分辨率逐步提升（先 224，最后 448 或原生）能更快收敛且最终表现更好。

Cambrian-1 做了分辨率与 token 的权衡实验：在固定计算下，你可以选择在较低分辨率下更多 tokens，或在较高分辨率下更少 tokens。高分辨率对 OCR 有利；低分辨率 + 更多 tokens 对一般场景理解更优。

2026 年的生产配方：Stage 1 固定在 384 训练，Stage 2 使用动态分辨率并扩展到 1280（针对 OCR 密集任务）。

### Prismatic 的受控比较

Prismatic VLMs（Karamcheti 等，2024）是那篇在所有轴上都做受控比较的论文。相同的 13B LLM、相同的指令数据、相同的评估——每次只变动一个轴。结果：

- 每图视觉 token 数解释了约 60% 的方差。
- 编码器选择解释了约 20%。
- 连接器架构解释了约 5%。
- 其余（数据混合、调度、学习率）解释了剩下的 ~15%。

这是一个粗略的分解，但它是文献中对“我该先做哪个消融”的最清晰答案。

### 2026 年的选择器（Picker）

基于证据，2026 年新项目的默认开源 VLM 配方：

- 编码器：SigLIP 2 SO400m/14，使用原生分辨率并配合 NaFlex；如果需要分割/定位，则与 DINOv2 ViT-g/14 特征拼接以获取稠密特征。
- 连接器：对 patch tokens 使用 2 层 MLP。除非受 token 限制，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，7B 适合成本敏感场景，70B 适合质量优先，根据目标延迟选择。
- 数据：PixMo + ShareGPT4V + Cauldron，并补充任务特定的指令数据。
- 分辨率：动态（长边最小 256，最大 1280 像素）。
- 调度：Stage 1 对齐（仅 projector），Stage 2 全量微调，Stage 3 任务特定微调。

上述每一项默认设置都可在文末引用的论文的某个消融实验中找到依据。

## 使用方法

`code/main.py` 是一个消融表解析器和配方选择器。它编码了 MM1 和 Idefics2 的消融表（已浓缩），并允许你查询：

- “给定预算 X 和任务 Y，哪个配方胜出？”
- “如果我在 7B Llama 上把 SigLIP 换成 CLIP，预期的 MMMU 差值是多少？”
- “对于 80% 置信度，第一个应该消融哪个轴？”

输出是按排名的配方列表，带有预期基准差值和“先消融”建议。

## 部署输出

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务混合、计算预算和延迟目标，它会输出完整的配方（编码器、连接器、LLM、数据混合、分辨率调度），并引用支持该选择的消融实验。能阻止工程师每次新项目启动时都去“重新发明” Idefics2 的消融表。

## 练习

1. 阅读 MM1 第 3.2 节。对于固定的 2B LLM 和 5000 万图像预算，哪个编码器胜出？在 13B LLM 下答案会翻转吗？为什么？

2. Cambrian-1 发现将 DINOv2 + SigLIP 拼接在视觉为中心的基准上优于单独使用任一者，但在 MMMU 上没有信号。预测哪些基准会有提升，哪些会保持平稳。

3. 你的目标是一个在 2B LLM 上运行的移动 UI 代理。选择编码器、连接器、分辨率和数据混合。用具体的消融表证明每个选择的合理性。

4. Molmo 发布了 4B 和 72B 模型。4B 在与封闭的 7B VLMs 竞争时有竞争力；72B 在 11/11 个基准上击败 Llama-3.2-90B-Vision。这对 LLM 大小饱和假设说明了什么？

5. 设计一个消融表以在 7B VLM 上分离数据混合质量与编码器质量。至少需要多少次训练运行？提出四个轴的设定。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|----------|
| Ablation | “转动一个旋钮” | 进行多次训练运行，每次仅在设计空间的一个轴上有差异，其他条件保持不变 |
| Connector | “桥” / “投影器” | 可训练模块，将视觉编码器输出映射到 LLM 的 token 空间（如 MLP、Q-Former、Perceiver） |
| Detailed human caption | “密集描述” | 多句的人类撰写描述（通常 80–300 tokens），比网页 alt 文本更丰富 |
| Distillation | “GPT-4V 描述” | 由更强大的专有 VLM 生成的训练数据；方便但易继承幻觉问题 |
| AnyRes / dynamic res | “AnyRes / 动态分辨率” | 通过平铺或 M-RoPE 等策略将高于编码器原生分辨率的图像馈入模型 |
| Resolution ramp | “分辨率课程/调度” | 从低分辨率开始并逐步提升的训练调度，加速对齐学习 |
| Vision-centric bench | “视觉为中心的基准（CV-Bench / BLINK）” | 强调细粒度视觉感知而非以语言推理为主的评估 |
| PixMo | “Molmo 的数据” | Allen AI 的 712K 张密集标注图像数据集；通过语音转文本产生密集描述 |

## 延伸阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)  
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)  
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)  
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)  
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)