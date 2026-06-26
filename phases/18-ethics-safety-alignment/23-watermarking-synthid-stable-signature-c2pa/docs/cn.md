# Watermarking — SynthID, Stable Signature, C2PA

> 三种技术构成了 2026 年 AI 生成内容溯源的结构。SynthID（Google DeepMind）——图像水印在 2023 年 8 月推出，文本+视频在 2024 年 5 月（Gemini + Veo），文本在 2024 年 10 月通过 Responsible GenAI Toolkit 开源，统一的多媒体检测器在 2025 年 11 月随 Gemini 3 Pro 推出。文本水印通过对下一个 token 的采样概率进行不可察觉的微调来嵌入信号；图像/视频水印则能在压缩、裁剪、滤镜、帧率变化下存活。Stable Signature（Fernandez 等，ICCV 2023，arXiv:2303.15435）——对潜在扩散模型的解码器进行微调，使每个输出都包含固定消息；在裁剪到 10% 内容时生成图像的检测率在 FPR<1e-6 下仍 >90%。后续论文 “Stable Signature is Unstable”（arXiv:2405.07145，2024 年 5 月）指出：通过微调可以移除水印同时保留图像质量。C2PA——加密签名的防篡改元数据标准（C2PA 2.2 Explainer 2025）。水印与 C2PA 互为补充：元数据可以被剥离但携带更丰富的溯源信息；水印能在转码过程中保持但携带的信息更少。

**Type:** 构建
**Languages:** Python（stdlib，token-watermark embed + detect）
**Prerequisites:** Phase 10 · 04（采样），Phase 01 · 09（信息理论）
**Time:** ~75 分钟

## 学习目标

- 描述基于 token 的水印（SynthID-text 风格）及其可检测的机制。
- 描述 Stable Signature 及 2024 年的移除攻击如何破坏它。
- 说明 C2PA 的作用以及为何它与水印互为补充。
- 描述关键限制：模型特定信号、在改写下的鲁棒性、以及保持语义的不破坏攻击（arXiv:2508.20228）。

## 问题背景

2023–2024 年间，深度伪造和 AI 生成内容在政治和消费场景中大规模出现。水印是提议的技术溯源信号：在生成时标记内容，后续可检测。到 2025 年的证据表明：没有任何水印是无条件鲁棒的，但与 C2PA 元数据层叠使用时，两者结合能提供可用的溯源链条。

## 概念

### 文本水印（SynthID-text 风格）

Kirchenbauer 等人 2023 年提出的机制，由 Google 工业化：

1. 在每个解码步，根据前 K 个 token 哈希，生成词汇表的伪随机划分为“绿色（green）”和“红色（red）”集合。
2. 通过在绿色 token 的 logits 上加上 δ 来偏置采样，倾向于选择绿色集合。
3. 生成文本中绿色 token 的比例高于随机概率。

检测：对每个前缀重新哈希，统计生成文本中绿色 token 的数量，计算 z-score。水印文本的 z-score > 0，人工文本约为 0。

属性：
- 对读者不可察觉（δ 很小，质量损失微弱）。
- 在拥有词汇划分函数的前提下可检测。
- 对改写不鲁棒——重写文本会破坏信号。

SynthID-text 在 2024 年 10 月通过 Google 的 Responsible GenAI Toolkit 开源。

### Stable Signature（图像）

Fernandez 等人，ICCV 2023。对潜在扩散模型的解码器进行微调，使每个生成图像在潜在表示中包含固定的二进制消息。检测器通过神经解码器从潜在向量中解码该消息。即便图像被裁剪到 10% 内容，检测率在 FPR<1e-6 的情况下仍 >90%。

2024 年 5 月的 “Stable Signature is Unstable”（arXiv:2405.07145）：对解码器进行微调可以移除水印同时保持图像质量。对生成后进行的对抗性微调代价低；因此该水印在对抗性鲁棒性上有限。

### SynthID 统一检测器（2025 年 11 月）

随 Gemini 3 Pro 一同发布：一套多媒体检测器，可在单一 API 中读取来自文本、图像、音频和视频的 SynthID 信号。统一了 Google 的溯源栈。

### C2PA

内容溯源与真实性联盟（Coalition for Content Provenance and Authenticity）。加密签名的防篡改元数据标准。C2PA 2.2 Explainer（2025）。C2PA 清单记录溯源声明（谁创建、何时创建、做了哪些转换），并由创建者的密钥签名。

与水印互补：
- 元数据可以被剥离；水印则不易被移除。
- 元数据信息丰富（完整的溯源链）；水印仅携带比特位信息。
- C2PA 依赖平台采纳；水印可以自动嵌入生成内容。

Google 在搜索、广告和 “About this image” 中集成了两者。

### 限制

- **模型特定。** SynthID 仅对启用 SynthID 的模型进行水印。来自未启用 SynthID 的模型的生成不会被水印，因此“无 SynthID 信号”并不等同于真实性证明。
- **改写（paraphrase）。** 文本水印无法抵抗保持语义的改写。
- **变换攻击。** arXiv:2508.20228（2025）展示了保持语义的攻击，这类攻击可破坏文本水印和许多图像水印。
- **微调移除。** 正如 “Stable Signature is Unstable” 所示，生成后对解码器的微调可移除嵌入的水印。

### 欧盟 AI 法案 第 50 条

AI 生成内容标注的透明度守则（草案）：第一稿 2025 年 12 月，第二稿 2026 年 3 月，预计最终稿在 2026 年 6 月（参见 [European Commission status page](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)）。截至 2026 年 4 月，守则仍为草案，时间表可能变动。监管层要求技术层遵从。深度伪造必须被标注。

### 在 Phase 18 中的位置

第 22–23 课讨论模型输出的内容（私人数据、溯源信号）。第 27 课涵盖训练数据治理。第 24 课讨论要求这些技术措施的监管框架。

## 使用方式

`code/main.py` 构建了一个玩具的文本水印。token 为整数 0..N-1；水印采样会偏向哈希定义的绿色集合。检测器计算绿色 token 的 z-score。你可以在 1000-token 的生成上观察检测效果，观察改写如何破坏信号，并测量人工文本的假阳性率。

## 部署结果产物

本课产出 `outputs/skill-provenance-audit.md`。在给定一个带有溯源声明的内容部署时，该审计会评估：水印机制（如有）、C2PA 签名链（如有）、每者的对抗性鲁棒性以及按模态的覆盖情况。

## 练习

1. 运行 `code/main.py`。报告水印 1000-token 生成与人工创作文本的 z-score。确定在 95% 置信阈值下的假阳性率。

2. 实现一个改写攻击，用同义词替换 30% 的 token。重新测量 z-score。

3. 阅读 Kirchenbauer et al. 2023 第 6 节关于鲁棒性的讨论。为什么文本水印在改写下失效，但图像水印能在裁剪下存活？

4. 设计一个部署，结合 SynthID-text + C2PA 元数据。描述消费者看到的溯源链。指出每个组件的一个失效模式。

5. 2024 年的 “Stable Signature is Unstable” 结果表明微调可移除图像水印。设计一个部署控制来限制该攻击 —— 例如，要求对微调后的检查点进行签名发布。

## 关键词

| 术语 | 外界说法 | 实际含义 |
|------|----------|----------|
| SynthID | “Google 的水印” | 跨模态的溯源信号；文本、图像、音频、视频 |
| Token watermark | “Kirchenbauer 风格” | 通过偏置采样的文本水印，可通过绿色 token 的 z-score 检测 |
| Stable Signature | “图像水印” | 基于微调解码器的水印；ICCV 2023 |
| C2PA | “元数据标准” | 加密签名的防篡改溯源元数据 |
| Paraphrase robustness | “改写会不会破坏它” | 文本水印的属性；当前受限 |
| Fine-tune removal | “对抗性去水印” | 通过对解码器微调移除图像水印的攻击 |
| Cross-modal detector | “统一的 SynthID” | 2025 年 11 月推出的跨模态统一 API |

## 延伸阅读

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — token 水印机制的原论文
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — 图像水印论文
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — 关于移除攻击的论文
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — 跨模态水印项目页面
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — 元数据标准文档