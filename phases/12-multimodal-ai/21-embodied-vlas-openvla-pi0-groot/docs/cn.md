# Embodied VLAs: RT-2, OpenVLA, π0, GR00T

> 第一次让模型从网站上读取食谱并在厨房机器人上执行的是 RT-2（Google DeepMind，2023 年 7 月）。RT-2 将动作离散化为文本 token，共同微调（co-fine-tune）一个 VLM，使用网络数据和机器人动作数据，证明了网络级别的视觉-语言知识可以迁移到机器人控制。OpenVLA（2024 年 6 月）发布了开源 7B 参考实现。Physical Intelligence 的 π0 系列（2024–2025）加入了流匹配（flow-matching）动作专家。NVIDIA 的 GR00T N1（2025 年 3 月）在规模化的人形机器人上实现了双系统（System 1 / System 2）控制。VLA 原语 —— 视觉-语言-动作（vision-language-action），即一个能看、能读、能做的单一模型 —— 是当前这阶段理解模型与第 15 阶段自主系统之间的桥梁。

**Type:** 学习  
**Languages:** Python（stdlib、action tokenizer + VLA 推理骨架）  
**Prerequisites:** Phase 12 · 05 (LLaVA)，Phase 15（自主系统，已引用）  
**Time:** ~180 分钟

## 学习目标

- 描述动作标记化（action tokenization）：离散箱编码（RT-2）、FAST 高效动作 token、连续流匹配动作（π0）。
- 解释为什么在网络数据与机器人数据上联合微调可以将通用知识迁移到新任务上。
- 对比 OpenVLA（开源 7B Llama + VLM）、π0（流匹配）和 GR00T N1（双系统）在相同机器人任务上的表现。
- 说出 Open X-Embodiment 数据集的名称及其作为 RT-X 训练语料库的作用。

## 问题背景

自 1970 年代以来，一个能根据自然语言指令做家务的机器人就是研究目标。2020 年代的回答是：视觉-语言-动作（VLA）模型。使用与 VQA 相同的 VLM 架构，但输出是动作（关节力矩、末端执行器位姿、离散命令）而不是文本。

针对 VLA 的特有挑战：

1. 动作空间是连续的（关节角度、力）且维度高（7 自由度臂 + 3 自由度夹爪 = 每步 10 维，在 30 Hz 下）。
2. 机器人专用训练数据稀缺。Open X-Embodiment 约有 ~1M 条轨迹；网络图文数据则超过 50 亿条。
3. 控制频率很重要。30 Hz 的控制循环意味着每个动作只有 33 ms 的预算。
4. 安全性。错误动作会损坏硬件、伤害人或破坏财产。

## 概念

### 动作标记化（RT-2）

RT-2 的关键技巧：将每个关节目标表示为量化的文本 token。把归一化的 [-1, 1] 范围离散为 256 个箱（bins），将每个箱映射到词表 ID。一个 10 自由度的动作在每个控制步会变成 10 个 token。

共同微调一个 PaLM-X VLM，训练数据混合包括：

- 网络图像-文本对（captioning、VQA）。
- 机器人示例，动作作为 token 表示。

模型看到 “pick up the red cube”（语言）→ 图像（视觉）→ 10-token 的动作序列（离散化的关节目标）。网络预训练保留了通用知识迁移能力：RT-2 即使在训练数据中没有出现 “fast-moving”，也能执行 “move towards the fast-moving object” 之类的指令。

RT-2 论文中的推理频率为 3–5 Hz，受限于 VLM 的自回归解码。

### OpenVLA — 开源 7B 参考实现

OpenVLA（Kim 等人，2024 年 6 月）是开源权重的 RT-2 等价实现。采用 7B Llama 主干，DINOv2 + SigLIP 双视觉编码器，动作标记化使用 256 个箱。

在 Open X-Embodiment（22 台机器人、970k 条轨迹）上训练。支持 LoRA 微调以适配新机器人。

推理：在 A100 上量化后可达 4–5 Hz。够用于慢速操作，但不足以用于高频控制。

### FAST tokenizer — 更快的动作解码

Pertsch 等人（2024）指出离散箱标记化效率低下 —— 大多数动作集中在箱空间的一小部分。FAST（Frequency-domain Action Sequence Tokenizer）通过对动作序列做 DCT 并量化系数来压缩动作序列。

一个 30 步的动作轨迹可被压缩为约 10 个 FAST token，而不是 300 个离散箱 token。推理速度提升 3–5 倍，并且质量无损。

### π0 与流匹配动作

Physical Intelligence 的 π0（Black 等人，2024 年 10 月）用流匹配动作专家替代离散动作 token：

- 一个小型动作 transformer 读取 VLM 的隐藏态，并通过 rectified flow 输出一个连续的 50 步动作序列。
- 动作头用流匹配（flow-matching）损失训练；VLM 的预训练保持不变。
- 推理：在 ~5 步去噪后输出完整动作序列，等效于 50 Hz 控制。

π0 的宣称：在广泛的操作任务上优于 OpenVLA 和 Octo。连续动作的表述保留了离散化会破坏的平滑性。

π0.5 与 π0-FAST 为增量升级。π0-FAST 将 FAST 标记化与流匹配相结合。

### GR00T N1 — 面向人形机器人的双系统

NVIDIA 的 GR00T N1（2025 年 3 月）为人形机器人（>30 DOF，全身）设计：

- System 2：大型 VLM 读取场景 + 指令，产生高层子目标，频率约 1 Hz。
- System 1：小型动作头 transformer 基于子目标产生低层 50–100 Hz 的关节命令。

该划分映射到 Kahneman 的快慢思维：System 2 做规划，System 1 执行。优点是大型 VLM 的慢速规划不会阻塞快速控制；System 1 保持小以降低延迟。

GR00T N1.7（2025 年末）改进了数据规模扩展。GR00T 使用来自 Omniverse 的仿真到实物（sim-to-real）数据进行微调。

### Open X-Embodiment

训练数据集。RT-X（2023 年 10 月）汇集了 22 个数据集，覆盖 1M 条跨 22 台机器人的轨迹。Open X-Embodiment 是大家使用的语料库：

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table 等。
- 每个样本包含：（机器人状态、相机视图、指令、动作序列）。
- 训练规范：统一动作空间、归一化关节范围、调整相机尺寸等。

OpenVLA 与 π0 都在 Open X-Embodiment 上训练。对特定机器人存在的领域差距通常通过在 100–1000 条特定任务示例上做 LoRA 微调来关闭。

### 联合微调（Co-fine-tuning）与仅机器人训练

联合微调将网络 VQA 数据与机器人轨迹混合训练。比例很重要：VQA 太多会让模型忘记动作；机器人数据太多会让模型失去通用知识。

RT-2 的比例约为 ~1:1。OpenVLA：大约 0.5:1（web-to-robot）。π0 类似。精确比例是需要针对数据集大小调优的超参数。

仅机器人训练会产生任务特定模型，无法应对分布外的指令。联合微调的差别体现在能否处理 “pick up the red cube（示例中的表述）” 与 “pick up the third largest object from the left（新颖措辞）” 的区别上。

### 安全与动作限制

每个生产级 VLA 都配备了：

- 硬性关节极限（不能施加超出规格的力矩）。
- 速度限制（软裁剪）。
- 工作空间边界（末端执行器不能离开桌面）。
- 针对新任务的人机在环（human-in-the-loop）批准。

这些检查置于 VLA 之外，作为控制层的安全约束。VLA 的输出是建议，而非最终命令。

## 实践

`code/main.py`：

- 实现了 256-bin 的动作标记化与解标记化（de-tokenization）。
- 草绘了基于 DCT + 量化的 FAST 标记器。
- 比较了每步动作的 token 数（离散箱、FAST、连续流）。
- 打印 RT-2 → OpenVLA → π0 → GR00T 的谱系摘要。

## 部署建议

本课产出 `outputs/skill-vla-action-format-picker.md`。给定一个机器人任务（操作、导航、人形全身），选择离散箱 + RT-2、FAST + OpenVLA、流匹配 + π0，或双系统 + GR00T 之一。

## 练习

1. 一个在 30 Hz 控制率下工作的 10 自由度机械臂。256-bin 的离散箱标记化每秒会发出多少个 token？一个 7B VLM 能否跟上？

2. FAST 标记化将 30 步轨迹压缩为 ~10 个 token。如果轨迹包含高频运动（例如打鼓），用户会失去什么信息？

3. π0 的流匹配头在 ~5 步中去噪。将其吞吐量与 OpenVLA 在 4–5 Hz 的自回归解码进行比较。

4. GR00T 的 System 1 / System 2 划分对应 Kahneman 的模型。提出一种可能有助于双足行走的不同划分（System 3？）。

5. 阅读 Open X-Embodiment 第 4 节（数据集整理）。说出防止领域泄漏的三条整理规则。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| VLA | "Vision-language-action" | 接收图像 + 指令并输出动作命令的模型 |
| Action tokenization | "Discrete bins" | 将连续的关节目标量化为每维 256 个箱，每个箱对应一个词表 ID |
| FAST tokenizer | "Frequency action tokens" | 对轨迹做 DCT 并量化系数，将 30 步轨迹压缩为 ~10 个 token |
| Co-fine-tune | "Mix web + robot" | 在网络 VQA 数据与机器人示例上混合训练，以保留通用知识 |
| Flow-matching action head | "π0 continuous output" | 小型 transformer 通过 rectified flow 输出 50 步连续动作序列 |
| System 1 / System 2 | "Dual-system control" | 大型 VLM 慢速规划，小型动作头快速执行；GR00T 的模式 |
| Open X-Embodiment | "RT-X dataset" | 1M 条轨迹的跨机器人数据集；训练语料库 |

## 延伸阅读

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)  
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)  
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)  
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)  
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)