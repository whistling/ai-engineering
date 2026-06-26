# 音频-语言模型：从 Whisper 到 Audio Flamingo 3 的演进

> Whisper（Radford 等人，2022 年 12 月）解决了语音识别 —— 680k 小时弱监督多语言语音、一个简单的编码器-解码器 Transformer、一个随后每个 ASR 发布都会引用的基准。但识别并不是推理。问“这段录音里有哪些乐器”或“说话者表达了什么情绪”或“第 3 分钟发生了什么”需要音频理解，而不仅仅是转录。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025 年 7 月）逐步构建了这套堆栈：保留 Whisper 级别的编码器，接上 Q-former，在音频-文本指令数据上训练，加入思维链推理。本课讲述这条演进路径。

**Type:** 构建  
**Languages:** Python（标准库、log-Mel 频谱 + 音频 Q-former 骨架）  
**Prerequisites:** 第6阶段（语音和音频）、第12阶段 · 03（Q-Former）  
**Time:** ~180 分钟

## 学习目标

- 从波形计算 log-Mel 频谱：窗函数、FFT、滤波器组、对数变换。  
- 比较编码器选项：Whisper 编码器、BEATs、AF-Whisper 混合体。何时各自占优。  
- 构建音频 Q-former：N 个可学习查询对频谱帧做交叉注意力。  
- 解释级联（Whisper 然后 LLM）与端到端音频-LLM 训练：为什么端到端在推理能力上更易扩展。

## 问题背景

语音识别被 Whisper 解决了。把音频变成文本是商品化的。但“商品化”在转录处止步。如果模型不能就它听到的内容进行推理 —— 时间戳、说话者、情绪、音乐结构、环境声 —— 单纯的转录无法驱动产品特性。

三条明显路径：

1. 级联：Whisper 转录，LLM 在转录文本上进行推理。适用于纯语音场景。对音乐、环境音、多说话者重叠、情绪判断会失效。

2. 端到端音频-LLM：音频编码器直接将音频 token 输入 LLM，跳过转录。保留声学信息（情绪、说话者、环境）。需要新的训练数据。

3. 混合：音频编码器 + 文本解码器，既能转录也能推理。Qwen-Audio 和 Audio Flamingo 采用这一路径。

## 概念

### Log-Mel 频谱：输入特征

每个音频编码器都从同样的特征开始：log-Mel 频谱。

1. 重采样到 16 kHz。  
2. 采用 25 ms 窗、10 ms hop 做短时傅里叶变换（STFT）。  
3. 取 FFT 结果的幅度。  
4. 应用 Mel 滤波器组（通常 80 个滤波器，0–8000 Hz 对数间隔）以映射到感知频率。  
5. 对数压缩（log(1 + x)）以处理动态范围。

结果：形状为 (T, 80) 的二维数组，T 为时间帧数。对于 30 秒片段、100 Hz 帧率： (3000, 80)。

### Whisper 的编码器

Whisper 的编码器是一个 12 层 ViT 风格的 Transformer，把 log-Mel 频谱当作时间帧序列处理。输出：每个时间帧对应一个隐藏态向量。

在 ASR 场景，Whisper 的解码器是一个带交叉注意力的 Transformer，基于编码器输出生成文本 token。标准的编码器-解码器结构。

对于 ALM（音频-LLM），希望把编码器输出作为另一个 LLM 的输入。常见模式：冻结 Whisper 编码器、训练 Q-former、LLM 冻结或微调。

### BEATs 与音频专用编码器

Whisper 在以语音为主的数据上训练，面对音乐和环境音时表现较弱。

BEATs（Chen 等人，2022）是在 AudioSet 上自监督训练的 Transformer，相比同参数量的 Whisper 更擅长捕捉音乐与环境声音特征。

AF-Whisper（Audio Flamingo 3 的混合体）：将 Whisper 与 BEATs 特征串联作为音频输入。Whisper 提供语言信号，BEATs 提供声学信号。

### 音频 Q-former

与 BLIP-2 的视觉 Q-former 相同模式。固定数量的可学习查询（常见为 32 或 64）对音频编码器的输出帧做交叉注意力。查询最终成为被 LLM 消耗的音频 token。

对齐训练阶段：仅训练 Q-former，在音频-文本对（AudioCaps、Clotho）上使用对比与标注损失。指令阶段：端到端训练，解冻 LLM，在指令数据上训练。

### 演进路径 —— SALMONN、Qwen-Audio、AF3

SALMONN（Tang 等人，2023）：Whisper + BEATs + Q-former + LLaMA。首个具备认真推理能力的开源音频-LLM。在 MMAU 上复合得分约 0.55。

Qwen-Audio（Chu 等人，2023）：相似架构，使用更丰富的数据集训练，针对多轮对话进行了调优。MMAU 约 0.60。

LTU — Listen, Think, Understand（Gong 等人，2023）：使用显式的推理数据，专注于音频片段上的思维链推理。模型体量较小但更专注。

Audio Flamingo 3（Goel 等人，2025 年 7 月）：当前开源 SOTA。8B LLM 骨干（Qwen2 7B），Whisper-large 编码器串联 BEATs，64 查询 Q-former，在 100 万+ 音频-文本指令对上训练。MMAU 0.72，在部分子任务上匹敌闭源前沿。

AF3 还引入了按需的音频思维链：模型可以可选地输出思考 token（例如：“让我先识别这些乐器：...”）再给出最终答案。在启用思维链时，复杂推理任务的准确率提升 3–5 个百分点。

### 级联 vs 端到端

级联管道：

1. Whisper 转录音频 → 文本。  
2. LLM 在文本上做推理。

对于“总结这期播客”类任务非常有效。但会在以下场景失败：
- “这首歌的情绪是什么？”——情绪体现在声学中，而不是文字。  
- “谁在说话，是 Alice 还是 Bob？”——需要说话者识别。  
- “爆炸发生在第几秒？”——时间定位在转录中丢失。  
- “这是人声还是合成音频？”——深度伪造检测需要声学特征。

端到端保留了声学信号。Qwen-Audio 和 AF3 能原生处理音乐、环境音和情绪。

### 2026 年生产建议

对于新的音频理解产品：

- 如果：目标仅是转录，无音乐、无需情绪推断 —— 采用级联。  
- 如果：涉及音乐、情绪、多说话者或复杂音频推理 —— 采用 AF3 / Qwen-Audio 家族的端到端方案。

级联更便宜、更简单。端到端更有能力。

### MMAU —— 音频推理基准

MMAU（大规模多模态音频理解）是 2024–2025 年的音频推理基准：

- 包含 10,000 条跨语音、音乐、环境声音的音频-文本问答对。  
- 覆盖分类、时间推理、因果推理、开放式问答。  
- 测试级联管道系统性失效的场景。

开源 SOTA（AF3）为 0.72；闭源前沿约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。开源与闭源的差距比 VideoMME 小，表明音频-LLM 正在成熟。

## 使用示例

`code/main.py`：

- 在标准库中实现 log-Mel 频谱计算：窗函数、朴素 DFT、Mel 滤波器组。  
- 音频 Q-former 骨架：给定编码器输出帧，计算 Q、K、V、注意力，并输出 N 个 token。  
- 在一个玩具任务上比较级联与端到端。

## 发布

本课将生成 `outputs/skill-audio-llm-pipeline-picker.md`。给定一个音频任务（转录、音乐标注、情绪推断、多说话者分离、环境分类），该文档会推荐级联、端到端 AF3，或混合方案。

## 练习

1. 计算 30 秒片段在 16 kHz、25 ms 窗、10 ms hop、80 Mel 频带下的 log-Mel 频谱维度。换成 48 kHz 时如何变化？  

2. 为什么 Whisper 在音乐上表现不佳？BEATs 捕捉了哪些 Whisper 未捕捉到的音频特征？  

3. 64 查询的 Q-former 对比 32 查询：在何种任务复杂度下 64 查询更有价值？32 查询在什么场景下能节省计算量？  

4. 阅读 AF3 第 4 节关于按需思考（on-demand thinking）。提出三个在音频上思维链最有帮助的任务。  

5. 使用 AF3 的输出实现一个最小化的说话者划分（diarization）管道。你如何表示/标记说话者变化？

## 关键术语

| 术语 | 常说法 | 实际含义 |
|------|--------|----------|
| Log-Mel spectrogram | “Mel 特征” | 经过 Mel 滤波器组并取对数幅度后的二维（时间，频率）数组 |
| Audio Q-former | “Audio Perceiver” | 从音频编码器输出到固定长度查询、供 LLM 使用的交叉注意力瓶颈 |
| Cascaded | “ASR-then-LLM” | Whisper 转录然后文本 LLM 推理的流水线；会丢失声学信息 |
| End-to-end | “Audio-LLM” | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | “Audio AudioSet encoder” | 在 AudioSet 上自监督训练的 Transformer；对音乐和环境音更强 |
| MMAU | “Audio reasoning bench” | 跨语音、音乐、环境的 10k 问答对；2024 年的评估标准 |
| On-demand thinking | “Audio CoT” | 模型可选地在最终答案前输出推理 token（思维链），能提升 3–5 个点准确率 |

## 深入阅读

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)  
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)  
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)  
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)  
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)