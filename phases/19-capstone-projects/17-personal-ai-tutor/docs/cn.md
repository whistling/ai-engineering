# Capstone 17 — 个人 AI 导师（自适应、多模态、带记忆）

> Khanmigo（可汗学院）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 都在 2026 年发布了可规模化的自适应多模态辅导产品。共同形态是苏格拉底式策略（绝不直接给出答案）、在每次交互后更新的学习者模型（类似贝叶斯知识跟踪）、语音 + 文本 + 拍照数学输入、课程图检索、间隔重复调度，以及针对年龄的严格安全过滤。结业目标是发布一个学科特定的导师（K-12 代数或入门 Python），运行为期两周、10 名学习者的有效性研究，并通过内容安全审计。

**Type:** 结业项目  
**Languages:** Python (后端，learner model)，TypeScript (Web 应用)，SQL (使用 Postgres + Neo4j 实现课程图)  
**Prerequisites:** Phase 5（NLP），Phase 6（语音），Phase 11（LLM 工程），Phase 12（多模态），Phase 14（agent），Phase 17（基础设施），Phase 18（安全）  
**Phases exercised:** P5 · P6 · P11 · P12 · P14 · P17 · P18  
**Time:** 30 小时

## 问题

自适应辅导一度是教育技术的研究小众领域。到 2026 年它已成为面向消费者的产品。Khanmigo 在大多数美国学区部署；Duolingo Max 的月活跃用户达数千万；Google 的 LearnLM / Gemini for Education 支撑着 Google Classroom 中的辅导；Quizlet Q-Chat 与抽认卡并存；Synthesis Tutor 在“面向好奇孩子的导师”场景中走红。共有的要素包括：多模态输入（键入、说话、拍照识别方程）、苏格拉底式教学法（先问问题，再解释）、在每次交互后更新的学习者模型，以及严格的年龄适配安全策略。

你的任务是为特定人群构建其中之一。衡量标准是实际的有效性研究：在两周内对 10 名学习者进行前测和后测。语音交互要自然（参见 capstone 03 子栈）。记忆要尊重隐私。安全过滤必须通过针对 K-12 的 COPPA 意识化红队审查。

## 概念

四个组件。Tutor policy（导师策略）是一个苏格拉底循环：当学习者要求直接给出答案时，策略会提出引导性问题；当他们答对时，转到下一个概念；当卡住时，提供分层提示（scaffolded hint）。Learner model（学习者模型）采用贝叶斯知识跟踪（或其简单变体），在每次交互后更新课程节点的掌握概率。Curriculum graph（课程图）是一个 Neo4j，概念间有先决边；策略在图上行走以选择下一个概念。Memory（记忆）是情节（episodic）+ 语义（semantic）存储（agentmemory 风格），记录过去互动、错误模式和偏好。

用户体验是多模态的。文本输入用于键入答案。语音输入使用 LiveKit + Whisper（重用 capstone 03）。拍照数学使用 dots.ocr 或 PaliGemma 2。语音输出使用 Cartesia Sonic-2。安全使用 Llama Guard 4 加上年龄适配过滤（屏蔽成人内容、暴力、自伤）。记忆保留策略遵循 COPPA。

有效性研究是交付物：10 名学习者，前测/后测，两周。报告学习增益差值与置信区间。与非自适应基线（相同内容按线性顺序呈现，无导师策略）做比较。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 科目选择：K-12 代数 或 入门 Python（任选其一深入实现）  
- 导师策略：LangGraph 运行在 Claude Sonnet 4.7 上（带提示缓存）  
- 学习者模型：贝叶斯知识跟踪（经典）或用 FSRS 做间隔复习调度  
- 课程图：Neo4j 的概念节点 + 先决边 + OER 内容附件  
- 记忆：agentmemory 风格的持久向量存储 + 情节存储 + 语义存储  
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（重用 capstone 03 子栈）  
- 拍照数学：dots.ocr 或 PaliGemma 2 做方程识别  
- 安全：Llama Guard 4 + 自定义年龄适配过滤器  
- 评估：使用 Bloom 级别的问题生成、前/后测工具链、有效性研究工具

## 构建步骤

1. Curriculum graph（课程图）。构建一个包含 50–150 个概念节点的 Neo4j（例如从“数轴”到“求根公式”的 K-12 代数），并建立先决关系边。为每个节点附加 OER 内容（Open Textbook、OpenStax 等）。

2. Learner model（学习者模型）。初始化贝叶斯知识跟踪，设置先验参数：guess、slip、learn-rate。每次交互后更新每个概念的掌握概率。为每个学习者持久化状态。

3. Tutor policy（导师策略）。用 LangGraph 构建节点：`read_signal`（学习者的答案是正确 / 部分正确 / 卡住？）、`select_concept`（在课程图上行走选择优先级最高的概念）、`scaffold`（苏格拉底式提示）、`update_mastery`。

4. Memory（记忆）。每次交互写入情节存储。把重复错误模式和偏好提升为语义记忆。实施 COPPA 感知的保留策略：自动在 1 年后删除，家长可访问并请求删除。

5. 语音路径。将 LiveKit Agents worker 连接到导师策略。ASR 使用 Whisper-v3-turbo。TTS 使用 Cartesia Sonic-2。支持打断（barge-in）（重用 capstone 03 的机制）。

6. 拍照数学路径。上传或拍照；运行 dots.ocr 或 PaliGemma 2 来识别方程；以结构化输入的形式喂入导师策略。

7. 安全。所有模型输出都通过 Llama Guard 4 + 年龄适配过滤（屏蔽自伤、成人内容、暴力）。记忆访问按 learner ID 范围进行保护；为家长提供删除入口。

8. 有效性研究。招募 10 名学习者，进行前测（标准化 30 题基线）；两周导师使用期（每周 3 次会话）；后测。设置一个对照组（10 人，非自适应、随机概念顺序）。比较学习增益。

9. 每周进度报告。为每位学习者自动生成 PDF 总结：练习主题、掌握轨迹和建议的下一步学习内容。

## 使用示例

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

（注：上面为交互示例；在实现中应将交互文本本地化为目标语言并保留概念标识与掌握更新逻辑。）

## 发布交付物

`outputs/skill-ai-tutor.md` 是交付物。内容应包含学科特定的自适应导师（多模态输入、学习者模型、记忆、安保机制）与衡量有效性的结果。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Learning gain delta | 在两周、10 名学习者研究中的前测/后测差值 |
| 20 | Socratic fidelity | 在对话样本上的评分量表 |
| 20 | Multimodal UX | 语音 + 照片 + 文本端到端连贯性 |
| 20 | Safety + privacy posture | Llama Guard 4 的通过率 + COPPA 感知的保留策略 |
| 15 | Curriculum breadth and graph quality | 概念覆盖率 + 先决图一致性 |
| **100** | | |

## 练习

1. 运行有效性研究：有自适应学习者模型组和无自适应的基线组（随机概念顺序）。报告差值。预期自适应模型更优，但关注效果大小。

2. 增加一个多模态探针：将同一概念问题分别以文本、语音和拍照方式呈现。测量学习者是否在偏好模态下更快收敛。

3. 构建家长仪表盘：展示练习主题、掌握轨迹、即将学习的概念和任何安全事件（任何护栏触发）。确保符合 COPPA。

4. 添加语言切换模式：导师接受西班牙语输入并用西班牙语教学。评估 X-Guard（安全过滤器）的覆盖情况。

5. 强化记忆隐私防护：验证学习者 A 无法通过语音片段重摄入攻击访问学习者 B 的数据。记录尝试访问并触发告警。

## 关键术语

| 术语 | 人们如何说 | 实际含义 |
|------|-----------|---------|
| 苏格拉底式策略（Socratic policy） | "问，不要直接给答案" | 导师提出引导性问题而不是直接给出答案 |
| 贝叶斯知识跟踪（Bayesian knowledge tracing） | "BKT" | 针对每个概念的掌握概率的经典学习者模型方程 |
| FSRS | "Free Spaced Repetition Scheduler" | 2024 年流行的间隔重复调度算法，优于 SM-2 |
| Curriculum graph（课程图） | "概念 DAG" | 使用 Neo4j 表示、带有先决边的概念图 |
| 情节记忆（Episodic memory） | "每次交互的日志" | 每次交互都被存储以备检索 |
| 语义记忆（Semantic memory） | "学到的模式存储" | 从情节中压缩提取出的重复错误和偏好 |
| COPPA | "儿童隐私法" | 美国法律，限制对 13 岁以下儿童的数据收集 |

## 延伸阅读

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — K-12 导师参考  
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 语言学习导师参考  
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管参考模型  
- [Quizlet Q-Chat](https://quizlet.com) — 替代参考  
- [Synthesis Tutor](https://www.synthesis.com) — 创业公司参考  
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复调度实现  
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 学习者模型经典文献  
- [LiveKit Agents](https://github.com/livekit/agents) — 语音栈参考