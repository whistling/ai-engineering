# Capstone 15 — 宪法式安全护栏 + 红队演练场

> Anthropic 的 Constitutional Classifiers、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety，以及用于多语言覆盖的 X-Guard 构成了 2026 年的安全分类器栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准对抗评估工具。NeMo Guardrails v0.12 将它们绑定到生产流水线。本结业项目将这些组件串联：在目标应用周围构建分层安全护栏、运行一个自动化红队代理覆盖 6+ 类攻击，并运行一次宪法式自我批判以产出可衡量的无害性改变量。

**Type:** 结业项目  
**Languages:** Python（安全流水线、红队）、YAML（策略配置）  
**Prerequisites:** Phase 10（从零构建 LLM）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（代理）、Phase 18（伦理、安全、对齐）  
**Phases exercised:** P10 · P11 · P13 · P14 · P18  
**Time:** 25 小时

## 问题

到 2026 年，LLM 安全的前沿问题不再是分类器是否有效（大致上是有效的），而是如何在生产应用周围正确组合这些分类器，既不发生过度拒绝，又不留下显而易见的漏洞。Llama Guard 4 处理英文策略违规，X-Guard（132 种语言）处理多语言越狱，ShieldGemma-2 捕捉基于图像的提示注入，NVIDIA Nemotron 3 Content Safety 覆盖企业类别。Anthropic 的 Constitutional Classifiers 则是在训练期间使用的另一种方法，而不是用于推理服务时。

攻击演化同样重要。PAIR 和 TAP 能自动发现越狱。GCG 运行基于梯度的后缀攻击。多轮和代码切换攻击利用代理记忆。任何部署的 LLM 都需要一个红队演练场 —— garak 和 PyRIT 是标准驱动器 —— 并附带文档化的缓解措施和按 CVSS 打分的发现。

你将对目标应用（可选 8B 指令微调模型，或其他结业项目中的 RAG 聊天机器人之一）进行加固，针对它运行 6+ 类攻击，并产出前/后无害性测量结果。

## 概念

安全流水线由五层组成。  
- 输入清理（Input sanitize）：剥离零宽字符、解码 base64/rot13、规范化 Unicode。  
- 策略层（Policy layer）：NeMo Guardrails v0.12 护栏（越域、毒性、PII 提取）。  
- 分类器闸门（Classifier gate）：对英文使用 Llama Guard 4，对非英文使用 X-Guard，对图像提示使用 ShieldGemma-2。  
- 模型（Model）：目标 LLM。  
- 输出过滤（Output filter）：对输出运行 Llama Guard 4、Presidio PII 清洗、必要时执行引用强制。  
- 人在环（HITL）层：被标记为高风险的输出进入 Slack 队列供人工复核。

红队演练场运行在调度器上。PAIR 与 TAP 自主发现越狱；GCG 运行基于梯度的后缀攻击；ASCII / base64 / rot13 编码攻击；多轮攻击（角色扮演采纳、记忆利用）；代码切换攻击（交错使用英语与斯瓦希里语或泰语）。每次运行都生成结构化发现文件，包含 CVSS 评分与披露时间表。

宪法式自我批判运行是训练期干预。收集 1k 个有害尝试的提示，让模型草拟回应，然后根据书面宪法（不伤害规则）对其进行批判，并在批判循环上重训。对持出评估集测量前/后无害性改变量。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

（上述架构图为文本流水线示意；各节点对应前文五层与并行红队演练。）

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard  
- 护栏框架：NeMo Guardrails v0.12 + OPA  
- 红队驱动器：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo  
- 越狱代理：PAIR（Chao 等人，2023）、Tree-of-Attacks（TAP）、GCG 后缀攻击  
- 宪法式训练：Anthropic 风格的自我批判循环 + 基于批判的 SFT（监督微调）  
- PII 清洗：Presidio  
- 目标：一个 8B 指令微调模型或其他结业项目中的 RAG 聊天机器人之一

## 构建步骤

1. 目标搭建。部署一个 8B 指令微调模型在 vLLM（或复用其他结业项目中的 RAG 聊天机器人）。这是受测应用。  
2. 安全流水线封装。将五层流水线包裹到目标周围。验证每层都是可观测的（在 Langfuse 中为每层打点 span）。  
3. 分类器覆盖。加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在小型带标签的数据集上运行以建立基线。  
4. 红队调度器。调度 garak、PyRIT、一个 PAIR 代理、一个 TAP 代理、一个 GCG 运行器、一个多轮攻击器 和 一个代码切换攻击器。每个在独立队列运行。  
5. 攻击套件。六类攻击： (1) PAIR 自动越狱，(2) TAP Tree-of-Attacks，(3) GCG 梯度后缀，(4) ASCII / base64 / rot13 编码，(5) 多轮角色扮演，(6) 多语言代码切换。报告每类的成功率。  
6. 宪法式自我批判。收集 1k 个有害尝试提示。对每条提示，目标生成回应。一个批判者 LLM 根据书面宪法（“不伤害”、“引用证据”、“拒绝非法请求”）评分。被批判者反对的提示会被重写；目标在这些经过批判改进后的对对对对对对对上进行微调。对持出评估集测量前/后无害性。  
7. 过度拒绝测量。跟踪对良性提示集（例如 XSTest）的假阳性率。目标在良性问题上必须保持有帮助。  
8. CVSS 评分。对每个成功越狱按 CVSS 4.0 评分（攻击向量、复杂度、影响）。生成披露时间线和缓解计划。  
9. 演练场自动化。上述所有流程在 cron 上跑；发现写入队列；过度拒绝回归告警发到 Slack。

## 使用示例

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

（示例输出保留为命令行交互格式；在实际运行时会产出结构化发现文件和告警。）

## 交付物

`outputs/skill-safety-harness.md` 是交付物。要求为生产级的分层安全流水线以及可复现的红队演练场，并附带前/后无害性改变量。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Attack-surface coverage | 6+ attack families exercised, 2+ languages |
| 20 | True-positive / false-positive trade-off | Attack block rate vs XSTest benign pass rate |
| 20 | Self-critique delta | Before/after harmlessness on held-out eval |
| 20 | Documentation and disclosure | CVSS-scored findings with timeline |
| 15 | Automation and repeatability | Everything runs on cron with alerts |
| **100** | | |

## 练习

1. 在一个 RAG 聊天机器人上运行 garak 的 prompt-injection 插件，并比较有无输出过滤层时的攻击成功率。  
2. 增加第七类攻击：通过检索到的文档进行间接提示注入。测量为防御这类攻击所需的额外防御。  
3. 实现一个“拒绝但提供替代”模式：当护栏阻止请求时，目标不直接平铺拒绝，而是提供一个更安全的相关答案。测量 XSTest 的变化。  
4. 多语言覆盖缺口：找到 X-Guard 表现不佳的一种语言。提出针对该语言的微调数据集方案。  
5. 在 30B 模型上运行宪法式自我批判，测量改变量是否随模型规模扩展。

## 术语表

| Term | 人们通常如何称呼 | 实际含义 |
|------|-----------------|------------------------|
| 分层安全 (Layered safety) | "Defense in depth" | 在输入、闸门、输出、HITL 等处部署多个护栏 |
| Llama Guard 4 | "Meta 的安全分类器" | 2026 年参考级输入/输出内容分类器 |
| PAIR | "越狱代理" | 关于 LLM 驱动的越狱发现的论文（Chao 等） |
| TAP | "Tree-of-Attacks" | PAIR 的树搜索变体 |
| GCG | "Greedy coordinate gradient" | 基于梯度的对抗后缀攻击 |
| 宪法式自我批判 (Constitutional self-critique) | "Anthropic 风格训练" | 目标先生成 -> 批判者评分 -> 重写 -> 重新训练 |
| XSTest | "良性探测集" | 用于检测过度拒绝回归的基准 |
| CVSS 4.0 | "严重性评分" | 用于安全发现的标准漏洞评分 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练期参考  
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年的输入/输出分类器  
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像与多模态安全  
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考  
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132 语言的多语言安全  
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包  
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 的红队框架  
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 护栏框架文档  
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱代理论文