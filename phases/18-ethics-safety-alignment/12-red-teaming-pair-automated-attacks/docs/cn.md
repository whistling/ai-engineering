# Red-Teaming: PAIR and Automated Attacks

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419)。PAIR — Prompt Automatic Iterative Refinement（提示词自动迭代精炼）— 是经典的自动化黑盒越狱方法。攻击者 LLM 带着红队系统提示词对目标 LLM 迭代地提出越狱提示词，并将尝试和响应累积在其自身聊天历史中作为上下文反馈。PAIR 通常在 20 次查询内成功，效率比 GCG（Zou 等人的基于 token 的梯度搜索）高出好几个数量级，并且不需要白盒访问。PAIR 现在是 JailbreakBench（arXiv:2404.01318）和 HarmBench 的标准基线之一，与 GCG、AutoDAN、TAP 和 Persuasive Adversarial Prompt 并列。

**Type:** 构建  
**Languages:** Python（标准库，针对玩具目标的模拟 PAIR 循环）  
**Prerequisites:** Phase 18 · 01（指令遵循）、Phase 14（代理工程）  
**Time:** ~75 分钟

## 学习目标

- 描述 PAIR 算法：攻击者系统提示、迭代精炼、上下文反馈。
- 解释在目标为黑盒时为什么 PAIR 比 GCG 严格更高效。
- 列举另外四个自动化攻击基线（GCG、AutoDAN、TAP、PAP），并说明每个方法的一个区分特征。
- 描述 JailbreakBench 和 HarmBench 的评估协议，以及在每种协议下“攻击成功率”（ASR）的含义。

## 问题背景

红队测试过去是人工活动。少数专家测试人员构造对抗性提示词并记录哪些有效。这无法扩展：攻击成功率需要统计样本，而且目标模型在每次发布时都在变化。PAIR 将红队工作形式化为一个针对黑盒目标的优化问题。

## 概念

### PAIR 算法

输入：
- 目标 LLM T（我们要攻击的模型）。
- 裁判 LLM J（评分某个响应是否构成越狱）。
- 攻击者 LLM A（红队优化器）。
- 目标字符串 G：例如 “回复时包含 [有害指令]”。
- 预算 K（通常为 20 次查询）。

循环，k 从 1 到 K：
1. 将目标 G 和当前为止的（提示，回复）对历史作为上下文提示给 A。
2. A 生成一个新的提示 p_k。
3. 将 p_k 提交给 T；接收响应 r_k。
4. J 对 (p_k, r_k) 按目标打分。
5. 若分数 >= 阈值，则停止 — 找到越狱。
6. 否则，将 (p_k, r_k) 附加到 A 的历史；继续。

经验结果（NeurIPS 2023）：对 GPT-3.5-turbo、Llama-2-7B-chat 等模型的攻击成功率 >50%；成功所需的平均查询次数在 10–20 次区间。

### 为什么 PAIR 高效

GCG（Zou 等人，2023）通过对抗性 token 后缀做梯度搜索；它需要白盒模型访问并生成不可读的后缀。PAIR 是黑盒的，生成可读的自然语言攻击并能在模型之间迁移。PAIR 的上下文反馈让攻击者能从每次被拒绝中学习；而 GCG 没有等价机制（每次新的 token 更新都需要重新发现先前的进展）。

### 相关的自动化攻击

- **GCG (Zou et al. 2023, arXiv:2307.15043)。** 基于 token 级别的梯度搜索以寻找对抗性后缀。白盒、可迁移，但产生不可读字符串。
- **AutoDAN (Liu et al. 2023)。** 对提示词进行进化搜索，由分层目标引导。
- **TAP (Mehrotra et al. 2024)。** 带剪枝的攻击树（tree-of-attacks）— 在多个 PAIR 风格的 rollout 上分支。
- **PAP (Zeng et al. 2024)。** Persuasive Adversarial Prompts — 将人类说服技巧编码为提示模板。

### JailbreakBench 与 HarmBench

两者（2024 年）规范了评估：

- JailbreakBench（arXiv:2404.01318）。覆盖 10 个 OpenAI 策略类别下的 100 个有害行为。主要指标为攻击成功率（ASR）。需要裁判（例如 GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika 等，2024）。包含 7 个类别下的 510 个行为，含语义和功能性有害测试。对 18 种攻击在 33 个模型上进行比较。

ASR 通常在固定查询预算下报告。对比不同攻击必须匹配预算；在 200 次查询下的 90% ASR 无法与 20 次查询下的 85% ASR 比较。

### 这在 2026 年部署中的重要性

到 2026 年，几乎所有前沿实验室在发布生产模型前都会对其运行 PAIR 和 TAP。ASR 的轨迹会出现在模型卡（Lesson 26）和安全案例附录（Lesson 18）中。该攻击并不稀有 — 它已成为标准基础设施。

### 在 Phase 18 的位置

Lesson 12 是自动化攻击的基础。Lesson 13（多次越狱）是互补的长度利用。Lesson 14（ASCII 艺术 / 视觉）是编码攻击。Lesson 15（间接提示注入）是 2026 年的生产攻击面。Lesson 16 涵盖防御工具对应项（如 Llama Guard、Garak、PyRIT）。

## 使用方法

`code/main.py` 构建了一个玩具 PAIR 循环。目标是一个模拟的分类器，它会拒绝“明显”的有害提示（基于关键词过滤）。攻击者是一个基于规则的精炼器，会尝试释义、角色扮演框架化和编码。裁判对响应进行评分。你会看到攻击者在对抗关键词过滤器时大约在 ~5–15 次迭代内成功，而对抗语义过滤器时失败。

## 交付产物

本课会生成 `outputs/skill-attack-audit.md`。给定一份红队评估报告，它会做审计：运行了哪些攻击（PAIR、GCG、TAP、AutoDAN、PAP），每种攻击的预算是多少，用了哪个裁判，针对的是哪组有害行为（JailbreakBench、HarmBench、内部集合）。

## 练习

1. 运行 `code/main.py`。测量内置三种攻击策略的平均成功查询次数（mean-queries-to-success）。解释每种策略利用了目标防御的哪个假设。
2. 实现第四种攻击者策略（例如：翻译成另一种语言、base64 编码）。报告在关键词过滤目标和语义过滤目标下的新平均成功查询次数。
3. 阅读 Chao et al. 2023 中的图 5（PAIR vs GCG 对比）。描述两个场景，在这些场景中尽管 PAIR 更高效，仍然更倾向于使用 GCG。
4. JailbreakBench 在固定目标集上报告 ASR。设计一个额外的度量来衡量攻击多样性（成功提示词的方差）。解释为何多样性对防御评估很重要。
5. TAP（Mehrotra 2024）通过分支 + 剪枝扩展了 PAIR。为 `code/main.py` 草拟一个 TAP 风格的扩展，并描述计算成本与成功率之间的权衡。

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| PAIR | "automated jailbreak" | Prompt Automatic Iterative Refinement；攻击者 LLM + 裁判 LLM 循环 |
| GCG | "gradient jailbreak" | 基于白盒的 token 级别梯度搜索以寻找对抗后缀 |
| Attack success rate (ASR) | "% jailbreaks at k queries" | 主要指标；必须连同查询预算和裁判身份一并报告 |
| Judge LLM | "the scorer" | 对响应是否满足有害目标进行评分的 LLM |
| JailbreakBench | "the evaluation" | 带有标签类别的标准化有害行为集合 |
| HarmBench | "the broader bench" | 包含 510 个行为的更广泛基准，含功能性与语义性有害测试 |
| TAP | "tree of attacks" | 带分支与剪枝的 PAIR；在更高算力下能获得更好 ASR |

## 延伸阅读

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR 论文，NeurIPS 2023  
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG 论文  
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — 标准化评估  
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — 更广泛的评估