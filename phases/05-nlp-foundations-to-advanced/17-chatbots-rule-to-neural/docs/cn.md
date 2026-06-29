# 聊天机器人 — 从基于规则到神经网络再到 LLM Agent

> ELIZA 用模式匹配回复。DialogFlow 将意图映射到流程。GPT 从权重中生成回答。Claude 运行工具并进行验证。每个时代都解决了前一代最明显的失败点。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 5 · 13 (问答), Phase 5 · 14 (信息检索)  
**Time:** ~75 分钟

## 问题

用户说 “I want to change my flight.”（我想改签航班）。系统必须推断出他们的意图、缺失的信息、如何获取这些信息，以及如何完成操作。随后用户又说 “wait, what if I cancel instead?”（等等，如果我改为取消呢？），系统必须记住上下文、切换任务并保持状态一致。

对 ML 系统来说，对话很难。输入是开放式的。输出需要在多轮对话中保持一致。系统可能需要在现实世界中执行操作（改签航班、收取费用）。每一步错误都会被用户看见。

聊天机器人架构经历了四种范式的循环演进，每一种都是为了修补前一种明显的失败。本课按顺序讲解它们。到 2026 年的生产环境通常是后两者的混合体。

## 概念

![聊天机器人演进：基于规则 → 检索 → 神经 → LLM Agent](../assets/chatbot.svg)

**基于规则（ELIZA、AIML、DialogFlow）。** 手工编写的模式匹配用户输入并生成回复。意图分类器将请求路由到预定义流程。插槽填充状态机收集所需信息。在设计的狭窄范围内表现极好。超出范围则立即失败。仍然在那些不能容忍幻觉的安全关键领域部署（银行身份验证、航空订票）。

**基于检索。** 类似 FAQ 的系统。对每个（话术，回复）对进行编码。运行时对用户消息进行编码并检索最相近的存储回复。想想 Zendesk 的“相似文章”功能。比规则更能处理意译。没有生成，因此不会幻觉。

**神经（seq2seq）。** 基于对话日志训练的编码器-解码器。从头生成回复。流畅但易产生泛化输出（“我不知道”）和事实漂移。很难始终保持主题。这是 2016–2019 年 Google、Facebook 和 Microsoft 等公司的聊天机器人表现不佳的原因。

**LLM agent。** 把语言模型包在一个循环里：规划、调用工具并验证结果。它不是把长提示塞进去的聊天机器人。一个 agent 循环：plan → call tool → observe result → decide next step。基于检索的落地（RAG）可以防止其幻觉。工具调用让它真正能做事。这是 2026 年的架构。

这四种范式不是按顺序被替代的。到 2026 年，生产聊天机器人会在这四者间路由：对认证和破坏性操作使用基于规则，常见问答使用检索，生成自然措辞使用神经生成，而对模糊开放式查询使用 LLM agent。

## 实现

### 第 1 步：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

用 20 行实现了 ELIZA。反射技巧（"I feel sad" → "Why do you feel sad"）是 Weizenbaum 1966 年的经典心理治疗演示，仍很有启发性。

### 第 2 步：基于检索（FAQ）

该示例片段需要 `pip install sentence-transformers`（会拉入 torch）。本课的可运行 `code/main.py` 使用标准库的 Jaccard 相似度替代，因此无需外部依赖也能运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键设计选择。如果最佳匹配不够接近，返回 `None`，让系统进一步升级处理。

### 第 3 步：神经生成（基线）

使用小型的指令微调编码器-解码器（FLAN-T5）或一个微调的会话模型。到 2026 年，单独使用并不适合生产（矛盾、脱离主题、虚构事实），但在混合系统中用于自然措辞仍然会部署。像 DialoGPT 那样的解码器模型需要显式的回合分隔符和 EOS 处理以生成连贯回复；FLAN-T5 的 text2text 管道对于教学示例则开箱即用。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 第 4 步：LLM agent 循环

2026 年的生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

需要命名的三件事。工具是 LLM 可以调用的可执行函数。循环在 LLM 返回最终答案而不是工具调用时终止。步骤预算用于防止在模糊任务上出现无限循环。

真实的生产环境还会加入：检索优先的落地（在每次 LLM 调用前注入相关文档）、护栏（在执行破坏性操作前拒绝或要求确认）、可观察性（记录每一步）、以及评估（自动化检查 agent 行为是否符合规范）。

### 第 5 步：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式是：对任何破坏性操作使用确定性规则，常见问答使用检索，其他情况使用 LLM agent。这就是 2026 年客户支持系统的典型做法。

## 使用场景

2026 年的技术栈：

| 用例 | 架构 |
|---------|---------------|
| 预订、支付、身份验证 | 基于规则的状态机 + 插槽填充 |
| 客户支持常见问答 | 基于策划答案的检索 |
| 开放式帮助聊天 | 带 RAG + 工具调用的 LLM agent |
| 内部工具 / IDE 助手 | 带工具调用的 LLM agent（搜索、读取、写入） |
| 伴侣 / 角色聊天机器人 | 带人物系统提示的微调 LLM，结合知识检索 |

在生产中始终使用混合路由。没有单一架构能很好地处理所有请求。路由层通常是一个小型的意图分类器。

## 仍然会出现在生产中的失效模式

- **自信的虚构。** LLM agent 声称已经完成某个操作但其实没有。缓解：验证结果、记录工具调用、不要让 LLM 在没有成功工具返回的情况下声称已执行操作。
- **提示注入（Prompt injection）。** 用户插入文本覆盖系统提示。2025 年 OWASP LLM 应用程序十大排行中 LLM01。两种形式：直接注入（粘贴到聊天里）和间接注入（隐藏在文档、邮件或 agent 读取的工具输出中）。

  攻击成功率随场景而异。一般工具使用和编码基准上，对前沿模型的测得成功率约为 ~0.5–8.5%。一些高风险设置（对抗性地攻击 AI 编码 agent、脆弱的编排）已达 ~84%。生产中的 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）——一个在 Microsoft 365 Copilot 中被触发的零点击数据外泄漏洞，源于攻击者控制的邮件。

  缓解措施：在整个循环中将用户输入视为不可信；在调用工具前进行清理；将工具输出与主提示隔离；使用 Plan-Verify-Execute (PVE) 模式，让 agent 先规划、然后在执行前验证每个动作（这能阻止工具结果注入新的未规划动作）；对破坏性操作要求用户确认；对工具权限采用最小权限原则。

  仅靠提示词工程无法完全消除该风险。需要外部运行时防御层（LLM Guard、白名单校验、语义异常检测）。
- **范围蔓延（Scope creep）。** Agent 因工具调用返回的旁支信息而偏离任务。缓解：缩窄工具契约；保持系统提示聚焦；增加离题率评估。
- **无限循环。** Agent 不断调用同一个工具。缓解：步骤预算、工具调用去重、让 LLM 判断“我们是否在取得进展”。
- **上下文窗口耗尽。** 长对话把最早的回合挤出上下文。缓解：对早期回合做摘要、按相似度检索相关过往回合，或使用长上下文模型。

## 发布（Ship It）

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. 简单：实现上文的基于规则的 respond，为咖啡店点单机器人编写 10 条模式。测试边界情况：重复订单、修改、取消、意图不明确等。
2. 中等：构建混合的 FAQ + LLM 后备。为一个 SaaS 产品准备 50 条常见问答条目，LLM 后备对网站文档进行检索落地。在 100 条真实支持问题上测量拒绝率和准确率。
3. 困难：实现以上的 agent_loop，提供三个工具（search、read-user-data、send-email）。用 50 个测试场景进行评估，其中包含提示注入尝试。报告离题率、任务失败率以及任何注入成功案例。

## 关键词

| 术语 | 人们如何说 | 实际含义 |
|------|-----------------|-----------------------|
| Intent | 用户想要什么 | 分类标签（book_flight、reset_password）。路由到处理器。 |
| Slot | 一条信息 | 机器人需要的参数（日期、目的地）。插槽填充是连续询问的过程。 |
| RAG | 检索加生成 | 检索相关文档，然后以这些文档为依据生成 LLM 的回答。 |
| Tool call | 函数调用 | LLM 发出带名称与参数的结构化调用。运行时执行并返回结果。 |
| Agent loop | 规划、执行、验证 | 控制器，交替运行 LLM 调用与工具调用，直到任务完成。 |
| Prompt injection | 提示注入 | 恶意输入试图覆盖系统提示。 |

## 拓展阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始的基于规则聊天机器人论文。  
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — Google 的晚期神经聊天机器人论文，LLM agent 盛行之前的一篇重要工作。  
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 将 agent 循环模式命名的论文。  
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年的生产级指南，在 2026 年仍然适用。  
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 讨论提示注入的论文。  
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 将提示注入列为首要安全关注点的排行。  
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 关于编排层防御的实用指南，包括 Plan-Verify-Execute 和用户确认流程。  
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 间接提示注入导致的零点击数据外泄 CVE 的典型案例。说明了为何具有写权限的 agent 需要运行时防御。