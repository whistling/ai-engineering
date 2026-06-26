# FIPA-ACL 与言语行为的传承

> 在 MCP、在 A2A 之前，有 FIPA-ACL。2000 年，IEEE 的 Foundation for Intelligent Physical Agents 通过了一种智能体通信语言，定义了二十个 performative、两种内容语言，以及一组交互协议——合同网、订阅/通知、请求-条件（request-when）。由于本体开销对 Web 来说太重，这套规范在工业界逐渐淡出，但 LLM 驱动的多代理系统正在悄然以更松散的形式重塑这些思想：JSON 合同替代了 performative，自然语言替代了本体。本课认真解读 FIPA-ACL，帮助你看清 2026 年的协议决策哪些是重复发明，哪些是真正的新意，以及当前浪潮将重新遇到哪些 2000 年代已解决的问题。

**Type:** 学习  
**Languages:** Python（标准库）  
**Prerequisites:** Phase 16 · 01（为何多代理）  
**Time:** ~60 分钟

## 问题

2026 年的代理协议生态非常热闹：用于工具的 MCP、用于代理间通信的 A2A、用于企业审计的 ACP、用于去中心化信任的 ANP、用于自然语言内容的 NLIP，外加 CA-MCP 和数十项研究提案。每个规范都自称是基础性的。

诚实地说，它们中的大多数在重新发现一个非常具体的二十年前的决策树。Austin（1962）与 Searle（1969）的言语行为理论给了我们“话语就是行动”的观点。KQML（1993）将其变为一个线协议。FIPA-ACL（2000 年通过）给出了参考标准化：二十个 performative、内容语言 SL0/SL1、用于合同网和订阅-通知的交互协议。JADE 和 JACK 是 Java 的参考平台。随着本体论开销过重而 Web 占据主导，这一努力在约 2010 年左右衰退。

当你看到 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储时，你看到的是 FIPA 决策的更宽松、原生 JSON 的重写。了解其传承会告诉你两件事：哪些所谓的新“创新”其实是重复发明，哪些老的失败模式会被新规范重新遇到。

## 概念

### 用一段话说言语行为

Austin 注意到有些句子不是在描述世界——它们改变世界。“我承诺。”“我请求。”“我宣布。”他称这些为施为性话语（performative utterances）。Searle 将其形式化为五类：断言性（assertive）、指令性（directive）、承诺性（commissive）、表达性（expressive）、宣告性（declarative）。KQML（Finin 等，1993）把这套理论化为可运行的软件代理模型：一条消息由一个 performative（动作）加上内容（动作的对象）组成。FIPA-ACL 弥补了 KQML 的不足，并把二十个 performative 标准化了。

### 二十个 FIPA performative（部分列举）

| Performative | 意图 |
|---|---|
| `inform` | “我告诉你 P 为真” |
| `request` | “我请求你做 X” |
| `query-if` | “P 是真吗？” |
| `query-ref` | “X 的值是什么？” |
| `propose` | “我建议我们做 X” |
| `accept-proposal` | “我接受该提议” |
| `reject-proposal` | “我拒绝该提议” |
| `agree` | “我同意去做 X” |
| `refuse` | “我拒绝去做 X” |
| `confirm` | “我确认 P 为真” |
| `disconfirm` | “我否认 P” |
| `not-understood` | “你的消息无法解析” |
| `cfp` | “就 X 发布征求提案（Call for Proposals）” |
| `subscribe` | “当 X 变化时通知我” |
| `cancel` | “取消当前的 X” |
| `failure` | “我尝试了 X 但失败了” |

完整列表见 `fipa00037.pdf`（FIPA ACL Message Structure）。关键不是要记住它——关键是这些中的每一个最终都会对应到某个 LLM 协议里被重新加入的原语。

### 规范的 FIPA-ACL 消息示例

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载协议信封；一个字段（`content`）承载有效负载。其余字段正是你每次把重试、线程化和本体绑进 JSON 协议时都会重新发明的那些内容。

### 两个遗留平台

**JADE**（Java Agent DEvelopment framework，1999–2020s）是最常用的 FIPA 兼容运行时。代理继承自基类，交换 ACL 消息，在容器内运行，并通过“行为（behaviors）”进行协调。交互协议库自带合同网、订阅-通知、请求-条件和提议-接受等模式。

**JACK**（Agent Oriented Software，商业产品）在 FIPA 消息之上强调 BDI（Belief-Desire-Intention）推理。更正式，但采纳度较低。

随着 Web 技术栈占据多代理的使用场景，这两者都走向衰退。MCP 和 A2A 是 2026 年的运行时“容器”。

### FIPA 褪色的原因

- 本体开销。FIPA 要求共享本体来解析 `content`。达成本体一致是一个多年期的标准化过程。Web 更喜欢 HTTP + JSON 的方式。
- 很少有人使用的形式语义。SL（Semantic Language）给出了严格的真值条件，但大多数生产系统使用自由格式的内容并忽略了形式化。
- 工具锁定。JADE 仅限 Java；JACK 是商业产品。多语言团队都会绕开它们。
- 互联网赢了整个栈。REST、随后是 JSON-RPC、再到 gRPC 替代了 ACL 的传输层。

### LLM 复兴是 FIPA-lite

把 FIPA 的 `request` 与 MCP 的 `tools/call` 对比：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

同样的信封，不同的语法。两者都包含：谁、给谁、意图、有效负载、相关性 id。它们之间并没有革命性的差别——只是对相同设计的不同权衡。

2025 年 Liu 等人的综述（“A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP”，arXiv:2505.02279）明确指出了这一路径：MCP 对应工具使用类的言语行为，A2A 对应代理同行的言语行为，ACP 对应审计轨迹的言语行为，ANP 对应去中心化身份的扩展。新规范是带有 JSON 语法且语义更松散的 ACL 后代。

### 明说一下权衡

什么是 FIPA 给你的但现代规范放弃了：

- 形式语义——你可以证明 `inform` 暗示发送者相信其内容为真。
- 一个规范化的 performative 目录——你不必重新争论“我们是否需要 `cancel`？”。
- 数十年的交互协议模式——合同网、订阅-通知、提议-接受——带有已知的正确性性质。

现代规范给你的而 FIPA 没有的：

- 与现代工具兼容的原生 JSON 有效载荷。
- LLM 可以直接解释的自然语言内容，而无需手工编码的本体。
- Web 栈的传输（HTTP、SSE、WebSocket）。
- 通过自描述文档进行的能力发现（MCP `listTools`、A2A Agent Card）。

更松的意图语义换来更容易实现的系统。这正是两者之间的权衡。

### 值得迁移的交互协议

FIPA 发布了约 15 个交互协议。三类值得在 LLM 多代理系统中继续沿用：

1. 合同网协议（Contract Net Protocol，CNP）。管理者发布 `cfp`（征求提案）；投标者以 `propose` 响应；管理者进行 accept/reject。它是典型的任务市场模式（Phase 16 · 16 谈判）。
2. 订阅/通知（Subscribe/Notify）。订阅者发送 `subscribe`；发布者在主题变化时发送 `inform`。这是 2026 年几乎所有事件总线的实现方式。
3. 请求-条件（Request-When）。“当条件 Y 成立时执行 X。”带前置条件的延迟动作。2026 年的类比是持久化工作流引擎中的延后任务（Phase 16 · 22 生产扩展）。

每一种都能清晰地映射到现代的消息队列、HTTP + 轮询或 SSE 流式传输。

### 当你放弃本体会出现什么问题

没有共享本体，代理将从自然语言内容中推断含义。2026 年记录的失败模式是语义漂移（semantic drift）：两个代理使用相同的词（比如 `"customer"`）指代微妙不同的概念，接收方代理基于错误的解释采取了行动，而没有模式校验器能捕获这个错误。FIPA 的本体要求会在解析时就拒绝该消息。

在不完全回归到本体化的情况下的缓解措施：

- 在 `content` 上使用 JSON Schema——在通信层拒绝结构错误。
- 使用类型化工件（A2A）——拒绝错误的模态（modality）。
- 在信封中显式 performative——即使内容是自然语言也能使意图明确。

### 2026 年规范与言语行为的对应关系

| Modern spec | FIPA analog | 它保留了什么 | 它放弃了什么 |
|---|---|---|---|
| MCP `tools/call` | `request` | 显式意图、相关性 id | 形式语义、本体 |
| MCP `resources/read` | `query-ref` | 显式意图、相关性 id | 形式语义 |
| A2A Task lifecycle | contract-net + request-when | 异步生命周期、状态转换 | 形式完整性保证 |
| A2A streaming events | subscribe/notify | 异步推送 | 类型谓词订阅 |
| CA-MCP shared context | 黑板（Hayes-Roth 1985） | 多写者共享内存 | 逻辑一致性模型 |
| NLIP | 自然语言内容 | LLM 原生 | 模式（schema） |

从表格自上而下读，模式是：保留结构性原语，放弃形式化，让 LLM 用模糊性来掩盖差异。

## 构建它

`code/main.py` 实现了一个仅用标准库的 FIPA-ACL 转换器。它对规范的 ACL 信封进行编码和解码，并展示了每个 MCP / A2A 消息形状如何归约到相同的七个字段。演示包括：

- 将五个 MCP 风格和 A2A 风格的消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等价形式。
- 使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 在一个管理者和三个投标者之间运行一个玩具合同网谈判。

运行：

```
python3 code/main.py
```

输出是并列的跟踪，显示每条现代消息在 2026 年 JSON 形式与其 FIPA-ACL 形式下的对比，然后进行一次合同网竞标的往返。相同的协议原语在往返中幸存；只有语法不同。

## 使用它

`outputs/skill-fipa-mapper.md` 是一个技能（skill），它读取任何代理协议规范并生成 FIPA-ACL 的映射。在采用新协议之前使用它来回答：“这是真正的新东西，还是带 JSON 语法的 `inform`？”

## 发布它

不要把 FIPA-ACL 原封不动地带回来。带回它的清单（checklist）即可：

- 每条消息的意图原语（performative）是什么？
- 请求—响应和取消是否有相关性 id（correlation id）？
- 是否存在显式的内容语言（JSON-RPC、纯文本、结构化的类型化工件）？
- 交互协议是否是一级公民，还是你在从头实现合同网？
- 当两个代理对内容含义发生分歧（语义漂移）时会发生什么？

在把任何新协议投入生产之前，为其记录这五个问题。

## 练习

1. 运行 `code/main.py`。观察往返编码。识别哪些 FIPA performative 对应 `tools/call`、`resources/read` 和 A2A 任务创建。
2. 在合同网演示中加入一个 `cancel` performative，使管理者能够在竞标进行中撤回任务。`cancel` 解决了哪种仅靠重试无法解决的失败情形？
3. 阅读 FIPA ACL Message Structure (http://www.fipa.org/specs/fipa00037/) 第 4.1–4.3 节。选一个本课未覆盖的 performative，并描述其现代 JSON-RPC 对应物。
4. 阅读 Liu 等人，arXiv:2505.02279。对于 MCP、A2A、ACP、ANP，列出它们保留和放弃的 FIPA performative 家族。
5. 为你自己的系统中 `request` performative 的 `content` 字段设计一个最小的 JSON-Schema。与纯自然语言相比，该 schema 给了你什么，代价又是什么？

## 关键词

| 术语 | 人们如何说 | 实际含义 |
|------|----------------|------------------------|
| 言语行为（Speech act） | “一个执行某事的话语” | Austin/Searle：把话语看作行动。ACL 的理论根源。 |
| FIPA | “那个老掉牙的 XML 东西” | IEEE Foundation for Intelligent Physical Agents。2000 年标准化了 ACL。 |
| ACL | “Agent Communication Language” | FIPA 的信封格式：performative + content + 元数据。 |
| Performative | “动词” | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | “FIPA 的前身” | Knowledge Query and Manipulation Language（1993）。更简单、更窄。 |
| 本体（Ontology） | “共享词汇” | 内容语言所谈论概念的形式化定义。 |
| SL0 / SL1 | “FIPA 的内容语言” | Semantic Language 的 0 与 1 级——形式化内容语言家族。 |
| 合同网（Contract Net） | “任务市场” | 管理者发出 cfp；投标者 propose；管理者接受。典型的交互协议。 |
| 交互协议（Interaction protocol） | “消息模式” | 一系列具有已知正确性性质的 performative：request-when、subscribe-notify 等。 |

## 延伸阅读

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 将现代规范与 FIPA 传承联系起来的 2025 年权威综述  
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000 年通过的信封格式规范  
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 完整的 performative 目录  
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 现代的工具使用等价物（`request`/`query-ref`）  
- [A2A specification](https://a2a-protocol.org/latest/specification/) — 现代的代理同行等价物（合同网与订阅-通知）