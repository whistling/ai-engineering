# Agent 核心循环：观察、思考、行动 (Observe, Think, Act)

> 2026 年的每个 Agent — Claude Code, Cursor, Devin, Operator — 都是 2022 年 ReAct 循环的变体。在停止条件触发前，推理 Token 与工具调用、观察结果交替进行。在接触任何复杂框架之前，请务必彻底掌握这个循环。

---

## 一、 文档主要内容解读 (docs/en.md)

文档 [en.md](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/docs/en.md) 阐述了 Agent 系统的底层架构基石——**ReAct (Reason + Act，推理 + 行动)** 模式。

### 1. 核心概念与 ReAct 经典范式
* **LLM 本身的局限**：原生的大语言模型只是一个文本自动补全器。你给它输入，它返回字符串。它无法主动读取文件、调用 API、执行数据库查询或进行事实核查。
* **ReAct 的核心解决方案**：引入一个循环结构，允许模型中途暂停、调用外部工具、获取执行结果（Observation/观察），然后继续思考（Thought），直到满足停止条件。
* **标准流程轨迹 (Trace)**：
  ```
  Thought: 我需要查询法国的首都是什么。
  Action: search("capital of France")
  Observation: 巴黎是法国的首都。
  Thought: 答案是巴黎。
  Action: finish("Paris")
  ```

### 2. 2026 年的技术演进：原生推理通道 (Native Reasoning)
早期 ReAct 采用在 Prompt 中约束模型输出 `"Thought:"` 字符的临时方案（2022年做法）。在 2025-2026 年，新一代模型（如 Responses API 等）引入了**原生推理通道**：模型在独立的通道中生成思考过程，该过程在对话多轮迭代中流转（在生产服务商之间加密流转）。但**其底层的控制环路（Observe → Think → Act）并没有改变**。

### 3. 构建 Agent 循环的 5 个关键要素
缺一不可，否则就只是一个 Chatbot，而不是 Agent：
1. **动态增长的消息缓冲区 (Message Buffer)**：保存用户输入、助手思考、工具返回结果的上下文历史。
2. **工具注册表 (Tool Registry)**：维护工具名到具体可执行函数的映射，负责校验输入参数，并向模型吐出结果字符串。
3. **停止条件 (Stop Condition)**：模型输出 finish/最终答案，或者没有产生工具调用，或者达到最大轮数限制。
4. **单次任务的步数预算限制 (Turn Budget)**：安全阀门，设定最大执行步骤数（如 10-12 次），防止由于异常导致死循环消耗大量 Token。
5. **观察格式化器 (Observation Formatter)**：将工具执行返回的结果（包括异常报错）转换为模型可读的纯文本，保证流程能从错误中恢复而不崩溃。

### 4. 2026 年开发面临的陷阱与痛点
* **信任边界崩溃 (Trust boundary collapse)**：外部工具返回的数据属于不可信输入。例如网页抓取回来的 PDF 可能包含诱导删除仓库的指令。
* **级联失败 (Cascading failure)**：当其中一步工具返回错误时，Agent 难以区分“接口出错”和“任务不可能完成”，容易在报错上继续产生幻觉。
* **步骤长度爆炸 (Loop length explosion)**：现代 Agent 经常运行数十步到数百步。调试时非常依赖可观测性指标。

---

## 二、 代码仔细解读 (code/main.py)

代码文件 [main.py](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py) 是上述 5 大要素的极简 Python 标准库实现。

### 1. 核心数据结构

* [ToolCall](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py#L21-L23)：
  封装模型解析出的工具调用请求，包含工具名称 (`name`) 和参数字典 (`args`)。
* [Turn](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py#L27-L31)：
  表示对话历史中的每个轮次步骤，通过 `kind` 区分类型：用户输入 (`"user"`)、思考过程 (`"thought"`)、工具调用 (`"action"`) 还是最终结果 (`"final"`)。当 `kind` 为 `"action"` 时，会携带 [ToolCall](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py#L21-L23) 的实例以及工具的返回值 `observation`。

### 2. 工具注册表与外部工具定义

* [ToolRegistry](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py#L34-L54)：
  负责收集工具函数，并通过 `dispatch` 进行派发。使用 `try-except` 包裹执行过程，将所有错误捕获并以格式化字符串返回（例如参数不匹配的 `TypeError` 或其他异常），从而满足“**观察格式化器**”的要求。
* **系统自带工具**：
  * [calculator](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py#L56-L63)：支持字符白名单校验的算术计算器，防止恶意代码通过 `eval` 注入。
  * [KVStore](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py#L66-L76)：内存键值存储，模拟外部数据库，支持 `kv_get` 和 `kv_set`，允许 Agent 在多次迭代之间读写数据。

### 3. 确定性 Mock LLM 驱动

* [ToyLLM](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py#L78-L95)：
  为了保证离线测试的稳定和确定性，代码使用了一个基于剧本脚本（`script`）的伪 LLM。它根据预设的步骤，依次模拟 LLM 输出 thought、action 甚至最终的 finish。在实际生产部署中，这里可以无缝替换为调用 OpenAI/Anthropic 等厂商的原生推理 API。

### 4. ReAct 状态机引擎 (Agent Loop)

* [AgentLoop](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py#L98-L121) 驱动了整个执行流：
  * `run(user_message)`：
    1. 接收用户的提示词，追加为 `Turn`（消息缓冲区）。
    2. 进入设定上限 `max_turns` 的 `for` 循环（步数预算）。
    3. 获取 LLM 响应：
       * 如果响应类型是 `"finish"`，将其作为 `"final"` 加入历史，停止循环并返回最终答案。
       * 否则，记录思考过程 (`"thought"`)。
       * 提取工具名称与参数，通过 [ToolRegistry](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py#L34-L54) 的 `dispatch` 派发执行，并拿到 `observation`。
       * 封装为 `"action"` 类型记录存入历史，进入下一次循环。
    4. 若超过预算限制，则返回 `"budget exhausted"` 并终止。

### 5. 调试输出与组装运行

* [pretty_trace](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py#L124-L137)：用于打印结构化的运行轨迹。
* [build_demo_agent](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py#L139-L159)：将工具注册，定义了 6 步模拟剧本（设置基准价 -> 计算税额 -> 保存税额 -> 计算总价 -> 验证基准价 -> 最终输出）。
* [main](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/01-the-agent-loop/code/main.py#L162-L174)：运行整个 Demo 并输出控制台跟踪轨迹。

---

## 三、 运行轨迹追踪实例

以下为脚本执行的完整生命周期追踪：

```text
[00    user] What is 120 plus 15% tax, stored in kv?
[01 thought] store the base price
[02  action] kv_set({'key': 'base', 'value': '120'}) -> stored base
[03 thought] compute 15% tax
[04  action] calculator({'expr': '120 * 0.15'}) -> 18.0
[05 thought] store the tax
[06  action] kv_set({'key': 'tax', 'value': '18.0'}) -> stored tax
[07 thought] compute total
[08  action] calculator({'expr': '120 + 18.0'}) -> 138.0
[09 thought] confirm stored values
[10  action] kv_get({'key': 'base'}) -> 120
[11   final] the total including 15% tax is 138.0
```

1. **第 0 步 (User)**：用户输入。
2. **第 1-2 步 (Observe-Think-Act)**：LLM 思考需要先存入基准价。调用 `kv_set` 工具，得到 Observation 为 `stored base`。
3. **第 3-4 步**：LLM 思考需要计算 15% 的税金。调用 `calculator` 工具，计算结果 `18.0` 作为 Observation 返回。
4. **第 5-6 步**：LLM 思考需将税金记录到 KV 中。调用 `kv_set`，得到 Observation。
5. **第 7-8 步**：LLM 思考计算总额。调用 `calculator` 进行 `120 + 18.0` 的数学运算，得到 `138.0` 并返回。
6. **第 9-10 步**：LLM 思考在完成任务前需确认存入的值是否正确。调用 `kv_get(base)` 检索得到 `120`。
7. **第 11 步 (Final)**：LLM 终止动作，给出结论并输出。

---

## 四、 总结与进阶展望

这套底层的 Loop 是目前最复杂的各种 Agent 框架的核心。后续各种高级框架（如 Claude SDK 提供的子智能体派发、LangGraph 提供的状态机检查点等）本质上都是在此基础之上的封装：
* **LangGraph (Lesson 13)**：增加了状态的 Graph 表示和在每个节点步后的持久化状态检查点 (Checkpoints)。
* **AutoGen (Lesson 14)**：引入了基于 Actor 模型的异步消息通讯。
* **OpenAI/Claude Agents SDK (Lesson 16/17)**：增加了原生 Handoff（转接）、Guardrails（安全护栏）和会话上下文自动管理。
