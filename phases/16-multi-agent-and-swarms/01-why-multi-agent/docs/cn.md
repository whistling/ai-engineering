# 为什么要使用多智能体？

> 单个 agent 会遇到瓶颈。聪明的做法不是做一个更大的 agent —— 而是使用更多的 agent。

**Type:** 学习  
**Languages:** TypeScript  
**Prerequisites:** Phase 14 (Agent Engineering)  
**Time:** ~60 分钟

## 学习目标

- 识别单智能体天花板（上下文溢出、混合专业、顺序瓶颈），并说明何时把任务拆分为多个智能体是正确的选择
- 比较编排模式（流水线、并行扇出、监督者、分层）并为给定的任务结构选择合适的模式
- 设计具有清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多智能体复杂性（延迟、成本、调试难度）与单智能体简洁性之间的权衡

## 问题

你在 Phase 14 构建了一个单智能体。它能工作。它可以读取文件、运行命令、调用 API，并对结果进行推理。然后你把它指向一个真实代码库：200 个文件、三种语言、依赖基础设施的测试，并且在编写代码前需要研究外部 API。

智能体崩溃了。不是因为 LLM 愚蠢，而是因为任务超出单个智能体循环能处理的范围。上下文窗口被文件内容填满。智能体忘记了 40 次工具调用之前读过的东西。它试图同时充当研究员、编码者和审核者，结果三样都不够好。

这就是单智能体天花板。每当任务需要以下任一项时你都会遇到它：

- **比单个窗口能容纳的更多上下文** - 读取 50 个文件会超过 200k token
- **不同阶段需要不同专业知识** - 研究需要不同的提示策略，和代码生成不同
- **可以并行进行的工作** - 为什么要顺序读取三个文件而不是同时读取它们？

## 概念

### 单智能体天花板

单智能体就是一个循环、一个上下文窗口、一个系统提示。想象它：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

三个问题会出现：

1. **上下文饱和** - 工具结果堆积。到第 30 回合时，智能体已经消耗了 150k tokens 的文件内容、命令输出和先前的推理。第 5 回合的重要细节丢失了。

2. **角色混淆** - 一个同时说“你是研究员、编码者、审核者和测试者”的系统提示，会产生一个半研究、半写码、从未完成审核的智能体。

3. **顺序瓶颈** - 智能体先读文件 A，然后读文件 B，再读文件 C。三个串行的 LLM 调用。三个串行的工具执行。没有并行性。

### 多智能体解决方案

把工作分拆。给每个智能体一个任务、一个上下文窗口和一个为该任务调优的系统提示：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个智能体拥有：
- 一个专注的系统提示（“你是代码审核者。你的唯一任务是发现 bug。”）
- 它自己的上下文窗口（不会被其他智能体的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 已有的真实系统示例

**Claude Code 子智能体** - 当 Claude Code 使用 `Task` 派生子智能体时，会创建一个带有作用域任务的子 agent。父智能体保持上下文清洁。子智能体做专注工作并返回摘要。

**Devin** - 运行一个规划者 agent、一个编码者 agent 和一个浏览器 agent。规划者将工作拆分为步骤。编码者编写代码。浏览器进行文档研究。每个都有独立的上下文。

**多智能体编码团队（SWE-bench）** - 在 SWE-bench 上表现最好的系统使用一个读取代码库的研究者、一个设计修复的规划者和一个实现修复的编码者。单智能体系统得分更低。

**ChatGPT Deep Research** - 并行派生多个搜索 agent，每个从不同角度探索，然后综合结果。

### 光谱

多智能体不是非黑即白，而是一个光谱：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       roles
```

- **Single agent** - 一个循环，一个提示。适用于简单任务。
- **Subagents** - 父 agent 派生子任务给子 agent。父 agent 保持计划。子 agent 汇报结果。这就是 Claude Code 的做法。
- **Pipeline（流水线）** - agents 顺序运行。A 的输出成为 B 的输入。适合分阶段的工作流：研究 -> 编码 -> 审核 -> 测试。
- **Team（团队）** - agents 并行运行并共享消息总线。每个 agent 有明确角色。一个编排者负责协调。适合同时需要不同技能的场景。
- **Swarm（群）** - 许多相同或近似相同的 agent 共享状态。没有固定的编排者。agent 从队列中领取任务。适合高吞吐量并行任务。

### 四种多智能体模式

#### 模式 1：流水线（Pipeline）

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个 agent 转换数据并传递给下一个。易于推理。某一阶段失败会阻塞后续阶段。

#### 模式 2：扇出/扇入（Fan-out / Fan-in）

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

将工作拆分给并行的 agents，然后合并结果。适合可以分解为独立子任务的场景。

#### 模式 3：编排者-工作者（Orchestrator-Worker）

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

一个聪明的编排者决定要做什么，委派给工作者，并合成结果。编排者本身也是一个具有派生工作者工具的 agent。

#### 模式 4：对等群（Peer Swarm）

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有中央编排者。agents 对等通信。决策从交互中涌现。更难调试，但能扩展到大量 agents。

### 何时不要使用多智能体

多智能体会增加复杂性。每条 agent 间的消息都是潜在的故障点。调试从“读一个对话”变成“追踪五个 agent 之间的消息”。

保持单智能体的场景：
- 任务能适配一个上下文窗口（在大约 ~100k tokens 以内的工作数据）
- 不需要在不同阶段使用不同的系统提示
- 顺序执行足够快速
- 任务足够简单，拆分只会带来额外开销而无增益

复杂性成本：
- 每个 agent 边界都是一次有损压缩：agent A 的完整上下文被摘要成发送给 agent B 的消息
- 协调逻辑（谁做什么、何时做、按什么顺序）本身就是错误来源
- 延迟增加：N 个 agent 意味着至少 N 次串行 LLM 调用，若需要来回通信则更多
- 成本成倍增长：每个 agent 都要独立消耗 tokens

经验法则：如果一个任务在 20 次工具调用以内并且能适配 100k tokens，保持单智能体。

```figure
swarm-messages
```

## 构建示例

### 第 1 步：过载的单智能体

下面是一个试图做所有事情的单智能体。它有一个巨大的系统提示和一个包含研究、代码和审核的大上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

该方法的问题：
- 上下文窗口会随着每个阶段增长。到审核步骤时，它包含研究笔记、代码和先前的推理。
- 系统提示过于通用，无法为每个阶段单独调优。
- 没有并行执行。

### 第 2 步：专职智能体

现在把任务拆分。每个智能体只做一件事：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专职智能体都有一个聚焦的提示。每个智能体只获得它需要的输入，保持干净的上下文窗口。

### 第 3 步：通过消息进行协调

用显式的消息传递把专职智能体连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发给它的消息。没有上下文污染。研究者读取的 50k tokens 文档永远不会进入审核者的上下文中。

### 第 4 步：比较

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多智能体版本总体上使用更多的 tokens（三个智能体、三次独立的 LLM 调用），但每个智能体的上下文保持干净。由于系统提示被专门化，各阶段的质量都有提升。

## 使用方法

本课产出一个可复用的决策提示，用于决定何时采用多智能体。见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 增加第四个专职智能体：一个 “tester” 智能体，它接收来自 coder 的代码和来自 reviewer 的审核反馈，然后编写测试
2. 修改流水线，使 reviewer 能向 coder 发送反馈以进行修订循环（最多 2 轮）
3. 将顺序流水线转换为扇出：并行运行 researcher 和一个 “requirements analyzer” 智能体，然后在传给 coder 之前合并它们的输出

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Swarm | "A hive mind of AI agents" | 一组对等 agent，具有共享状态且没有固定领导者。从局部交互中涌现系统行为。 |
| Orchestrator | "The boss agent" | 一个其工具包括派生与管理其他 agent 的 agent。它负责规划和委派，但不一定亲自完成实际工作。 |
| Coordinator | "The traffic cop" | 一个非 agent 的组件（通常只是代码，而非 LLM），根据规则在 agent 之间路由消息。 |
| Consensus | "The agents agree" | 一个协议，要求多个 agent 在继续之前达成一致。用于需要解决冲突输出的场景。 |
| Emergent behavior | "The agents figured it out themselves" | 从 agent 交互中产生的系统级模式，虽然不是显式编程得到的。可能有用也可能有害。 |
| Fan-out / fan-in | "Map-reduce for agents" | 将任务拆分给并行的 agents（扇出），然后合并它们的结果（扇入）。 |
| Message passing | "Agents talk to each other" | agent 之间的通信机制：将结构化数据从一个 agent 发送到另一个 agent，替代共享上下文窗口。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 多智能体模式综述  
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - 微软的多智能体对话框架  
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何使用 Task 进行委派  
- [CrewAI documentation](https://docs.crewai.com/) - 基于角色的多智能体框架文档