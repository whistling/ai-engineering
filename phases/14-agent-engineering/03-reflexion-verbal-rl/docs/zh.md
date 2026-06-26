# Reflexion：基于语言的强化学习 (Verbal Reinforcement Learning)

> **核心思想：** 传统的基于梯度的强化学习（Gradient-based RL）需要成千上万次尝试和庞大的 GPU 集群才能修正一个失败的运行模式。Reflexion（Shinn et al., NeurIPS 2023）通过**自然语言**来解决这个问题：在每一次尝试失败后，Agent 会写下一段反思（Reflection），将其存储在情境记忆（Episodic Memory）中，并在下一次尝试时将此记忆作为 Prompt 输入，指导下一次运行。这种模式是 Letta 异步睡眠反思机制、Claude Code 的 `CLAUDE.md` 学习沉淀以及工作流 `/learn-rule` 背后共同的设计原理。

---

## 一、文档核心内容解读 (`docs/en.md`)

在 [docs/en.md](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/03-reflexion-verbal-rl/docs/en.md) 中，详细对比了 Reflexion 与传统强化学习的理念，并指出了其三阶段设计及普适性。

### 1. 核心痛点
当 Agent 任务失败时，传统的 RL 需要通过大量的样本迭代来计算梯度并更新权重。这在工程部署中极其昂贵且缓慢，且绝大多数生产环境下的 Agent 无法在运行时承担微调模型的开销。

Reflexion（arXiv:2303.11366）提出了一种新思路：如果 Agent 能够自己想明白为什么失败，并在下一次运行的 Prompt 中带上这句反思会怎样？**不更新权重，不计算梯度，仅通过在多轮尝试之间传递自然语言反思来实现自我纠偏**。该设计在 ALFWorld、HotpotQA 和 HumanEval（代码生成）上均取得了当时最先进的效果。

### 2. Reflexion 的三大组件与数据结构
* **Actor（执行器）**：负责生成交互轨迹（通常是一个 ReAct 风格的循环）。
* **Evaluator（评估器）**：对 Actor 的结果进行打分（可以是二分类、启发式或自我评估）。
* **Self-Reflector（自我反思器）**：分析失败的轨迹，并写下一段自然语言反思（例如：“我选错了工具，因为我误把关于 X 的提问当成了 Y”）。
* **Episodic Memory（情节记忆）**：存储此前各轮失败反思的列表，并在下一次尝试时拼接到 Actor 的输入 Prompt 中。

### 3. 三种 Evaluator 类型

| 类型 | 机制 | 优缺点与适用场景 |
| :--- | :--- | :--- |
| **Scalar (标量/数值)** | 外部客观的二进制或数值信号。 | 例如：单元测试是否通过（Pass/Fail）。信号最明确，效果最好。 |
| **Heuristic (启发式)** | 预设的代码逻辑检测。 | 例如：“如果连续产生两次相同动作，判定为陷入死循环”。作为系统安全网使用。 |
| **Self-evaluated (自评估)** | 调用 LLM 对自己的轨迹打分。 | 适用于没有标准答案的开放场景。信号稍弱，通常配合 CRITIC（工具核验）使用。 |

生产环境通常混合使用：首选客观 Scalar 信号，无客观答案时退回到 Self-eval，同时使用 Heuristics 作为边界安全护栏。

### 4. 行业应用演化
Reflexion 已经超越了一个单一算法，成为了事实上的通用开发模式。例如：
* **Letta**：在空闲时间（Sleep-time）启动专门的反思 Agent，整理过去的对话并将其沉淀进记忆模块。
* **Claude Code**：每次失败后总结教训并持久化到本地的 `CLAUDE.md` 文件中，供后续开发会话载入。
* **开发辅助**：通过手动或自动的 `/learn-rule` 来固化特定的修复规则。
* **LangGraph**：构造专门的反思节点（Reflection Node），如果评分不合格则重定向回原节点修复。

### 5. 局限性与“记忆衰退”
Reflexion 并不适用于：首尝试即成功、外部环境故障（如网络断开）或随机偶发错误的场景。此外，多轮反思堆叠会导致 **Memory Rot（情节记忆膨胀劣化）**，使后续的推理速度变慢。解决方案通常是定期对其进行压缩提炼（Compaction）、设置生存时间（TTL）或使用后台 Agent 异步清理。

---

## 二、代码详细解读 (`code/main.py`)

[code/main.py](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/03-reflexion-verbal-rl/code/main.py) 用纯 Python 标准库模拟了一个基于 Reflexion 解决数字拼图的例子：选择 3 个范围在 `[1..9]` 的整数，使其总和恰好等于目标数字（TARGET = 20）。

下面是主要代码模块的作用剖析：

### 1. 情节记忆与反思存储
```python
@dataclass
class Reflection:
    trial: int  # 尝试的轮次编号
    text: str   # 具体的自然语言反思内容

@dataclass
class EpisodicMemory:
    items: list[Reflection] = field(default_factory=list)
    max_len: int = 6  # 记忆最大长度上限

    def add(self, r: Reflection) -> None:
        self.items.append(r)
        if len(self.items) > self.max_len:
            self.items.pop(0) # 采用 FIFO 机制进行队列滚动剔除，防缓冲区爆炸

    def as_prompt(self) -> str:
        if not self.items:
            return "(no prior reflections)"
        lines = [f"- trial {r.trial}: {r.text}" for r in self.items]
        return "\n".join(lines)
```
* [EpisodicMemory](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/03-reflexion-verbal-rl/code/main.py#L22-L35) 模拟了 Agent 的外部短效记忆。它限制了最大缓冲区长度（`max_len = 6`），并提供了 `as_prompt` 拼装函数，在多轮运行间传递经验。

### 2. 带有脚本反思逻辑的 Actor
```python
class Actor:
    def act(self, memory: EpisodicMemory) -> list[int]:
        n = len(memory.items)
        if n == 0:
            return [1, 2, 3] # 初始尝试：盲目返回 [1, 2, 3] (和为 6)
        if n == 1:
            return [5, 6, 7] # 看到第 1 个反思后，向着大数值修正为 [5, 6, 7] (和为 18)
        if n == 2:
            return [6, 7, 7] # 看到第 2 个反思后，精准修正到 [6, 7, 7] (和为 20)
        return [6, 7, 7]
```
* [Actor](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/03-reflexion-verbal-rl/code/main.py#L38-L50) 内部是一个脚本化的决策函数。如果不传递有效的 `memory`，它每一次的尝试都会从盲目的起点 `[1, 2, 3]` 开始（即 baseline 组）；若传递了 `memory`，它会根据已有的反思轮数做出适应性纠偏。

### 3. 评估器 Evaluator 与反思生成器 SelfReflector
```python
def binary_evaluator(attempt: list[int], target: int) -> tuple[bool, int]:
    total = sum(attempt)
    return total == target, total - target # 返回是否成功，以及偏离目标的差值 (delta)

class SelfReflector:
    def reflect(self, attempt: list[int], delta: int) -> str:
        if delta < 0:
            return f"sum {sum(attempt)} is {-delta} short; pick larger values"
        if delta > 0:
            return f"sum {sum(attempt)} overshoots by {delta}; pick smaller values"
        return "succeeded"
```
* [binary_evaluator](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/03-reflexion-verbal-rl/code/main.py#L53-L55) 扮演 Evaluator，计算当前解是否符合条件，并给出客观数值偏移（Delta）。
* [SelfReflector](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/03-reflexion-verbal-rl/code/main.py#L58-L64) 根据偏离的 Delta 产生指导性质的自然语言反思文本（例如 “还差 14，请尝试更大数值”）。

### 4. 反思运行与基线对比
```python
def run_reflexion(max_trials: int, use_memory: bool) -> list[TrialResult]:
    actor = Actor()
    reflector = SelfReflector()
    memory = EpisodicMemory()
    trials: list[TrialResult] = []
    for t in range(1, max_trials + 1):
        # 如果不启用记忆，每次执行都传入一个空的 EpisodicMemory()
        attempt = actor.act(memory if use_memory else EpisodicMemory())
        success, delta = binary_evaluator(attempt, TARGET)
        text = reflector.reflect(attempt, delta)
        trials.append(TrialResult(t, attempt, success, delta, text))
        if success:
            break
        # 失败则将反思写入 EpisodicMemory
        memory.add(Reflection(trial=t, text=text))
    return trials
```
* [run_reflexion](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/03-reflexion-verbal-rl/code/main.py#L76-L89) 是调度控制环路。如果 `use_memory=False`，Actor 每次看到的都是空的记忆体，永远无法自我进化（陷入 Baseline 局）；如果 `use_memory=True`，Actor 能看见以往各轮失败后生成的 Reflection，从而纠偏并在第 3 轮成功收敛：

```text
BASELINE (无情节记忆 - use_memory=False)：
  尝试 1: [1, 2, 3] sum=6 delta=-14 ... 
  尝试 2: [1, 2, 3] sum=6 delta=-14 ... 
  尝试 3: [1, 2, 3] sum=6 delta=-14 ... 
  最终：第四轮后超时失败。

REFLEXION (启用情节记忆 - use_memory=True)：
  尝试 1: [1, 2, 3] sum=6 delta=-14 ... -> 反思：sum 6 还差 14，选更大的数
  尝试 2: [5, 6, 7] sum=18 delta=-2 ... -> 反思：sum 18 还差 2，选更大的数
  尝试 3: [6, 7, 7] sum=20 delta=+0 OK  -> 成功解决！
```

---

## 三、总结

Reflexion 验证了**自然语言是比底层权重参数更加通用、且便于操作的学习媒介**。在实际工程项目中，如果您开发的项目经常出现步骤超限、调用工具逻辑偏离或接口微小报错，可以通过将上一轮失败的提示词与错误观测总结拼接到下一次运行的系统提示词中，让大模型自适应纠偏，从而避免频繁重新训练。
