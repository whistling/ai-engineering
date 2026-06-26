# ReWOO 和 Plan-and-Execute：解耦规划与执行

> **核心思想：** ReAct 在单个流中交替进行思考（Thought）和行动（Action）。而 ReWOO 将它们分离开来：先在最前面生成一个完整的规划，然后逐步执行。这种做法减少了约 5 倍的 Token 消耗，在 HotpotQA 数据集上提高了 4% 的准确率，并且支持将规划器（Planner）蒸馏为 7B 大小的轻量模型。Plan-and-Execute 进一步泛化了这一模式，而 Plan-and-Act 则将其扩展到了长周期的网页导航任务。

---

## 一、文档核心内容解读 (`docs/en.md`)

在 [docs/en.md](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/02-rewoo-plan-and-execute/docs/en.md) 中，详细对比了 **ReAct**（交替推理与执行）与 **ReWOO / Plan-and-Execute**（解耦规划与执行）两种 Agent 架构，指出了 ReWOO 的核心设计和优势。

### 1. ReAct 的痛点
ReAct 采用“思考-行动-观察（Thought-Action-Observation）”的交替循环。每次调用工具时，大模型都需要携带**先前所有的上下文**（包括之前的 Thoughts、Actions 和 Observations）。这会导致：
* **Token 消耗呈二次方增长（Quadratic Growth）**：步骤越深，重复携带的历史信息越多。
* **容错性低**：如果中间某个工具调用出错，模型需要在当前的错误上下文中重新推理整个计划，容易走偏或卡死。

### 2. ReWOO 的三阶段解耦设计
ReWOO（Xu et al., May 2023）提出“一次性规划，按需执行，最后求解”的解耦方案：
* **Planner（规划器）**：接收用户问题，输出一个步骤有向无环图（Plan DAG）。步骤中可以使用占位符（如 `#E1`, `#E2`，表示第 1、2 步的输出）。**Planner 规划时不需要看真实的工具返回结果**。
* **Workers（执行器）**：按照依赖关系（拓扑排序）执行工具调用，替换占位符，收集证据（Evidence）。由于没有推理历史，每个 Worker 调用都非常轻量。
* **Solver（求解器）**：将原始问题、Plan DAG 和 Workers 收集的所有 Evidence 输入给大模型，生成最终答案。

### 3. ReWOO 的优势
* **Token 节省 ~5倍**：省去了中间步骤不断叠加历史 Observation 的 Token 消耗。
* **更强的鲁棒性**：工具报错时只会作为局部 Evidence 返回给 Solver，Solver 可以基于全局计划做优雅降级，而不是在中间步骤里“越陷越深”。
* **Planner 蒸馏**：因为 Planner 规划时不需要真实的 Observation，所以可以用一个 175B 的大模型生成规划轨迹，去微调（蒸馏）一个 7B 的小模型专门做 Planner。在生产环境中，这能实现“小模型规划 + 外部执行 + 大模型求解/总结”的高效架构。

### 4. 衍生模式对比

| 模式 (Pattern) | 适用场景 |
| :--- | :--- |
| **ReAct** | 任务较短、环境未知、需要灵活的即时错误处理。 |
| **ReWOO** | 结构化任务、工具集已知、对 Token 敏感、证据获取可并行化。 |
| **Plan-and-Execute** | 类似 ReWOO，但执行一部分步骤后可以有一个 **Replanner（重新规划器）** 根据最新结果修正计划。 |
| **Plan-and-Act** | 针对超长链路（>30步）的网页/移动端/电脑操作 Agent，通过合成数据训练 Planner 以防轨迹失焦。 |

---

## 二、代码详细解读 (`code/main.py`)

[code/main.py](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/02-rewoo-plan-and-execute/code/main.py) 用 Python 标准库模拟了一个 Toy 版本的 ReWOO。它解决了这样一个问题：“What is the population of the capital of France, rounded to millions? (法国首都的人口是多少，四舍五入到百万？)”

下面我们顺着代码的模块进行拆解：

### 1. 数据结构定义
```python
@dataclass
class PlanStep:
    id: str             # 步骤 ID，例如 "E1", "E2"
    tool: str           # 使用的工具名
    args: dict[str, Any] # 传递的参数，可能包含依赖项如 "#E1"

@dataclass
class Plan:
    steps: list[PlanStep] # Plan 是由一系列步骤组成的列表
```
* [PlanStep](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/02-rewoo-plan-and-execute/code/main.py#L18-L22) 模拟了 Planner 输出的步骤节点。
* [Plan](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/02-rewoo-plan-and-execute/code/main.py#L25-L27) 包含了所有步骤。

### 2. 工具注册表
```python
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., str]] = {}

    def register(self, name: str, fn: Callable[..., str]) -> None:
        self._tools[name] = fn

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        fn = self._tools.get(name)
        if fn is None:
            return f"error: unknown tool {name!r}"
        try:
            return fn(**args)
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"
```
* [ToolRegistry](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/02-rewoo-plan-and-execute/code/main.py#L30-L46) 维护了一个工具名到函数的映射。在运行阶段，通过 `dispatch` 动态调用对应的工具函数，并捕获异常返回错误字符串，防止程序崩溃。

### 3. 依赖解析与拓扑排序
这是 ReWOO 执行阶段的核心逻辑：
```python
REFERENCE_RE = re.compile(r"#E(\d+)")

def resolve_references(value: Any, evidence: dict[str, str]) -> Any:
    if not isinstance(value, str):
        return value
    return REFERENCE_RE.sub(lambda m: evidence.get(f"E{m.group(1)}", m.group(0)),
                            value)
```
* [resolve_references](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/02-rewoo-plan-and-execute/code/main.py#L50-L54) 使用正则匹配参数中的 `"#E1"`, `"#E2"`，并从 `evidence` 字典中取出之前执行完的步骤结果进行替换。

```python
def topological(plan: Plan) -> list[PlanStep]:
    resolved: list[PlanStep] = []
    known: set[str] = set()
    pending = list(plan.steps)
    while pending:
        progress = False
        rest: list[PlanStep] = []
        for step in pending:
            # 查找参数中依赖的所有步骤（如 E1, E2 等）
            refs = REFERENCE_RE.findall(str(step.args))
            # 如果所有的依赖步都已经执行完毕（存在于 known 中），则当前步骤可以执行
            if all(f"E{r}" in known for r in refs):
                resolved.append(step)
                known.add(step.id)
                progress = True
            else:
                rest.append(step)
        # 如果一轮循环下来没有任何步骤被解析，说明存在循环依赖（死锁）或未定义引用
        if not progress:
            raise RuntimeError("cyclic plan or unresolved reference")
        pending = rest
    return resolved
```
* [topological](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/02-rewoo-plan-and-execute/code/main.py#L57-L75) 实现了标准的**拓扑排序**。它检查尚未执行的 `pending` 步骤，一旦发现某步骤的依赖项都已在 `known`（已执行完成的步骤）中，就将其放入 `resolved` 并从 `pending` 中移除。这样保证了执行顺序一定是“先依赖、后消费”。

### 4. 执行器 Worker
```python
def run_workers(plan: Plan, tools: ToolRegistry) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for step in topological(plan):
        # 1. 替换参数中的占位符（如 #E1 -> Paris）
        bound_args = {k: resolve_references(v, evidence) for k, v in step.args.items()}
        # 2. 执行工具调用并存入 evidence
        evidence[step.id] = tools.dispatch(step.tool, bound_args)
    return evidence
```
* [run_workers](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/02-rewoo-plan-and-execute/code/main.py#L78-L83) 会按照拓扑排序后的顺序，依次替换变量、调用工具，并将输出存入 `evidence` 字典中。

### 5. Scripted Planner & Solver (脚本化的规划与求解)
由于是 Toy 级别，代码中避免了真实的 LLM API 调用，使用脚本写死了 Planner 的 Plan 输出和 Solver 的格式化输出：
```python
class ScriptedPlanner:
    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    def plan_for(self, question: str) -> Plan:
        return self.plan

class ScriptedSolver:
    def __init__(self, answer_template: str) -> None:
        self.template = answer_template

    def solve(self, question: str, plan: Plan, evidence: dict[str, str]) -> str:
        return self.template.format(**evidence)
```

### 6. ReWOO vs. ReAct 的 Token（字符数）开销估算
为了展示 ReWOO 的 Token 优势，代码对两者的输入字符数进行了模拟度量：
* **ReWOO 的 Token 统计** ([run_rewoo](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/02-rewoo-plan-and-execute/code/main.py#L130-L143))：
  $$\text{Total Chars} = \text{Planner Chars} + \text{Worker Chars} + \text{Solver Chars}$$
  其中 Planner 只跑一次，Worker 只接收当前的工具调用（无历史），Solver 只接收问题+所有的证据。整个过程没有历史冗余。
* **ReAct 的 Token 统计** ([run_react_mock](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/02-rewoo-plan-and-execute/code/main.py#L146-L156))：
  每一轮调用工具，都需要带上 `prompt_chars` (原始问题) + `history_chars` (前面所有轮次的 Thought + Action + Observation)。
  在第 $N$ 步，其携带的历史长度是线性的，导致总 Token 呈二次方累积。

### 7. 运行结果验证
执行该脚本，输出如下：
```text
PLAN
  E1: search({'query': 'capital of France'})
  E2: search({'query': 'population of #E1'})
  E3: round_million({'text': '#E2'})

EVIDENCE
  E1 -> Paris
  E2 -> 11.2 million metro
  E3 -> 11 million

FINAL: The capital of France is Paris; rounded population is 11 million.

TOKEN INTUITION (chars, approximate)
  react total  : 873
  rewoo total  : 495
  ratio        : 1.76x
```
在这个简单的 3 步任务中，即便问题很简单，ReAct 的模拟消耗字符数也是 ReWOO 的 **1.76倍**。如果是 HotpotQA 这种长上下文、多依赖的复杂推理数据集，论文实测 ReWOO 的 Token 消耗能降低将近 **5倍**。

---

## 三、总结与思考

1. **核心逻辑**：ReWOO 先让模型扮演 **Planner** 生成一份带有数据占位符的步骤依赖图，再通过 **拓扑排序** 驱动 **Worker** 执行这组图（替换占位符参数），最后由 **Solver** 根据收集好的数据和原始问题直接给出结论。
2. **适用性**：如果任务高度结构化，例如：“对比 A 公司的财报与 B 公司的财报”，Planner 可以直接下发两个独立子任务（并行查询 A 和 B 财报），再由 Solver 进行对比。这种场景最适合 ReWOO。
3. **延伸思考**：当真实的业务遇到无法预期的错误时（例如第一步查询失败导致接下来的步骤无法进行），往往需要在 ReWOO 的基础上引入 **Replanner（重新规划器）**，这就是 **Plan-and-Execute** 模式。
