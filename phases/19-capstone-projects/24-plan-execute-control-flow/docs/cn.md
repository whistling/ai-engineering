# Plan-Execute Control Flow

> A plan that cannot survive a failure is a script. A script that can replan is an agent. Build the replanner first.

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** 第13阶段 课程 01-07，第14阶段 01  
**Time:** ~90 分钟

## Learning Objectives
- 将计划表示为有序的、带类型的步骤列表，以便执行器可以推理进度和结果。
- 以受控的失败交接回到规划器来顺序执行步骤。
- 从当前游标重新规划，并在上下文中包含先前的错误，使下一个计划能够获得信息。
- 在每次修订时发出计划差异（plan diff），以便下游跟踪器或 UI 可以展示为何计划发生变化。
- 强制执行两个预算：一个硬性的步骤上限和一个硬性的重规划上限。

## Plan and execute, not chain-of-thought

思维链（chain-of-thought）代理会发出令牌并让循环猜测工具调用在哪里结束。计划-执行（plan-and-execute）代理先发出结构化的计划，然后确定性地执行每个步骤。计划是可以被外层框架（harness）检查的数据。执行则是框架将该数据通过调度器运行。

两部分：生成计划的规划器（planner）和运行计划的执行器（executor）。有趣的地方是当执行器遇到失败时会发生什么。三种选项：

```text
1. Abort         (return failed, surface the error)
2. Skip          (mark step failed, continue with the rest)
3. Replan        (hand the error to the planner, get a new plan from the cursor)
```

Replan 是将脚本变成代理的关键。

## The Step shape

```text
Step
  id              : int           (单个计划修订内单调递增)
  tool_name       : str
  args            : dict
  expected_outcome: str           (规划器声明的成功条件)
  result          : Any | None
  error           : str | None
```

`expected_outcome` 是规划器在步骤旁边发出的简短句子。执行器不会强制执行它。它用于两件事：重规划器在修订计划时会读取它；事件流会发出它，以便跟踪器可以显示“此步骤预期要做 X”。

## The planner shape

```python
def planner(goal: str, history: list[Step], last_error: str | None) -> list[Step]:
    ...
```

这是一个纯函数。`goal` 是用户目标。`history` 是已执行的步骤（其中已填充结果和错误）。`last_error` 在第一次调用时为 None，后续每次调用为最近的失败消息。规划器返回从游标开始的下一个计划。

规划器不知道执行器的存在。它不知道重试，也不知道超时。它只产生计划，仅此而已。

## The executor

执行器是一个小型状态机。每个步骤通过调度器运行。结果为三种之一：成功（success）、可重规划的失败（failure-replannable）、致命失败（failure-fatal）。可重规划的失败会将错误交还给规划器。致命失败（例如预算被超出、重规划上限达到）会返回 `FAILED` 会话结果。

```mermaid
stateDiagram-v2
    [*] --> EXEC
    state EXEC as 执行
    state NEXT as 下一个
    state DONE as 完成
    state REPLAN as 重新规划
    state FAILED as 失败
    EXEC --> NEXT: 成功
    NEXT --> EXEC: n+1 < len(plan)
    NEXT --> DONE: n+1 == len(plan)
    EXEC --> REPLAN: 失败
    REPLAN --> EXEC: 新计划, replans_used < max_replans
    REPLAN --> FAILED: replans_used >= max_replans
    FAILED --> [*]
    DONE --> [*]
```

## Plan diffs on revision

当规划器在失败后返回新计划时，执行器会发出一个包含三个字段的 `plan.diff` 事件。

```text
removed: list of step ids that were in the old plan and are not in the new
added  : list of step ids in the new plan that were not in the old
revised: list of step ids whose tool_name or args changed
```

跟踪器或 UI 可以将其渲染为对被移除步骤的删除线以及对新增步骤的高亮。重点不是差异格式，而是修订是一个可见事件，而不是默默的重写。

## Two budgets, both hard

`max_steps` 限制整个会话的总步骤执行次数，包括重规划期间的执行。默认值为十二。一个线性五步的计划如果重规划两次并且每次添加三个步骤，就会达到十六次执行并超出预算。执行器会拒绝重规划并返回 FAILED。

`max_replans` 限制在首次计划之后规划器被调用的次数。默认值为五。这个限制更为重要。一个规划器如果连续五次返回相同的错误计划，否则会一直循环直到步骤预算耗尽。限定重规划次数可以更快地暴露失败并使原因更清晰。

## The deterministic planner in this lesson

本课不调用模型。课程附带的确定性规划器根据 `last_error` 选择计划：

```text
last_error is None    -> emit a four-step plan
last_error matches X  -> emit a three-step plan that routes around X
last_error matches Y  -> emit a two-step plan that gives up gracefully
otherwise             -> return [] (signals nothing to replan)
```

这足以测试执行器在每条转换路径上的行为：成功、重规划一次、重规划两次、重规划耗尽，以及步骤预算耗尽。

## Result shape

```text
SessionResult
  status      : "completed" | "failed"
  reason      : str     ("goal_met" | "step_budget" | "replan_budget" | "no_plan")
  history     : list[Step]
  revisions   : list[PlanDiff]
  events      : list[Event]
```

第二十课的 harness 循环可以直接读取这个结果。第二十三课的调度器用于执行每个步骤。第二十一课的注册表用于验证每个步骤的 args。第二十二课的传输层会通过 JSON-RPC 将整个流程展示给模型客户端。

## How to read the code

`code/main.py` 定义了 `PlanExecuteAgent`、`Step`、`PlanDiff`、`SessionResult` 和确定性规划器。执行器是一个单一的 `run(goal)` 方法，返回一个 `SessionResult`。计划差异通过比较步骤 id 和 `(tool_name, args)` 元组计算得出。

`code/tests/test_agent.py` 覆盖了线性成功、计划中期失败后重规划一次、重规划耗尽返回 `failed:replan_budget`、步骤预算耗尽，以及 plan-diff 事件格式等用例。

## Going further

当你把这个框架接到真实模型上时，会想要做的两个扩展。首先，部分计划缓存（partial-plan caching）：当一个计划在六个步骤中前面三步成功然后失败时，你不希望重新运行前面三步。执行器已经保留了历史；规划器只需读取它即可。其次，平行分支（parallel branches）：当前执行器是严格顺序的。一个发出独立分支的规划器（使用 `gather_step` 而不是 `next_step`）可以通过调度器并发运行两个工具调用。

两者都会增加实际复杂度。两者在线性执行器稳定之后更容易添加。本课正是为了固定线性执行器而设计的。