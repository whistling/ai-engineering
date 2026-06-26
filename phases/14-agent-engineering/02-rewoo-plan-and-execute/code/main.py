"""Toy ReWOO — Planner, Workers, Solver. Stdlib only.

Demonstrates the decoupled pattern from Xu et al. (arXiv:2305.18323):
  1. Planner emits a DAG of (tool, args) steps with references (#E1, #E2, ...).
  2. Workers run each step in topological order.
  3. Solver composes the final answer from question + plan + evidence.

Compare run_rewoo() vs run_react() at the bottom for token-use intuition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PlanStep:
    """表示规划中的单个执行步骤。

    id: 步骤的唯一标识（例如 "E1", "E2" 等，用作后续步骤的占位符）
    tool: 要调用的工具名称
    args: 传递给工具的参数字典，值中可能包含前置步骤的引用（如 "#E1"）
    """
    id: str
    tool: str
    args: dict[str, Any]


@dataclass
class Plan:
    """表示整个规划方案，包含一系列的执行步骤（有向无环图）。"""
    steps: list[PlanStep]


class ToolRegistry:
    """工具注册表，用于管理可用的外部工具并统一进行分发调用。"""
    def __init__(self) -> None:
        # 内部存储：工具名称 -> 可调用函数（返回值为字符串形式的执行结果）
        self._tools: dict[str, Callable[..., str]] = {}

    def register(self, name: str, fn: Callable[..., str]) -> None:
        """注册一个新工具"""
        self._tools[name] = fn

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        """根据工具名称分发并调用对应工具，同时做好异常保护"""
        fn = self._tools.get(name)
        if fn is None:
            return f"error: unknown tool {name!r}"
        try:
            return fn(**args)
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"


# 用于匹配前置步骤引用的正则表达式，如匹配 "#E1" 中的 "1"
REFERENCE_RE = re.compile(r"#E(\d+)")


def resolve_references(value: Any, evidence: dict[str, str]) -> Any:
    """解析并替换参数值中的前置步骤引用（占位符）。

    如果参数是字符串且包含像 "#E1" 的引用，则使用已经算出来的 E1 的实际结果进行替换。
    """
    if not isinstance(value, str):
        return value
    # 将找到的 "#E<N>" 替换为 evidence 字典中对应的键值 "E<N>" 的结果，若找不到则保持原样
    return REFERENCE_RE.sub(lambda m: evidence.get(f"E{m.group(1)}", m.group(0)),
                            value)


def topological(plan: Plan) -> list[PlanStep]:
    """对规划中的步骤进行拓扑排序，以确保每个步骤在其所有依赖的步骤执行完之后才执行。

    如果在排序过程中发现有循环依赖或无法解析的引用，则抛出异常。
    """
    resolved: list[PlanStep] = []  # 已排好序的步骤列表
    known: set[str] = set()        # 记录已被解析的步骤 ID（如 "E1"）
    pending = list(plan.steps)     # 待解析的步骤列表
    while pending:
        progress = False
        rest: list[PlanStep] = []
        for step in pending:
            # 查找当前步骤的 args 参数中引用的所有前置依赖项编号（例如 ["1", "2"]）
            refs = REFERENCE_RE.findall(str(step.args))
            # 如果所有的依赖步都已经在 known 集合中（说明对应的前置数据已具备），即可安全执行当前步
            if all(f"E{r}" in known for r in refs):
                resolved.append(step)
                known.add(step.id)
                progress = True
            else:
                # 还有依赖项没被满足，留到下一轮继续检查
                rest.append(step)
        # 如果一轮循环下来 pending 没有任何减少，说明步骤之间存在循环依赖或引用了不存在的步骤
        if not progress:
            raise RuntimeError("cyclic plan or unresolved reference")
        pending = rest
    return resolved


def run_workers(plan: Plan, tools: ToolRegistry) -> dict[str, str]:
    """按照拓扑排序的顺序依次执行所有规划步骤，并收集证据（执行结果）。"""
    evidence: dict[str, str] = {}
    for step in topological(plan):
        # 1. 替换参数中所有的前置引用（例如：将 "population of #E1" 替换为 "population of Paris"）
        bound_args = {k: resolve_references(v, evidence) for k, v in step.args.items()}
        # 2. 调用相应的工具获取执行结果，并存入证据库中
        evidence[step.id] = tools.dispatch(step.tool, bound_args)
    return evidence


class ScriptedPlanner:
    """脚本化规划器（模拟 LLM Planner 的输出逻辑）。在此 Toy 版本中，直接返回固定的静态 Plan。"""
    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    def plan_for(self, question: str) -> Plan:
        return self.plan


class ScriptedSolver:
    """脚本化求解器（模拟 LLM Solver 的总结输出逻辑）。

    接收一个模板，并将最终收集齐的证据格式化输出，得到最终答案。
    """
    def __init__(self, answer_template: str) -> None:
        self.template = answer_template

    def solve(self, question: str, plan: Plan, evidence: dict[str, str]) -> str:
        return self.template.format(**evidence)


def fake_search(query: str) -> str:
    """模拟搜索引擎的简易工具"""
    if "capital of france" in query.lower():
        return "Paris"
    if "population of paris" in query.lower():
        return "11.2 million metro"
    if "capital of germany" in query.lower():
        return "Berlin"
    return f"no result for {query!r}"


def rounded_million(text: str) -> str:
    """从文本中提取数字并四舍五入到百万的简易工具"""
    m = re.search(r"([0-9]+\.?[0-9]*)", text)
    if not m:
        return "unknown"
    return f"{round(float(m.group(1)))} million"


@dataclass
class ReWOORun:
    """表示一次完整的 ReWOO 运行状态及其字符（Token）开销估算。"""
    question: str
    plan: Plan
    evidence: dict[str, str] = field(default_factory=dict)
    answer: str = ""
    planner_chars: int = 0
    worker_chars: int = 0
    solver_chars: int = 0


def run_rewoo(question: str, planner: ScriptedPlanner,
              tools: ToolRegistry, solver: ScriptedSolver) -> ReWOORun:
    """调度 ReWOO 的核心流程：规划 -> 执行 -> 求解，并统计输入输出字符开销。"""
    # 1. 规划阶段：Planner 仅根据 question 产生 Plan，此时没有真实的 Observation 参与
    plan = planner.plan_for(question)
    planner_chars = len(question) + sum(len(s.tool) + len(str(s.args))
                                        for s in plan.steps)

    # 2. 执行阶段：Workers 并发/顺序调用工具收集 evidence
    evidence = run_workers(plan, tools)
    worker_chars = sum(len(str(s.args)) + len(v) for s, v in zip(plan.steps,
                                                                 evidence.values()))

    # 3. 求解阶段：Solver 根据 question + evidence 合成最终答案
    answer = solver.solve(question, plan, evidence)
    solver_chars = len(question) + worker_chars + len(answer)

    return ReWOORun(question=question, plan=plan, evidence=evidence,
                    answer=answer,
                    planner_chars=planner_chars, worker_chars=worker_chars,
                    solver_chars=solver_chars)


def run_react_mock(question: str, tools: ToolRegistry,
                   trajectory: list[tuple[str, dict[str, Any]]]) -> int:
    """模拟 ReAct 架构的字符（Token）开销。

    在 ReAct 中，每前进一步，都会将前面的“问题 + 历史 Thought + 历史 Action + 历史 Observation”
    作为新的 Prompt 输入给模型。随着步数加深，其上下文体积呈二次方/线性增长累加。
    """
    prompt_chars = len(question)
    total = 0
    history_chars = 0
    for name, args in trajectory:
        # 每一轮的 LLM 输入：原始 prompt + 累计的历史信息 + 当前规划的 Action
        total += prompt_chars + history_chars + len(name) + len(str(args))
        # 执行工具调用，得到 Observation
        obs = tools.dispatch(name, args)
        # 将本次 Action 和 Observation 累加进历史信息中
        history_chars += len(name) + len(str(args)) + len(obs) + 40  # 40 为 Thought 描述的概算
    # 最后一轮 LLM 求解总结的输入
    total += prompt_chars + history_chars
    return total


def main() -> None:
    print("=" * 70)
    print("REWOO — Planner, Workers, Solver (Phase 14, Lesson 02)")
    print("=" * 70)

    # 注册可用工具
    tools = ToolRegistry()
    tools.register("search", fake_search)
    tools.register("round_million", rounded_million)

    # 静态实例化一个规划：
    # E1 搜法国首都 -> E2 基于 E1 的结果查人口 -> E3 对 E2 的人口做四舍五入
    plan = Plan(steps=[
        PlanStep("E1", "search", {"query": "capital of France"}),
        PlanStep("E2", "search", {"query": "population of #E1"}),
        PlanStep("E3", "round_million", {"text": "#E2"}),
    ])
    planner = ScriptedPlanner(plan)
    solver = ScriptedSolver(
        "The capital of France is {E1}; rounded population is {E3}."
    )

    # 运行 ReWOO 调度流程
    run = run_rewoo("What is the population of the capital of France, rounded?",
                    planner, tools, solver)

    print("\nPLAN")
    for step in run.plan.steps:
        print(f"  {step.id}: {step.tool}({step.args})")
    print("\nEVIDENCE")
    for k, v in run.evidence.items():
        print(f"  {k} -> {v}")
    print(f"\nFINAL: {run.answer}")

    # 模拟对比 ReAct 的字符消耗
    react_chars = run_react_mock(
        run.question, tools,
        [("search", {"query": "capital of France"}),
         ("search", {"query": "population of Paris"}),
         ("round_million", {"text": "11.2 million metro"})])
    rewoo_chars = run.planner_chars + run.worker_chars + run.solver_chars

    print("\nTOKEN INTUITION (chars, approximate)")
    print(f"  react total  : {react_chars}")
    print(f"  rewoo total  : {rewoo_chars}")
    print(f"  ratio        : {react_chars / max(rewoo_chars, 1):.2f}x")
    print("\npaper claim: ~5x fewer tokens on HotpotQA. toy approximates the shape.")


if __name__ == "__main__":
    main()
