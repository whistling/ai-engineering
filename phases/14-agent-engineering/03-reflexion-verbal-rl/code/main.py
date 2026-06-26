"""Toy Reflexion loop — Actor, Evaluator, Self-Reflector, Episodic memory.

Task: pick three integers from 1..9 that sum to a target. The Actor is
scripted to start with a bad strategy and adapt when reflections are present.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# 设定的解题目标：三个 1..9 的整数之和必须等于 TARGET (20)
TARGET = 20


@dataclass
class Reflection:
    """表示针对某次失败尝试做出的反思记录。

    trial: 尝试的轮次序号（1-indexed）
    text: 反思的自然语言文本描述
    """
    trial: int
    text: str


@dataclass
class EpisodicMemory:
    """情节记忆模块，模拟 Agent 的外部记忆缓冲区，用于记录并提供此前所有失败尝试的反思。"""
    items: list[Reflection] = field(default_factory=list)
    max_len: int = 6  # 限制记忆缓冲区的最大长度，避免随着步数增加导致 context 爆炸

    def add(self, r: Reflection) -> None:
        """向记忆库中添加一条新的反思记录，若超出最大限制则遵循先进先出原则移除旧记忆。"""
        self.items.append(r)
        if len(self.items) > self.max_len:
            self.items.pop(0)

    def as_prompt(self) -> str:
        """将情节记忆转换为可以拼接到系统提示词（Prompt）中的自然语言格式。"""
        if not self.items:
            return "(no prior reflections)"
        lines = [f"- trial {r.trial}: {r.text}" for r in self.items]
        return "\n".join(lines)


class Actor:
    """脚本化执行器（模拟大模型的动作生成）。

    无反思（EpisodicMemory 为空）时，该 Agent 会盲目重复失败方案 [1, 2, 3]；
    当观察到历史反思时，Agent 能根据记忆中的反思调整选择，并逐步逼近目标。
    """

    def act(self, memory: EpisodicMemory) -> list[int]:
        n = len(memory.items)
        if n == 0:
            # 初始状态或基线情况（无记忆）：返回盲目解 [1, 2, 3]，其和为 6
            return [1, 2, 3]
        if n == 1:
            # 拥有 1 条反思：通过反思知道和太小，尝试将数值调大为 [5, 6, 7]，其和为 18
            return [5, 6, 7]
        if n == 2:
            # 拥有 2 条反思：经过第二次微调，精准收敛到 [6, 7, 7]，其和为 20
            return [6, 7, 7]
        return [6, 7, 7]


def binary_evaluator(attempt: list[int], target: int) -> tuple[bool, int]:
    """外部评估器（Evaluator），对 Agent 产生的解进行评估。

    返回一个元组 (是否达成目标, 解与目标值之间的偏离值 delta)
    """
    total = sum(attempt)
    return total == target, total - target


class SelfReflector:
    """自我反思器（Self-Reflector），用于在任务失败后扮演“老师”给出自然语言层面的校准意见。"""
    def reflect(self, attempt: list[int], delta: int) -> str:
        # 如果 delta < 0，代表数值偏小，反馈要求增加数值
        if delta < 0:
            return f"sum {sum(attempt)} is {-delta} short; pick larger values"
        # 如果 delta > 0，代表数值超标，反馈要求减小数值
        if delta > 0:
            return f"sum {sum(attempt)} overshoots by {delta}; pick smaller values"
        return "succeeded"


@dataclass
class TrialResult:
    """用于存储单轮尝试的所有细节信息，方便最终对比统计。"""
    trial: int
    attempt: list[int]
    success: bool
    delta: int
    reflection: str


def run_reflexion(max_trials: int, use_memory: bool) -> list[TrialResult]:
    """核心调度循环：运行 Reflexion 环路。

    max_trials: 允许尝试的最大轮次
    use_memory: 是否启用情节记忆。如果关闭，模拟 baseline（每次尝试均处于无历史记忆状态）
    """
    actor = Actor()
    reflector = SelfReflector()
    memory = EpisodicMemory()
    trials: list[TrialResult] = []
    for t in range(1, max_trials + 1):
        # 决定当前 Actor 在执行决定时能否获取先前的失败记忆
        attempt = actor.act(memory if use_memory else EpisodicMemory())
        # 评估本轮计算结果
        success, delta = binary_evaluator(attempt, TARGET)
        # 反思器给出总结分析
        text = reflector.reflect(attempt, delta)
        # 记录结果
        trials.append(TrialResult(t, attempt, success, delta, text))
        if success:
            # 达成目标，提前退出环路
            break
        # 任务失败，将本轮的反思结果写入情节记忆，供下轮作为上下文使用
        memory.add(Reflection(trial=t, text=text))
    return trials


def summarize(trials: list[TrialResult], name: str) -> None:
    """格式化汇总输出多次尝试的轨迹细节"""
    print(f"\n{name}")
    print("-" * 60)
    for r in trials:
        mark = "OK " if r.success else "..."
        print(f"  trial {r.trial}: {r.attempt} sum={sum(r.attempt)} "
              f"delta={r.delta:+d} {mark} -> {r.reflection}")
    last = trials[-1]
    print(f"  final: {'success' if last.success else 'failed'} "
          f"at trial {last.trial}")


def main() -> None:
    print("=" * 70)
    print(f"REFLEXION — pick three ints in [1..9] summing to {TARGET}")
    print("Phase 14, Lesson 03")
    print("=" * 70)

    # 1. 运行基线测试组：关闭情节记忆。模型因记不住错误教训，始终在低效策略上打转
    trials_no_mem = run_reflexion(max_trials=4, use_memory=False)
    summarize(trials_no_mem, "BASELINE (no episodic memory)")

    # 2. 运行 Reflexion 实验组：打开情节记忆。模型利用前轮反思自适应优化
    trials_mem = run_reflexion(max_trials=4, use_memory=True)
    summarize(trials_mem, "REFLEXION (episodic memory on)")

    baseline_steps = len(trials_no_mem)
    reflex_steps = len(trials_mem)
    print()
    print(f"baseline used {baseline_steps} trials; reflexion used {reflex_steps}.")
    print("Without a reflection in the prompt, the scripted actor never adapts.")
    print("With one reflection, the actor corrects; with two, it converges.")


if __name__ == "__main__":
    main()
