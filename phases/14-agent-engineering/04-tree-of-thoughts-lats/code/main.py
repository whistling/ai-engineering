"""Toy Tree-of-Thoughts BFS and LATS MCTS on a stylized arithmetic search.

Task: given integers [4, 6, 4, 1], find an expression using +, -, *, / that
evaluates to 24. This mirrors the Game of 24 benchmark from Yao et al.

ToT is a BFS with a prompted value function. LATS is MCTS over the same
search space with UCT selection.

Stdlib only; no LLM. Value function is symbolic (distance from 24).
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field


# 初始给定的数字列表与目标值
NUMBERS = [4, 6, 4, 1]
TARGET = 24
OPS = ["+", "-", "*", "/"]


@dataclass
class Node:
    """表示搜索树中的一个节点（推理状态）。

    state: 当前状态中的数字列表（元组形式，例如 [24, 4, 1]）
    trace: 记录从根节点到达此状态所执行的操作步骤（轨迹）
    visits: 该节点在蒙特卡洛树搜索（MCTS）中被访问的次数 N
    value_sum: 探索时该节点及其子孙节点获得的估值（Reward）累计总和
    children: 子节点列表
    """
    state: tuple[float, ...]
    trace: list[str]
    visits: int = 0
    value_sum: float = 0.0
    children: list["Node"] = field(default_factory=list)

    @property
    def q(self) -> float:
        """平均价值估值 Q = 累计估值 / 访问次数"""
        return self.value_sum / self.visits if self.visits else 0.0


def evaluate(a: float, op: str, b: float) -> float | None:
    """对两个数执行指定的算术操作，遇到除零时返回 None。"""
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b if b != 0 else None
    return None


def expand(node: Node) -> list[Node]:
    """生成当前节点所有可能的下一步子状态。

    从当前状态的数字元组中任选两个进行所有可行的四则运算，并生成新节点。
    这相当于大模型生成候选动作（Policy）的步骤。
    """
    children: list[Node] = []
    state = node.state
    if len(state) < 2:
        return children
    # 两两数字进行组合
    for i, j in itertools.combinations(range(len(state)), 2):
        for op in OPS:
            a, b = state[i], state[j]
            v = evaluate(a, op, b)
            if v is None:
                continue
            # 剩余未被选取的数字
            remaining = [s for k, s in enumerate(state) if k not in (i, j)]
            # 将新算出的数放回，并按从大到小降序排列以统一状态表达
            new_state = tuple(sorted(remaining + [v], reverse=True))
            step = f"{a}{op}{b}={v}"
            children.append(Node(state=new_state, trace=node.trace + [step]))
    return children


def value(node: Node) -> float:
    """估值函数（模拟 LLM 的 Self-Evaluation 机制）。

    根据当前状态中数字距离目标 24 的偏差来打分。
    完全等于 24 时得满分 1.0，偏离越远分数越低。
    """
    if len(node.state) == 1:
        result = node.state[0]
        return 1.0 if abs(result - TARGET) < 1e-6 else -abs(result - TARGET) / 100.0
    best_distance = min(abs(v - TARGET) for v in node.state)
    return -best_distance / 100.0


def tot_bfs(root: Node, max_expansions_per_level: int = 8,
            max_depth: int = 3) -> tuple[Node | None, int]:
    """Tree of Thoughts 广度优先搜索（BFS）算法实现。

    max_expansions_per_level: 每一层保留的优秀候选节点上限（相当于 Beam Width）
    max_depth: 最大搜索深度（最多进行 3 次数字合并）
    """
    frontier = [root]
    expansions = 0
    for _ in range(max_depth):
        scored: list[tuple[float, Node]] = []
        for node in frontier:
            # 扩展当前层的所有边界节点
            for child in expand(node):
                expansions += 1
                scored.append((value(child), child))
                # 提前找到精准解，直接返回
                if value(child) > 0.99:
                    return child, expansions
        # 根据自我评估得分，降序排序
        scored.sort(key=lambda p: p[0], reverse=True)
        # 只保留分数最高的前 max_expansions_per_level 个节点作为下一层的搜索边界
        frontier = [n for _, n in scored[:max_expansions_per_level]]
    best = max(frontier, key=value) if frontier else None
    return best, expansions


def uct(parent: Node, child: Node, c: float = 1.4) -> float:
    """计算子节点的置信区间上限（UCT 值），用于在 MCTS 中平衡 Exploitation 与 Exploration。

    c: 探索常数。值越大越倾向于探索访问次数较少的未知路径。
    """
    if child.visits == 0:
        # 未访问过的节点优先级最高，确保每个分支至少被尝试一次
        return float("inf")
    return child.q + c * math.sqrt(math.log(parent.visits) / child.visits)


def select(node: Node) -> Node:
    """在已建好的树结构中，利用 UCT 指标一路向下选择，直到未完全展开的叶子节点。"""
    while node.children:
        node = max(node.children, key=lambda ch: uct(node, ch))
    return node


def simulate(node: Node, depth: int, rng: random.Random) -> float:
    """从给定节点开始向下进行模拟（Rollout 快速推导）。

    每一步随机挑选一个子节点（Policy 随机行动），直至叶子状态，最后用 value 计算其评估分数。
    """
    current = node
    for _ in range(depth):
        options = expand(current)
        if not options:
            break
        current = rng.choice(options)
    return value(current)


def backprop(path: list[Node], reward: float) -> None:
    """反向传播：沿探索路径向上回传 Rollout 奖励，更新沿途所有祖先节点的 visits 和 value_sum。"""
    for n in path:
        n.visits += 1
        n.value_sum += reward


def mcts(root: Node, iterations: int, rng: random.Random) -> tuple[Node, int]:
    """LATS (Language Agent Tree Search) 的 MCTS 搜索流程控制。

    iterations: MCTS 迭代次数
    """
    expansions = 0
    for _ in range(iterations):
        path = [root]
        cur = root
        
        # 1. 选择 (Selection)：根据 UCT 寻找最优发展路线
        while cur.children:
            cur = max(cur.children, key=lambda ch: uct(cur, ch))
            path.append(cur)
            
        # 2. 扩展 (Expansion)：如果节点已被访问过，则生成其所有子节点并选取首个进行扩展
        if cur.visits > 0 and len(cur.state) > 1:
            cur.children = expand(cur)
            expansions += len(cur.children)
            if cur.children:
                cur = cur.children[0]
                path.append(cur)
                
        # 3. 模拟 (Simulation)：从当前被展节点执行快速 Rollout 推导打分
        reward = simulate(cur, depth=max(0, 3 - len(cur.trace)), rng=rng)
        
        # 4. 反向传播 (Backpropagation)：回传更新路径节点的估值
        backprop(path, reward)
        
    # 迭代结束后，遍历整棵树返回估值最高的叶子节点作为最终解
    best_leaf = max(_all_leaves(root), key=value, default=root)
    return best_leaf, expansions


def _all_leaves(node: Node) -> list[Node]:
    """递归辅助函数：提取以该节点为根的树中所有的叶子节点。"""
    if not node.children:
        return [node]
    out: list[Node] = []
    for ch in node.children:
        out.extend(_all_leaves(ch))
    return out


def main() -> None:
    print("=" * 70)
    print("TREE OF THOUGHTS + LATS — Phase 14, Lesson 04")
    print("=" * 70)
    print(f"numbers: {NUMBERS}  target: {TARGET}")

    # 1. 运行 ToT BFS 算法并计时
    root_tot = Node(state=tuple(sorted(NUMBERS, reverse=True)), trace=[])
    best_tot, n_tot = tot_bfs(root_tot)
    print("\nToT BFS")
    print("-" * 60)
    if best_tot is not None:
        print(f"  best trace: {best_tot.trace}")
        print(f"  final state: {best_tot.state}  value: {value(best_tot):.3f}")
    print(f"  expansions: {n_tot}")

    # 2. 运行 LATS MCTS 算法
    rng = random.Random(7)
    root_lats = Node(state=tuple(sorted(NUMBERS, reverse=True)), trace=[])
    root_lats.children = expand(root_lats)
    for ch in root_lats.children:
        ch.visits = 0
    best_lats, n_lats = mcts(root_lats, iterations=80, rng=rng)
    print("\nLATS MCTS")
    print("-" * 60)
    print(f"  best trace: {best_lats.trace}")
    print(f"  final state: {best_lats.state}  value: {value(best_lats):.3f}")
    print(f"  node expansions: {n_lats}")

    print()
    print("Paper headlines (for reference):")
    print("  ToT Game-of-24:  GPT-4 CoT 4%  -> ToT 74%")
    print("  LATS HumanEval:  pass@1 92.7% with GPT-4 (SOTA at paper time)")
    print("  Cost: ToT uses 100-1000x the tokens of CoT. Use with intent.")


if __name__ == "__main__":
    main()
