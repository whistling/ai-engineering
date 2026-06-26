# Tree of Thoughts (思维树) 和 LATS：有意的搜索 (Deliberate Search)

> **核心思想：** 单一的思维链（CoT）轨迹没有回溯的余地。Tree of Thoughts (ToT, Yao et al., 2023) 将推理过程转变为一棵树，并对每个节点进行自我评估。LATS (Zhou et al., 2024) 进一步将 ToT、ReAct 和 Reflexion 统一在蒙特卡洛树搜索（MCTS）下。在“24点游戏”中，准确率从 4% (CoT) 暴增至 74% (ToT)；而 LATS 则在 HumanEval 代码生成上实现了 92.7% 的 pass@1 准确率。

---

## 一、文档核心内容解读 (`docs/en.md`)

[docs/en.md](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/04-tree-of-thoughts-lats/docs/en.md) 详细探讨了将 Agent 推理过程建模为“搜索”的思路，对比了 ToT 与 LATS 两种主流搜索架构。

### 1. 核心痛点
Chain-of-thought (思维链) 是一种线性的执行流。一旦模型在第一步做出了错误的判断，后续所有的推理都会建立在错误的假设之上。在“24点游戏”中，GPT-4 使用 CoT 只能达到 4% 的准确率，因为模型非常容易在前期挑错子表达式且无法回头。

推理过程需要具备：**提出多个候选方案、对方案进行评估、选择有前景的路径并在遇到死胡同（Dead End）时回溯**的能力。这就是**搜索**的作用。

### 2. 思维树 Tree of Thoughts (ToT)
ToT（NeurIPS 2023）将中间的每一个推理步骤建模为树的节点（“一个思维”）。
* 每个节点可以向外扩展出 $K$ 个子思维。
* 调用 LLM 进行**自我评估（Self-Evaluation）**，打分方式包括：划分 `sure / likely / impossible`、进行 `1..10` 评分或在候选方案中投票。
* 结合传统的图搜索算法（如 BFS 广度优先、DFS 深度优先或 Beam Search 束搜索）在树上探索。

### 3. LATS (Zhou et al., ICML 2024)
LATS 巧妙地将 ToT 的状态评估、ReAct 的动作触发和 Reflexion 的失败反思融合进了**蒙特卡洛树搜索（MCTS）**算法中。大模型在其中扮演三个角色：
1. **Policy (策略)**：提出下一步可选的 Action（ReAct 风格）。
2. **Value Function (值函数)**：对当前的局部路径进行评估打分（ToT 风格）。
3. **Self-reflector (自我反思器)**：在探索失败时，编写自然语言反思并喂给接下来的搜索轮次（Reflexion 风格）。

此外，LATS 还会把真实的工具 Observation 混合进值函数中，使搜索方向基于客观结果，而非大模型的一家之言。

### 4. MCTS（蒙特卡洛树搜索）的四个阶段
1. **Select (选择)**：利用 UCT 公式，从根节点向下选择最值得探索的子节点，直到叶子节点。
2. **Expand (扩展)**：使用 Policy 模型为选定的叶子节点生成 $K$ 个子节点。
3. **Simulate (模拟/Rollout)**：从子节点出发随机向下推导几步，并使用值函数（或环境 Reward）为这个最终状态评分。
4. **Backpropagate (反向传播)**：将分数沿路径传回，更新祖先节点的访问次数（Visits）和均值估值（Q）。

**UCT公式**：$Q(s, a) + c \times \sqrt{\frac{\ln N(s)}{N(s, a)}}$。第一项负责“利用（Exploitation，去得分最高的地方）”，第二项负责“探索（Exploration，去访问次数少的地方）”。

### 5. 搜索的代价与现实应用
搜索会导致 Token 消耗暴增。在 24 点游戏中，ToT 的消耗是 CoT 的 100 到 1000 倍。因此，在生产环境中，**搜索算法只被保留在特定的高要求领域**：
* 容易低成本评估、但解答难度极大的任务（例如使用单元测试作为 Value Function 的编程 Agent）。
* 要求正确性远高于运行时间开销的深度研究型 Agent。
* 只有当“任务复杂度 > 门槛”时，才会通过条件路由进入搜索子图。

---

## 二、代码详细解读 (`code/main.py`)

[code/main.py](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/04-tree-of-thoughts-lats/code/main.py) 用纯 Python 标准库分别实现了一个简易的 ToT (BFS) 和 LATS (MCTS)，用于解决数字 24 点游戏（给定 `[4, 6, 4, 1]`，通过运算凑齐 24）。

下面是主要代码模块的实现分析：

### 1. 树节点 Node
```python
@dataclass
class Node:
    state: tuple[float, ...]  # 节点状态，例如最初为 (6, 4, 4, 1)；当执行 6*4 之后更新为 (24, 4, 1)
    trace: list[str]          # 记录推导步骤的历史轨迹
    visits: int = 0           # 蒙特卡洛搜索中的访问次数 N
    value_sum: float = 0.0    # 蒙特卡洛搜索中累计的回报值总和
    children: list["Node"] = field(default_factory=list) # 子节点

    @property
    def q(self) -> float:
        # 估值 Q ＝ 累计回报总和 / 访问次数
        return self.value_sum / self.visits if self.visits else 0.0
```
* [Node](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/04-tree-of-thoughts-lats/code/main.py#L25-L36) 保存了树形推理结构中的所有状态，并提供了用于 UCT 计算的 `q` 属性。

### 2. 状态扩展与评估函数
```python
def expand(node: Node) -> list[Node]:
    children: list[Node] = []
    state = node.state
    if len(state) < 2:
        return children
    # 两两组合当前状态中的数字，并尝试所有的四则运算
    for i, j in itertools.combinations(range(len(state)), 2):
        for op in OPS:
            a, b = state[i], state[j]
            v = evaluate(a, op, b) # 排除除零等非法操作
            if v is None:
                continue
            # 将剩下的数字与运算新结果合并，并降序排列，组合成新的子节点
            remaining = [s for k, s in enumerate(state) if k not in (i, j)]
            new_state = tuple(sorted(remaining + [v], reverse=True))
            step = f"{a}{op}{b}={v}"
            children.append(Node(state=new_state, trace=node.trace + [step]))
    return children

def value(node: Node) -> float:
    # 距离 24 越近，评估分数越高 (符号距离)
    if len(node.state) == 1:
        result = node.state[0]
        return 1.0 if abs(result - TARGET) < 1e-6 else -abs(result - TARGET) / 100.0
    best_distance = min(abs(v - TARGET) for v in node.state)
    return -best_distance / 100.0
```
* [expand](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/04-tree-of-thoughts-lats/code/main.py#L50-L65) 模拟了大模型提出子思维的 Policy 动作。
* [value](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/04-tree-of-thoughts-lats/code/main.py#L68-L74) 是打分函数，模拟了大模型的自我评估（Self-Evaluation）。距离目标 24 差值越小，分数越高。凑出 24 时得分为 1.0。

### 3. Tree of Thoughts (BFS 广度优先搜索)
```python
def tot_bfs(root: Node, max_expansions_per_level: int = 8, max_depth: int = 3) -> tuple[Node | None, int]:
    frontier = [root] # 维护当前层的边界节点
    expansions = 0
    for _ in range(max_depth):
        scored: list[tuple[float, Node]] = []
        for node in frontier:
            for child in expand(node):
                expansions += 1
                scored.append((value(child), child))
                if value(child) > 0.99: # 找到解，立即返回
                    return child, expansions
        # 将所有子节点按照估值降序排序，只保留前 N 个表现最好的节点进入下一层
        scored.sort(key=lambda p: p[0], reverse=True)
        frontier = [n for _, n in scored[:max_expansions_per_level]]
    best = max(frontier, key=value) if frontier else None
    return best, expansions
```
* [tot_bfs](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/04-tree-of-thoughts-lats/code/main.py#L76-L91) 是典型的 ToT 实现。它在一层内全面铺开扩展，计算所有分支的 Self-Evaluation 值，然后实施 **Beam-Search (束搜索)** 过滤（保留最优秀的 $N$ 个，抛弃其余低分分支），再进入下一层。

### 4. LATS (MCTS 蒙特卡洛树搜索)
```python
def mcts(root: Node, iterations: int, rng: random.Random) -> tuple[Node, int]:
    expansions = 0
    for _ in range(iterations):
        path = [root]
        cur = root
        # 1. 选择 (Select)：利用 UCT 算法寻找最值得延伸的子叶子节点
        while cur.children:
            cur = max(cur.children, key=lambda ch: uct(cur, ch))
            path.append(cur)
        
        # 2. 扩展 (Expand)：如果选出的节点已被访问过且任务没结束，展开它的子树
        if cur.visits > 0 and len(cur.state) > 1:
            cur.children = expand(cur)
            expansions += len(cur.children)
            if cur.children:
                cur = cur.children[0]
                path.append(cur)
        
        # 3. 模拟 (Simulate/Rollout)：从当前选定位置开始随机推导到底，并由 value 打分评估
        reward = simulate(cur, depth=max(0, 3 - len(cur.trace)), rng=rng)
        
        # 4. 反向传播 (Backpropagate)：将评分沿着探索路径一路累加回根节点，更新 visits 和 Q
        backprop(path, reward)
        
    best_leaf = max(_all_leaves(root), key=value, default=root)
    return best_leaf, expansions
```
* [mcts](file:///Users/shmihanzhongqing/Workspace/ai-engineering-from-scratch/phases/14-agent-engineering/04-tree-of-thoughts-lats/code/main.py#L122-L139) 则是 LATS 的核心实现。它不像 BFS 那样死板地在每一层广撒网，而是结合节点均值估值 $Q$ 和访问热度（UCT公式），智能地选择最可能成功的分支进行纵向挖掘和 Rollout。

### 5. 运行结果分析

* **ToT (BFS) 表现**：
  * 推导轨迹：`['6*4=24', '4-1=3', '24+3=27']`（没凑出 24）
  * 扩展节点数：152
* **LATS (MCTS) 表现**：
  * 推导轨迹：`['6*4=24', '24*1=24']`（成功解出，最终状态中包含 24）
  * 扩展节点数：286

在限制层数和 Beam 宽度的简单 BFS 中，ToT 很容易丢失局部最优组合；而 LATS (MCTS) 在不断的 Rollout 模拟和反向传播修正下，即使需要更多的扩展尝试（286次），也能够精准锁定制胜路径。
