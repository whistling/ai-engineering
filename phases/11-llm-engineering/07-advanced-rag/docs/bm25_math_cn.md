# BM25 数理细节与公式推导（供手写实现参考）

为了从零实现 BM25 算法，你需要理解其背后的数学表达和参数控制：

## 1. 数学公式

对于查询 $q$ 和文档 $d$，BM25 的相关性得分计算公式如下：

$$BM25(q, d) = \sum_{t \in q} IDF(t) \cdot \frac{tf(t, d) \cdot (k_1 + 1)}{tf(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}$$

## 2. 公式核心拆解与代码对应

*   **逆文档频率 $IDF(t)$**：在代码中计算为 `math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)`，它在数学上等价于 $\ln \left( \frac{N + 1}{n(t) + 0.5} \right)$。其中 $N$ 是总文档数，$n(t)$ 是命中该词的文档数。利用 $0.5$ 作为平滑值，避免了分母为零或大词项出现负分的问题。
*   **词饱和度参数 $k_1$（默认 1.2）**：控制词频得分饱和的速度。随着词频 $tf \to \infty$，词频部分的得分贡献上限被锁定在 $k_1 + 1$。
*   **长度惩罚参数 $b$（默认 0.75）**：控制对长文档的惩罚力度。当 $b=1$ 时，根据文档长度与平均长度之比 $\frac{|d|}{avgdl}$ 进行完全的长度惩罚；当 $b=0$ 时，完全关闭长度惩罚。

## 3. 极简数理实操示例

为了直观地理解 BM25 的运作，我们假设一个由 3 篇极简文档组成的语料库（总文档数 $N = 3$），并使用超参数 $k_1 = 1.2, b = 0.75$：

*   **文档 1 ($D_1$)**: "Artificial intelligence is the future of AI technology" （长度 $|D_1| = 8$）
*   **文档 2 ($D_2$)**: "AI technology shapes advanced robotics and artificial neural networks" （长度 $|D_2| = 9$）
*   **文档 3 ($D_3$)**: "Robotics and automation are growing rapidly" （长度 $|D_3| = 6$）

此时，平均文档长度 $avgdl = \frac{8 + 9 + 6}{3} \approx 7.67$。

我们发起的查询为 **$q$: "AI robotics"**（包含词项 "ai" 和 "robotics"）。

### 第一步：计算各词项的 IDF 值
*   词项 **"ai"** 仅在 $D_1$ 和 $D_2$ 中出现过，因此 $n(\text{"ai"}) = 2$：
    $$IDF(\text{"ai"}) = \ln \left( \frac{3 - 2 + 0.5}{2 + 0.5} + 1 \right) = \ln(0.6 + 1) = \ln(1.6) \approx 0.470$$
*   词项 **"robotics"** 仅在 $D_2$ 和 $D_3$ 中出现过，因此 $n(\text{"robotics"}) = 2$：
    $$IDF(\text{"robotics"}) = \ln \left( \frac{3 - 2 + 0.5}{2 + 0.5} + 1 \right) = \ln(1.6) \approx 0.470$$

### 第二步：逐个文档计算 BM25 得分

*   **文档 $D_1$ 得分**：
    *   包含 "ai"（词频 $tf = 1$），不包含 "robotics"（$tf = 0$）。
    *   对于 "ai" 计算其分母部分中的阻尼常数 $K$：
        $$K_{1} = 1.2 \cdot \left(1 - 0.75 + 0.75 \cdot \frac{8}{7.67}\right) \approx 1.2 \cdot (0.25 + 0.782) \approx 1.238$$
    *   "ai" 对得分的贡献：
        $$Score(\text{"ai"}, D_1) = 0.470 \cdot \frac{1 \cdot (1.2 + 1)}{1 + 1.238} \approx 0.470 \cdot \frac{2.2}{2.238} \approx 0.462$$
    *   **$Score(q, D_1) \approx 0.462$**。

*   **文档 $D_2$ 得分**：
    *   包含 "ai"（$tf = 1$）和 "robotics"（$tf = 1$）。
    *   对于这两个词计算其阻尼常数 $K$（两词共享文档长度归一化）：
        $$K_{2} = 1.2 \cdot \left(1 - 0.75 + 0.75 \cdot \frac{9}{7.67}\right) \approx 1.2 \cdot (0.25 + 0.880) \approx 1.356$$
    *   "ai" 对得分的贡献：
        $$Score(\text{"ai"}, D_2) = 0.470 \cdot \frac{1 \cdot 2.2}{1 + 1.356} \approx 0.470 \cdot 0.934 \approx 0.439$$
    *   "robotics" 对得分的贡献同样为 $0.439$。
    *   **$Score(q, D_2) \approx 0.439 + 0.439 = 0.878$**。

*   **文档 $D_3$ 得分**：
    *   不包含 "ai"（$tf = 0$），包含 "robotics"（$tf = 1$）。
    *   对于 "robotics" 计算其阻尼常数 $K$：
        $$K_{3} = 1.2 \cdot \left(1 - 0.75 + 0.75 \cdot \frac{6}{7.67}\right) \approx 1.2 \cdot (0.25 + 0.587) \approx 1.004$$
    *   "robotics" 对得分的贡献：
        $$Score(\text{"robotics"}, D_3) = 0.470 \cdot \frac{1 \cdot 2.2}{1 + 1.004} \approx 0.470 \cdot \frac{2.2}{2.004} \approx 0.516$$
    *   **$Score(q, D_3) \approx 0.516$**。

### 排序结果与长度惩罚现象
最终的检索排序结果为：**$D_2$ ($0.878$) > $D_3$ ($0.516$) > $D_1$ ($0.462$)**。

> [NOTE]
> **长度惩罚效能对比**：
> 观察 $D_1$ 与 $D_3$ 的对比。它们在查询词项中各只包含了一个词（$D_1$ 命中 "ai"，$D_3$ 命中 "robotics"），且这两个词的全局 IDF 是完全一样的（均为 $0.470$）。
> 但是，$D_3$ 的长度较短（$|D_3|=6 < avgdl$），而 $D_1$ 的长度较长（$|D_1|=8 > avgdl$）。由于长度惩罚机制的作用，短文档 $D_3$ 在命中词时的最终得分（$0.516$）显著高于长文档 $D_1$（$0.462$）。这生动地证明了 BM25 长度惩罚参数 $b$ 对较短文档相关性的提振效果。
