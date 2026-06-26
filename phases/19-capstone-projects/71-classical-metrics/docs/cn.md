# Classical Metrics

> BLEU, ROUGE-L, F1, exact-match, accuracy。五个指标仍然占据大多数已发表的 LLM 评估数值。请从头实现每一个指标，这样你才知道这些数字的含义。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第19阶段 Track B 基础，lesson 70
**Time:** ~90 分钟

## Learning objectives

- 用明确的分词规则实现基于标记的 exact-match、F1 和 accuracy。
- 从零实现 BLEU-4：修改的 n-gram 精度、n=1 到 4 的几何平均、简短惩罚（brevity penalty）。
- 使用最长公共子序列实现 ROUGE-L，并用 F-beta（beta=1）组合精度与召回率。
- 在 dispatcher 中根据 lesson 70 的 metric_name 字段分派，使运行器保持与指标无关。
- 使用来自示例推导的参考向量来固定行为，而不是依赖第三方库。

## Why reimplement

你会看到论文报告 BLEU 28.3，而另一篇报告 BLEU 0.283。你会发现两个库给出的 ROUGE-L 相差十点，因为一个库将文本截断并小写化，而另一个没有。停止困惑的最快方法是自己写实现，然后指出选择分词器的那一行以及应用平滑的那一行。之后对比论文之间的数值只需要阅读指标设置，而不是争论使用了哪个库。

标准库加上 numpy 就够了。BLEU 是计数和截断。ROUGE-L 是动态规划。F1 是令牌集合的交集。最难的是选择一个分词器并坚持它。

## Tokenisation

分词器为 `re.findall(r"\w+", text.lower())`。小写，字母数字连续片段，丢弃标点。本课中的每个指标都使用这个精确的分词器。运行器无权选择。如果你更换分词器，那就是在跑另一个基准。

```python
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
def tokenize(text):
    return TOKEN_RE.findall(text.lower())
```

这是一个刻意的简化。生产环境会关心中日韩（CJK）、缩写和代码标识符。要点是：分词器是一个契约，而不是一个可调旋钮。

## Exact match

```python
def exact_match(pred, targets):
    return float(any(pred.strip() == t.strip() for t in targets))
```

它在每个任务上返回 1.0 或 0.0。整个数据集上的聚合是均值。这是算术题、多选题和短分类任务的主力。

## Token-level F1

为预测和目标构建标记多重集合。精度是多重集合交集除以预测多重集合大小。召回是相同的交集除以目标多重集合大小。F1 是调和平均。实现处理空预测和空目标的边界情况。

```mermaid
flowchart LR
    A[pred 文本] -->|tokenize| P[pred 标记]
    B[target 文本] -->|tokenize| T[target 标记]
    P --> X[多重集合交集]
    T --> X
    X --> PR[精度 = 交集 / 预测数量]
    X --> RE[召回 = 交集 / 目标数量]
    PR --> F[F1 = 2 * P * R / (P + R)]
    RE --> F
```

对于多目标任务，我们在目标列表中取最高 F1。这与文献中广泛报道的 SQuAD 风格行为一致。

## BLEU-4

BLEU 是经典的机器翻译指标，在摘要工作中仍然出现。我们使用的形式是语料级别的 BLEU-4，带有标准的简短惩罚，并对修改后的 n-gram 计数采用加一平滑（additive-one smoothing），以便单个缺失的 4-gram 不会把分数推到零。

对于每个候选-参考对，我们计算 n=1,2,3,4 的修改后 n-gram 精度。修改后精度将候选 n-gram 的计数按参考中该 n-gram 的最大计数进行截断，防止候选通过重复短语来人为抬高值。四个精度的几何平均再乘以简短惩罚（brevity penalty）。

```mermaid
flowchart TD
    A[候选 标记] --> B[统计 n-gram，n=1..4]
    R[参考 标记] --> C[每个 n-gram 的最大计数]
    B --> D[截断后的 n-gram 计数]
    C --> D
    D --> E[修改后精度 p_n]
    A --> F[候选 长度 c]
    R --> G[参考 长度 r]
    F --> BP[BP = 1 当 c >= r 否则 exp(1 - r/c)]
    G --> BP
    E --> M[p_n 的几何平均]
    M --> S[BLEU = BP * 几何平均]
    BP --> S
```

平滑规则是 Lin 和 Och 所称的 method 1：在取对数之前，对每个 n-gram 精度的分子和分母都加一。这避免了当参考没有匹配的 4-gram 时出现 `log 0`，并且在较长的候选上接近未平滑值。

## ROUGE-L

ROUGE-L 比较候选和参考标记序列的最长公共子序列（LCS）。LCS 捕捉了词序，但不强制连续性，这也是它成为默认摘要指标的原因。我们用标准的动态规划表来计算 LCS 长度，然后把召回定义为 `lcs / 参考长度`，精度为 `lcs / 候选长度`，并用 F-beta（beta=1，即对称 F1）进行结合。

```python
def lcs_length(a, b):
    n, m = len(a), len(b)
    dp = numpy.zeros((n + 1, m + 1), dtype=int)
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1, j+1] = dp[i, j] + 1
            else:
                dp[i+1, j+1] = max(dp[i+1, j], dp[i, j+1])
    return int(dp[n, m])
```

numpy 表格使实现可读；纯 Python 列表也可行。选择使用 ROUGE-L 的任务要承担每个任务 O(n m) 的代价。对于典型的摘要长度，这通常低于一毫秒。

## Accuracy

对于多目标分类任务，accuracy 简化为针对单个归一化目标的 exact-match。我们将其作为单独函数暴露，以便 dispatcher 可以根据 `metric_name` 分派，而无需在运行器内部进行字符串比较。

## Dispatch contract

单一入口是 `score(metric_name, prediction, targets)`。它返回一个范围在 `[0, 1]` 内的浮点数。运行器不在内部基于指标名称分支。它把调用交给这个接口并写出结果。这是 lesson 75 将与 lesson 70 的任务规范对接的表面契约。

```python
def score(metric_name, pred, targets):
    if metric_name == "exact_match":
        return exact_match(pred, targets)
    if metric_name == "f1":
        return max(f1_score(pred, t) for t in targets)
    if metric_name == "bleu_4":
        return max(bleu4(pred, t) for t in targets)
    if metric_name == "rouge_l":
        return max(rouge_l(pred, t) for t in targets)
    if metric_name == "accuracy":
        return accuracy(pred, targets)
    raise ValueError(f"unknown metric_name: {metric_name}")
```

`code_exec` 在 lesson 72 中处理，并在那里插入到 dispatcher。

## What this lesson does not do

它不调用模型。它不会对生成结果做归一化，除了 lesson 70 已经规定的后处理规则。它不计算置信区间。它不做 BLEURT 或 BERTScore（那些需要模型，属于另一课）。要点是底线：五个指标，一个分词器，一个分派表。

## How to read the code

`main.py` 将每个指标定义为独立函数，并包含 dispatcher。参考向量位于文件底部的 `_reference_examples` 区块。演示对八个示例运行 dispatcher 并打印每个指标的分数。`code/tests/test_metrics.py` 中的测试固定了参考向量并覆盖每个边界情况（空预测、空参考、无共享标记、完全匹配、重复短语截断）。

从 `main.py` 顶部自上而下阅读。函数按复杂度排序。exact_match 和 accuracy 各一行。F1 六行。BLEU 和 ROUGE-L 是重量级部分，并包含关于平滑规则和 LCS 递归的详细注释。

## Going further

经典指标是必要的，但不充分。它们奖励表面重合而忽略语义。解决办法是在经典基线之上叠加基于模型的指标（BLEURT、BERTScore、GEval），前提是你已经信任了经典基线。这是后续的课题。现在：让这五个指标工作、用测试固定它们，你就拥有了一个可审计、快速且可复现的指标栈。