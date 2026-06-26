# 层级架构及其失效模式

> 层级（Hierarchical）是监督者嵌套。管理者代理（manager agents）在子管理者之上，再在其之上是执行者。CrewAI 的 `Process.hierarchical` 是教科书式的实现：一个 `manager_llm` 动态地分配任务并验证输出。LangGraph 的等价实现是 `create_supervisor(create_supervisor(...))`。当任务本身就是组织结构图时，这是自然的模式。但它也最容易崩溃为管理回环 —— 管理者代理错误分配工作、误解子输出或无法达成共识。顺序流水线常常胜出。

**Type:** 学习 + 构建  
**Languages:** Python (stdlib)  
**Prerequisites:** 阶段 16 · 05（Supervisor Pattern）  
**Time:** ~60 分钟

## 问题

一旦监督者模式上手，接下来的自然问题是“如果执行者本身也是监督者怎么办？”团队下有子团队，公司有部门的部门。层级架构就是这种映射。

问题在于：LLM 管理者并不等同于人类管理者。人类管理者对其下属知道什么有稳定的先验。LLM 管理者每次都是从上下文重新推理组织结构。上下文出现微小漂移，整棵树就会错误分配工作。

## 概念

### 结构形态

```
                 管理者
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
      子管理者 A           子管理者 B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

每个内部节点负责计划、分派与合成。只有叶节点（workers）在执行具体工作。

### 适用场景

- **与真实组织映射清晰。** 如果真实任务是按部门分工（“法务审查文档、财务审查文档、工程审查文档，然后向高层汇总”），层级结构很自然。
- **局部汇总。** 每个子管理者在顶层管理者看到之前会先汇总其团队输出。顶层管理者看到的是三个子管理者的摘要，而不是十五个执行者的原始输出。

### 容易失效的地方

2026 年的事后分析不断发现三类失败模式：

1. Task assignment error（任务分配错误）。管理者读取目标后产生幻觉式的分解，并把任务分配给了错误的子管理者。由于子管理者顺从执行被分配的工作，错误直到顶层合成才暴露——这比人类更难在底层被发现。
2. Output misinterpretation（输出误解）。子管理者返回“无法验证声明 X”。顶层管理者将其汇总为“声明 X 未被确认”。在每一级上含义都在漂移。
3. Consensus loops（共识循环）。两个子管理者意见不合；顶层要求他们调和；他们又向下重新分派；执行者重跑；子管理者返回略有差异的答案；循环往复。CrewAI 的 `Process.hierarchical` 用步骤上限来防护，但上限本身又成了一个超参数。

### 决定性问题

顺序（线性流水） vs 层级：你的任务是真正有独立子团队，还是一个线性流程假装成树？如果是后者，使用顺序。如果是前者，使用层级，但要预算好明确的调和规则。

### CrewAI 的实现

`Process.hierarchical` 将一个 manager LLM 置于专门化小组（crews）之上。管理者会：

- 接收顶层任务，
- 将子任务分配给小组，
- 评估小组输出，
- 决定接受、重新分派或迭代。

文档：https://docs.crewai.com/en/introduction （查找 “Hierarchical Process” 于 Core Concepts 下）。

### LangGraph 的实现

LangGraph 使用嵌套的 `create_supervisor` 调用。内部 supervisor 有自己的图；外部 supervisor 将内部图视为一个不透明节点。对调试来说这比 CrewAI 更干净（你可以分别逐步执行每个图），但在表达树的动态重塑方面更困难。

参考：https://reference.langchain.com/python/langgraph-supervisor。

## 构建它

`code/main.py` 运行一个三层层级：

- 顶层管理者：把任务拆分为“engineering”和“legal”分支，
- engineering 子管理者：再拆分为“frontend”和“backend”执行者，
- legal 子管理者：只有一个执行者。

演示对比了正常路径（大家一致）与一个被扰动的路径——在扰动路径中，顶层管理者将“legal”错误标记为“finance”，并观察错误如何级联：子管理者顺从地做了财务相关工作，顶层合成器报告的是财务结论，原始的法律问题没有被回答。

运行：

```
python3 code/main.py
```

输出会并列显示“被要求的是什么”与“交付了什么”。

## 使用它

`outputs/skill-hierarchy-fitness.md` 评估某个给定任务是否应使用层级、顺序或扁平监督。输入：任务描述、组织结构、调和预算。输出：模式推荐以及需要防护的具体失败模式。

## 发布建议

如果你要发布层级架构：

- **将树深度限制为 2。** 三层已经将大多数错误隐藏，难以观测。
- **明确的调和预算。** 设定最大回合数，超过则顶层管理者必须作出决定。通常为 2 回合。
- **在每次合成上记录溯源（provenance）。** 每个节点的摘要必须引用哪些叶节点输出产生了它。
- **在分解漂移上报警。** 记录管理者每一步的分解；与用户查询做 diff。如果分解不再覆盖查询，触发警报。

## 练习

1. 运行 `code/main.py` 并比较正常路径与被扰动路径。在顶层输出完全偏离用户问题之前，需要多少层管理交接？
2. 添加第三层（top → sub → sub-sub → worker）。测量在深度增加时，被扰动路径自我纠正的频率与完全偏离的频率如何变化。
3. 在每个子管理者处实现一个“金丝雀”执行者（canary worker），始终以原始用户问题作答（不变）。使用金丝雀的回答检测分解漂移。当金丝雀与合成答案不一致时，管理者应如何反应？
4. 阅读 CrewAI 的 `Process.hierarchical` 文档。识别 CrewAI 采用的一个具体护栏（例如步骤上限、manager_llm 约束），并描述它针对哪种失败模式。
5. 将嵌套的 LangGraph supervisors 与 CrewAI 的层级实现比较。哪种在检测调和循环（reconciliation loops）时更便宜（更容易检测到）？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Hierarchical | “组织结构图模式” | 监督者之上是监督者；只有叶节点执行具体工作。 |
| Manager LLM | “老板” | 在内部节点上负责分解、分配与验证的 LLM。 |
| Decomposition drift | “老板偏离了主线” | 顶层管理者的分解不再覆盖原始问题。 |
| Reconciliation loop | “无休止的会议” | 子管理者意见不合；顶层重新分派；执行者重跑；直到预算耗尽的循环。 |
| Depth-2 ceiling | “不要超过两层” | 经验性护栏：3 层及以上会崩溃可观测性。 |
| Canary question | “每层的基准真相” | 一个始终被问到原始查询且不被改写的执行者，用来检测分解漂移。 |
| Provenance chain | “谁说了什么” | 从每次合成追溯到产生该合成的叶节点输出的链路。 |

## 延伸阅读

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction) — 教科书式的层级与 manager LLM  
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — 通过 `create_supervisor` 实现的嵌套监督者  
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 为什么 Anthropic 故意选择扁平监督而非分层  
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分类法；关于协调失败的章节记录了分解漂移问题