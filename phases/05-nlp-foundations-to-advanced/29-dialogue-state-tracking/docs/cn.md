# 对话状态跟踪

> “我想要一家北边的便宜餐厅……不，改成中档的……再加上意大利菜。” 三轮对话，三次状态更新。DST 保持槽-值字典与对话同步以保证预订正确执行。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 5 · 17（聊天机器人），Phase 5 · 20（结构化输出）  
**Time:** ~75 分钟

## 问题

在任务型对话系统中，用户的目标被编码为一组槽-值对：`{cuisine: italian, area: north, price: moderate}`。每个用户回合可以添加、修改或删除一个槽。系统必须读取完整的对话并正确输出当前状态。

错误一个槽就会导致系统预订错误的餐厅、安排错误的航班或扣错信用卡。DST 是用户话语与后端执行之间的枢纽。

为什么在 2026 年仍然重要（尽管有 LLM）：

- 合规敏感领域（银行、医疗、航空预订）需要确定性的槽值，而不是自由形式的生成。
- 使用工具的 agent 在调用 API 前仍然需要槽解析。
- 多轮纠正比看起来更难：比如 “不，其实改到周四”。

现代流水线：经典的 DST 概念 + LLM 提取器 + 结构化输出护栏（structured-output guardrails）。

## 概念

![DST: dialog history → slot-value state](../assets/dst.svg)

**任务结构。** 一个 schema 定义了域（restaurant、hotel、taxi）及其槽（cuisine、area、price、people）。每个槽可以为空，填入来自封闭集合的值（price: {cheap, moderate, expensive））或自由形式的值（name: "The Copper Kettle"）。

**两种 DST 表述。**

- **分类（Classification）。** 对于每个（slot, candidate_value）对，预测 yes/no。适用于封闭词汇槽。2020 年前的标准做法。
- **生成（Generation）。** 给定对话，生成槽值为自由文本。适用于开放词汇槽。现代默认方法。

**指标。** 联合目标准确率（Joint Goal Accuracy，JGA）——在每个回合中*所有*槽都正确的比例。全有或全无。到 2026 年 MultiWOZ 2.4 排行榜顶端约为 83%。

**架构。**

1. **基于规则（槽正则 + 关键词）。** 在窄域上是强基线。易于调试。
2. **TripPy / BERT-DST。** 基于拷贝的生成，使用 BERT 编码。LLM 出现前的标准。
3. **LDST（LLaMA + LoRA）。** 指令微调的 LLM，结合域-槽提示。可在 MultiWOZ 2.4 上达到 ChatGPT 级别质量。
4. **无本体（2024–26）。** 跳过 schema，直接生成槽名和值。适用于开放域。
5. **提示 + 结构化输出（2024–26）。** 使用 LLM + Pydantic schema + 约束解码。几行代码即可上线生产。

### 经典失败模式

- **跨轮共指。** “我们就选第一个。” 需要解析“第一个”指哪个选项。
- **覆盖还是追加。** 用户说“加上意大利菜”。是替换 cuisine 还是追加？
- **隐式确认。** “好吧，行”——这是否接受了系统提出的预订？
- **纠正。** “其实改成 7 点。” 必须更新时间而不清除其他槽位。
- **对上一次系统话语的共指。** “是的，就是那个。” 指的哪个“那个”？

## 实作

### 步骤 1：基于规则的槽提取器

见 `code/main.py`。正则 + 同义词字典覆盖窄域中 70% 的规范话语：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在规范词汇之外非常脆弱。适用于确定性的槽确认。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 绝不重置用户没有触及的槽。
- 显式否定（“不用考虑菜系”）必须清除相应槽位。
- 用户纠正（“其实……”) 必须覆盖，而不是追加。

### 步骤 3：基于 LLM 的 DST（结构化输出）

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证返回一个有效的状态对象。无正则，也无 schema 不匹配，也不会幻读出不存在的槽。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在多少回合里把“所有槽”都做对？在 MultiWOZ 2.4 上，2026 年的顶尖系统为 80–83%。你的域内系统在窄词汇上应该超过这个基线，或让 LLM 基线超越你。

### 步骤 5：处理纠正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

在检测到纠正时，应覆盖最近更新的槽而不是追加。没有 LLM 辅助很难做到完美。现代模式：每轮允许 LLM 根据完整历史重新生成整个状态，而不是增量更新——这自然处理了纠正。

## 陷阱

- **完整历史重生成的成本。** 每轮让 LLM 重生成状态会导致总 token 成本为 O(n²)。对话历史做截断或对旧轮进行摘要。
- **Schema 漂移。** 事后增加新槽会破坏旧训练数据。对 schema 进行版本管理。
- **大小写敏感问题。** “Italian” vs “italian” vs “ITALIAN” —— 在所有环节做归一化。
- **隐式继承。** 如果用户之前指定了“4 人”，新的时间请求不应清除 people。始终传递完整历史。
- **自由形式 vs 封闭集合。** 名称、时间和地址需要自由形式槽；菜系和区域是封闭集合。在 schema 中混合使用两者。

## 使用指南

2026 年栈：

| Situation | Approach |
|-----------|----------|
| 窄域（一个或两个意图） | 基于规则 + 正则 |
| 广域、有标注数据 | LDST（LLaMA + LoRA 在 MultiWOZ 风格数据上微调） |
| 广域、无标注、可投入生产 | LLM + Instructor + Pydantic schema |
| 语音 / 语音交互 | ASR + 归一化器 + LLM-DST |
| 多域预订流程 | 以 schema 为指导的 LLM，按域使用 Pydantic 模型 |
| 合规敏感 | 以基于规则为主，LLM 为后备并辅以确认流程 |

## 部署（Ship It）

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: 设计一个对话状态跟踪器 — schema、提取器、更新策略、评估。
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

给定一个用例（域、语言、词汇开放性、合规需求），输出：

1. Schema。域列表、每个域的槽、每个槽是开放词汇还是封闭词汇。
2. 提取器。基于规则 / seq2seq / LLM + Pydantic。说明理由。
3. 更新策略。整表重生成 / 增量更新；纠正处理；否定处理。
4. 评估。在留出对话集上的 Joint Goal Accuracy，槽级别的精确率/召回率，最难槽的混淆矩阵。
5. 确认流程。在哪些情况下需要明确询问用户确认（破坏性操作、低置信度提取）。

对于合规敏感的槽，拒绝仅 LLM 的 DST，必须有基于规则的二次校验。拒绝任何在用户纠正时无法回滚槽位的 DST。标记没有版本标签的 schema。
```

## 练习

1. 简单：在 `code/main.py` 中为 3 个槽（cuisine、area、price）构建基于规则的状态跟踪器。用 10 条手工对话测试。测量 JGA。
2. 中级：用相同数据集，使用 Instructor + Pydantic + 一个小型 LLM。比较 JGA。检查最困难的回合。
3. 困难：实现两者并路由：以基于规则为主，当基于规则输出的槽少于 2 个且置信度低时回退到 LLM。测量合并后的 JGA 与每轮推理成本。

## 术语

| 术语 | 常说法 | 实际含义 |
|------|--------|---------|
| DST | Dialogue state tracking | 在对话回合间维护槽-值字典。 |
| Slot | Unit of user intent | 后端需要的命名参数（cuisine、date）。 |
| Domain | The task area | 餐厅、酒店、出租车 — 一组槽。 |
| JGA | Joint Goal Accuracy | 每个回合所有槽都正确的比例。全有或全无。 |
| MultiWOZ | The benchmark | 多域 WOZ 数据集；标准的 DST 评测基准。 |
| Ontology-free DST | No schema | 直接生成槽名和值，无固定列表。 |
| Correction | "Actually..." | 覆盖先前已填充槽的回合。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 经典基准。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — 用 LLaMA + LoRA 进行 DST 的指令微调研究。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 拷贝式 DST 的代表作。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于 EM 的无监督 TOD 研究。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — MultiWOZ 的官方结果与排行。