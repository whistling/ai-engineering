---
title: DSPy
type: Technology
last_updated: 2026-07-04
sources:
  - "few_shot_cot_cn.md"
---
# DSPy

### Core Mechanism
DSPy treats prompting as an optimization problem rather than manual string manipulation. It compiles declarative signatures into optimized prompting pipelines.

### Code Example
```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o", temperature=0.7))

class MathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.solve(question=question)

solver = MathSolver()
# Self-consistency via majority vote
result = dspy.majority(
    [solver(question="Janet 的鸭子每天下 16 个蛋...") for _ in range(5)],
    field="answer"
)
```

## Outgoing Links
- [[chain_of_thought_cot_思维链|Chain of Thought (CoT) (思维链)]]
- [[self-consistency_自洽性|Self-Consistency (自洽性)]]

## References
- [[index|Wiki Index]]
