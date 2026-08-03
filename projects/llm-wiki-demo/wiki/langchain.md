---
title: LangChain
type: Technology
last_updated: 2026-07-04
sources:
  - "few_shot_cot_cn.md"
---
# LangChain

### Core Mechanism
An orchestration framework that provides built-in templates and selectors to manage complex prompting patterns like Few-Shot and Chain of Thought.

### Code Example
```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

example_prompt = PromptTemplate(
    input_variables=["question", "reasoning", "answer"],
    template="Q: {question}\nA: {reasoning} 答案是 {answer}."
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Q: {input}\nA: 让我们一步一步地思考。",
    input_variables=["input"]
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
chain = few_shot_prompt | llm
```

## Outgoing Links
- [[few-shot_prompting_少样本提示|Few-Shot Prompting (少样本提示)]]
- [[chain_of_thought_cot_思维链|Chain of Thought (CoT) (思维链)]]

## References
- [[index|Wiki Index]]
