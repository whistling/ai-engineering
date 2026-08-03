---
title: Chain of Thought (CoT) (思维链)
type: Concept
last_updated: 2026-07-04
sources:
  - "few_shot_cot_cn.md"
---
# Chain of Thought (CoT) (思维链)

### Core Mechanism
Introduced by Wei et al. (2022), Chain of Thought (CoT) prompts the model to generate intermediate reasoning steps before outputting the final answer. Because Transformer models generate tokens sequentially, each generated reasoning token becomes part of the context for the next token, effectively expanding the computational depth of the model.

### GSM8K Benchmark Performance
| Model | Zero-Shot | Zero-Shot CoT | Few-Shot CoT |
|-------|-----------|---------------|--------------|
| GPT-4o | 78% | 91% | 95% |
| GPT-5 | 94% | 97% | 98% |
| o4-mini (Reasoning) | 97% | — | — |
| Claude Opus 4.7 | 93% | 97% | 98% |
| Gemini 3 Pro | 92% | 96% | 98% |
| Llama 4 70B | 80% | 89% | 94% |
| DeepSeek-V3.1 | 89% | 94% | 96% |

### Prompt Templates
* **Zero-Shot CoT**: Append `"让我们一步一步地思考"` (Let's think step by step) to the prompt.
* **Few-Shot CoT**: Provide examples that explicitly show the step-by-step reasoning process.

```python
def build_cot_prompt(question, examples, num_examples=3):
    system = (
        "您是一个数学问题求解器。 "
        "对于每个问题，请展示您的逐步推理，"
        "然后在最后一行以 '答案是 [数字]' 的格式给出最终的数值答案。"
    )
    example_text = ""
    for ex in examples[:num_examples]:
        example_text += f"Q: {ex['question']}\n"
        example_text += f"A: {ex['reasoning']} 答案是 {ex['answer']}.\n\n"
    user = f"{example_text}Q: {question}\nA:"
    return system, user
```

### Parameters & Caveats
* **Token Overhead**: Adds 50-200 tokens per query.
* **Avoid For**: Simple factual recall, single-step classification, or high-throughput, low-complexity tasks where speed is critical.
* **Reasoning Models**: Models like OpenAI's o-series and DeepSeek-R1 have internal CoT; adding explicit CoT prompts is redundant and can be counterproductive.

## Outgoing Links
- [[few-shot_prompting_少样本提示|Few-Shot Prompting (少样本提示)]]
- [[self-consistency_自洽性|Self-Consistency (自洽性)]]
- [[tree_of_thoughts_tot_思维树|Tree of Thoughts (ToT) (思维树)]]
- [[react|ReAct]]

## References
- [[index|Wiki Index]]
