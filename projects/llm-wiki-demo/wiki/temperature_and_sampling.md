---
title: Temperature and Sampling
type: Concept
last_updated: 2026-07-04
sources:
  - "prompt_engineering_cn.md"
---
# Temperature and Sampling

### Core Mechanism
Temperature and Top-p are the primary parameters used to control the randomness and diversity of token generation in LLMs.
* **Temperature**: A scaling factor applied to the logit distribution before the softmax function. Mathematically, dividing logits by a lower temperature sharpens the distribution, making the highest-probability tokens much more likely to be chosen (approaching deterministic behavior at $T = 0$). A higher temperature flattens the distribution, increasing the probability of selecting less likely tokens.
* **Top-p (Nucleus Sampling)**: Limits token selection to the smallest subset of tokens whose cumulative probability exceeds the threshold $p$ (e.g., $p = 0.9$ means only the top 90% of the probability mass is considered, cutting off the long tail of highly unlikely tokens).

### Examples/Templates
| Setting | Temperature | Top-p | Use Case |
| :--- | :--- | :--- | :--- |
| **Deterministic** | 0.0 | 1.0 | Data extraction, classification, code generation |
| **Conservative** | 0.3 | 0.9 | Summarization, analysis, technical writing |
| **Balanced** | 0.7 | 0.95 | General Q&A, explanations |
| **Creative** | 1.0 | 1.0 | Brainstorming, creative writing, ideation |
| **Chaotic** | 1.5+ | 1.0 | Never use in production |

### Parameters/Caveats
* **Interaction Rule**: Adjust either Temperature or Top-p, but **never both simultaneously**, as they interact in unpredictable ways that degrade output control.

## Outgoing Links
- [[prompt_anatomy|Prompt Anatomy]]
- [[prompt_patterns|Prompt Patterns]]

## References
- [[index|Wiki Index]]
