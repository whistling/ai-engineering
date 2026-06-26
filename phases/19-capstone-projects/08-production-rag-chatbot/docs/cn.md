# Capstone 08 — 针对受监管行业的生产级 RAG 聊天机器人

> 到 2026 年，Harvey、Glean、Mendable 和 LlamaCloud 都在运行相同的生产形态。使用 docling 或 Unstructured 进行文档摄取，ColPali 处理视觉内容。混合检索。使用 bge-reranker-v2-gemma 重新排序。使用 Claude Sonnet 4.7 合成并启用提示词缓存，命中率目标 60–80%。使用 Llama Guard 4 和 NeMo Guardrails 进行护栏防护。使用 Langfuse 和 Phoenix 进行监控。用 200 题黄金集通过 RAGAS 打分。在受监管领域（法律、临床、保险）构建一个系统，capstone 的目标是通过黄金集、红队评估和漂移仪表盘。

**Type:** 结业项目  
**Languages:** Python (pipeline + API), TypeScript (chat UI)  
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure), Phase 18 (safety)  
**Phases exercised:** P5 · P7 · P11 · P12 · P17 · P18  
**Time:** 30 小时

## 问题

受监管领域的 RAG（法律合同、临床试验方案、保险条款）是 2026 年最常见的生产形态，因为投资回报明显且风险具体。Harvey（Allen & Overy）为法律构建了该形态。Mendable 提供面向开发者文档的产品化版本。Glean 覆盖企业搜索。模式是：高保真摄取；混合检索并重排序；在合成时强制引用并启用提示词缓存；多层护栏防护；持续监控漂移。

难点不在于模型本身，而在于合规（HIPAA、GDPR、SOC2）感知、逐条引用的审计能力、成本控制（提示词缓存高命中率能带来 60–90% 折扣）、通过 RAGAS 检测幻觉的能力，以及当源文档更新但索引未及时同步时的漂移检测。本 capstone 要求你在 200 题黄金集和红队套件下交付完整系统。

## 概念

流水线分为两端。摄取端：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的文档；分块同时生成摘要、标签与基于角色的访问标签。向量存入 pgvector + pgvectorscale（在 5000 万向量以下）或 Qdrant Cloud；并行运行稀疏 BM25。会话端：LangGraph 处理记忆与多轮对话；每次查询执行混合检索、用角色与司法管辖过滤、用 bge-reranker-v2-gemma-2b 重排序、用 Claude Sonnet 4.7（启用提示词缓存）合成、输出通过 Llama Guard 4 与 NeMo Guardrails，最终返回带引用的回答。

评估栈有四层。黄金集（200 个带引用标注的问答）用于正确性检验。红队（越狱、PII 提取尝试、越域问题）用于安全性。RAGAS 用于每轮的可信度/答案相关性/上下文精确度自动评估。漂移仪表盘（Arize Phoenix）监控检索质量和幻觉评分的周度变化。

提示词缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示 + 检索到的上下文。在 60–80% 命中率下，每次查询成本下降 3–5 倍。流水线需为稳定前缀（系统提示 + 重排序后的上下文优先）设计，以达到高缓存命中率。

## 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 摄取：Unstructured.io 或 docling 用于结构化文档；ColPali 用于视觉丰富的 PDF  
- 向量数据库：pgvector + pgvectorscale（在 5000 万向量以下）；否则使用 Qdrant Cloud  
- 稀疏检索：Tantivy 的 BM25，支持字段权重  
- 编排：LlamaIndex Workflows（摄取） + LangGraph（会话）  
- 重新排序器：bge-reranker-v2-gemma-2b 自托管，或 Voyage rerank-2 托管服务  
- 大模型：Claude Sonnet 4.7（启用提示词缓存）；回退模型为自托管的 Llama 3.3 70B  
- 评估：RAGAS 0.2 在线，DeepEval 用于幻觉与越狱测试套件  
- 可观测性：Langfuse 自托管并带注释队列；Arize Phoenix 用于漂移监控  
- 护栏：Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略框架，Presidio PII 清洗  
- 合规：分块上打角色访问标签与司法管辖标签，检索时强制执行

```figure
canary-rollout
```

## 构建步骤

1. **摄取。** 使用 Unstructured 或 docling 解析语料（严肃构建建议 1000–10000 文档）。对于扫描件/视觉密集页，走 ColPali 流程。输出分块并生成摘要、角色标签、司法管辖标签。

2. **索引。** 使用密集嵌入（Voyage-3 或 Nomic-embed-v2）插入 pgvector + pgvectorscale。通过 Tantivy 建立 BM25 旁索引。将角色与司法管辖作为 payload 存储以便过滤。

3. **混合检索。** 先按角色+司法管辖过滤；然后并行执行密集检索与 BM25；用互惠秩融合（reciprocal rank fusion）合并结果；将 top-20 送入重排序器；重排序后 top-5 进入合成阶段。

4. **启用提示词缓存进行合成。** 将系统提示 + 静态策略放在缓存前缀；将重排序后的上下文作为缓存扩展；用户问题作为不可缓存的后缀。稳态下目标命中率为 60–80%。

5. **护栏。** 在输入端运行 Llama Guard 4；NeMo Guardrails 拦截越域或策略禁止的话题；输出端使用 Presidio 清理意外泄露的 PII；最后进行引用强制检查。

6. **黄金集。** 由领域专家标注 200 对问答（含答案与引用）。按精确引用匹配、答案正确性、可信度（RAGAS）对 agent 评分。

7. **红队。** 准备 50 条对抗性提示：越狱（PAIR、TAP）、PII 外泄尝试、越域、跨司法管辖泄露。按通过/未通过与严重程度评分。

8. **漂移仪表盘。** Arize Phoenix 跟踪检索质量（nDCG、引用可信度）的周度变化。跌幅超过 5% 触发告警。

9. **成本报告。** Langfuse 报告提示词缓存命中率、每次查询的平均 token 数、按阶段的 $/query 明细。

## 使用示例

```
$ chat --role=analyst --jurisdiction=GDPR
> 我们合同下欧盟用户资料的数据保留义务是什么？
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

（注：命令行标志与引用保持原样，问题与上下文提示已翻译以示例用途。）

## 上线交付物

`outputs/skill-production-rag.md` 描述了交付物。受监管领域的聊天机器人需带合规标签，通过评分规则，并配备实时漂移监控。

| 权重 | 评估项 | 测量方法 |
|:-:|---|---|
| 25 | RAGAS 可信度 + 答案相关性 | 在黄金集（200 Q/A）上的在线得分 |
| 20 | 引用正确性 | 可验证源锚点的答案比例 |
| 20 | 护栏覆盖率 | Llama Guard 4 的通过率 + 越狱测试套件结果 |
| 20 | 成本 / 延迟工程 | 提示词缓存命中率、p95 延迟、每次查询成本 |
| 15 | 漂移监控仪表盘 | Phoenix 实时仪表盘与周度检索质量趋势 |
| **100** | | |

## 练习

1. 构建第二个司法管辖切片（例如在 GDPR 之外再加一个 HIPAA 切片）。演示角色+司法管辖过滤在 20 题跨域探测中阻止交叉泄露。  
2. 在一周生产流量中测量提示词缓存命中率。识别哪些查询打破了缓存前缀，并对缓存结构进行重构。  
3. 添加多轮记忆，使用 10k-token 的摘要缓冲区。测量会话增长时可信度是否下降。  
4. 将 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。比较每次查询成本与可信度差异。  
5. 添加“不确定”模式：当重排序得分低于阈值时，代理回复“我没有可靠的引用”而不是给出答案。衡量错误自信的减少量。

## 术语表

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| 提示词缓存 (Prompt caching) | "Cached system + context" | Claude/OpenAI 特性：命中时前缀 token 有 60–90% 的折扣 |
| RAGAS | "RAG evaluator" | 对可信度、答案相关性、上下文精确度的自动评分 |
| 黄金集 | "Labeled eval" | 200+ 由专家标注并带引用的问答；作为基准真值 |
| 司法管辖标签 | "Compliance label" | 附加到分块的 GDPR/HIPAA/SOC2 范畴；检索时强制过滤 |
| 引用可信度 | "Grounded answer rate" | 有可检索源片段支撑的声明比例 |
| 漂移 | "Retrieval quality decay" | nDCG 或引用得分的周度变化；告警阈值通常为 5% |
| 混合检索 (hybrid search) | — | 稠密嵌入 + 稀疏 BM25 并行检索并融合结果 |
| 嵌入 (Embeddings) | — | 将文本映射到向量空间以做近似检索 |
| 微调 (Fine-tuning) | — | 在下游任务上继续训练模型以提升性能 |
| 上下文窗口 (Context window) | — | 模型可处理的最大 token 长度 |
| 少样本 (few-shot) | — | 在少量示例下进行任务指示 |
| 思维链 (chain-of-thought) | — | 模型显式生成中间推理步骤来提升复杂推理准确率 |
| 护栏 (guardrails) | — | 输入/输出的策略与分类器，用于约束模型行为 |
| 函数调用 (function calling) | — | 模型将结构化指令映射到函数/API 调用的能力 |
| 投机性解码 (speculative decoding) | — | 为加速生成而使用的预测性解码策略 |
| 位置嵌入 (positional embeddings) | — | 表示序列中 token 位置的向量 |
| 自注意力 (self-attention) | — | Transformer 的核心机制，用于建模序列内依赖 |
| 指令微调 (instruction tuning) | — | 用指令式数据对模型进行微调以提高执行指令的能力 |
| 分布式训练 (distributed training) | — | 在多个设备/节点上并行训练大模型 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 法律生产栈参考  
- [Glean enterprise search](https://www.glean.com) — 企业级 RAG 参考  
- [Mendable documentation](https://mendable.ai) — 面向开发者文档的 RAG 参考  
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管摄取示例  
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考  
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — 规范化的 RAG 评估框架  
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 漂移可观测性参考  
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年的安全分类器参考  
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略护栏框架