# Capstone 05 — Autonomous Research Agent (AI-Scientist Class)

> Sakana 的 AI-Scientist-v2 发表了完整论文。Agent Laboratory 运行了实验。Allen AI 共享了痕迹。到 2026 年的形态是对实验进行基于计划-执行-验证的树搜索，受预算约束、沙箱化代码执行、视觉反馈的 LaTeX 撰写器，以及自动化的 NeurIPS 风格审稿人集合。这个 capstone 的目标是构建一个端到端系统，在每篇论文预算 $30 内运行完毕，并能通过 Sakana 记录的沙箱逃逸红队审查。

**Type:** 顶点项目  
**Languages:** Python（agent + sandbox）、LaTeX（输出）  
**Prerequisites:** Phase 2（机器学习），Phase 3（深度学习），Phase 7（Transformer 模型），Phase 10（从头训练的 LLMs），Phase 14（agents），Phase 15（自主），Phase 16（多智能体），Phase 18（安全）  
**Phases exercised:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18  
**Time:** 40 小时

## 问题

到 2026 年，自主研究智能体跨过了一个门槛。Sakana AI 的 AI-Scientist-v2 在 Nature 上发表了由其生成的论文并通过了研讨会同行评审。ShinkaEvolve（ICLR 2026）将该路线扩展为进化假设。AMD 的 Agent Laboratory 提供了可复现的痕迹。这些智能体并非魔法——它们是在候选实验树上运行的计划-执行-验证循环，有成本上限、受种子约束的沙箱和自动审稿。关键在于循环、预算与安全论证。

你通过针对一个小领域的初始想法实现该循环来学习（例如在一个 1 亿参数以下的 Transformer 上对注意力稀疏性进行消融实验）。价值不在于第一次运行就发现新东西，而在于基础设施：树搜索、实验沙箱、撰写-审稿循环、红队报告。Sakana 团队记录了沙箱逃逸失败；你的智能体必须通过相同的红队测试。

## 概念

该智能体是一个 best-first 树搜索。节点是实验规格：(假设, 配置, 代码, 预期结果)。一次扩展步骤提出带有小修改的子节点（更换优化器、调整批量大小、消融某个组件）。每个子节点在一个新的沙箱中运行并带有严格资源上限。结果反馈到一个评分函数，根据（新颖性 × 质量 × 剩余预算）对节点进行排序。树在预算耗尽前增长，然后将最佳分支写成论文。

撰写器是多模态的。它生成 LaTeX 草稿、编译、渲染图表，并将渲染后的 PDF 反馈给 Claude Opus 4.7 的视觉模式，以评估布局、图像可读性和论据-证据对齐。五个 LLM 审稿人组成的评审集合输出 NeurIPS 风格的评分（新颖性、严谨性、清晰度、可复现性、影响力）；若平均分低于阈值，论文将返回给撰写器并附带批评意见。

安全是承重部分。每个实验在 E2B 或 Daytona 沙箱中运行，无网络外连、绑定的墙钟时长和固定资源限制。智能体的代码生成步骤要通过策略层，屏蔽可能逃逸沙箱的系统调用。红队报告复现 Sakana 记录的攻击面（fork 炸弹、文件系统逃逸、由 LLM 编写的网络调用）。

## 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排：LangGraph，带检查点和人工审批门控  
- 树搜索：针对实验节点的自定义 best-first（沿用 Sakana v2 的 AB-MCTS 风格）  
- 沙箱：每次实验使用 E2B，回退为 Docker-in-Docker；通过 cgroups 强制资源上限  
- 文献：Semantic Scholar Graph API + OpenAlex + 本地 FAISS 摘要缓存  
- 撰写器：LaTeX 模板 + Claude Opus 4.7（视觉模式）用于图表批评和布局反馈  
- 审稿：五审组合（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），加权聚合  
- 实验框架：PyTorch 2.5 执行物理实验，W&B 用于日志记录  
- 可观测性：Langfuse 用于智能体痕迹，$30 的每篇硬预算

## 构建步骤

1. **种子与领域界定。** 取一个初始想法（例如 “研究 <1B Transformer 的注意力图的稀疏模式”）。定义搜索空间：模型、数据集、计算预算等。

2. **文献通读。** 使用 Semantic Scholar + OpenAlex 查询 50 篇最被引用的相关论文；本地缓存摘要；生成一页的领域摘要（domain digest）。

3. **树搭建。** 用种子假设初始化根节点。实现 `expand(node) -> children`，使用小编辑提案（每个子节点一个配置变更）。实现 `score(node)` 为加权的新颖性 × 质量 × 预算项。

4. **沙箱封装。** 每次实验运行如下命令或等效 E2B 策略：`docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`。随机种子写入沙箱；输出只读挂载返回主机。

5. **计划-执行-验证循环。** `plan` 提出子节点。`execute` 运行沙箱并捕获日志与指标。`verify` 在指标上运行单元检查（损失是否下降？消融是否隔离了效果？）。失败节点在树上存储失败原因。

6. **撰写器。** 预算耗尽后选择最佳分支。用 matplotlib 渲染图表。通过 Claude Opus 4.7 根据分支痕迹生成 LaTeX 草稿。编译。将编译后的 PDF 送回 Opus 4.7 的视觉模块进行批评。迭代修改。

7. **审稿集合。** 五位评审基于（新颖性、严谨性、清晰度、可复现性、影响力）的 NeurIPS 量表为草稿评分。若平均分 < 4.0/5，则将评论返回给撰写器并重写。最多强制停止在 3 次重写后。

8. **红队。** 构建或集成一组针对沙箱的对抗性任务：fork 炸弹、网络外排尝试、文件系统逃逸、由 LLM 编写的 shell 元字符与网络调用。确认所有尝试均被阻断。写出发现报告。

9. **可复现性。** 每篇论文附带树搜索痕迹 JSON、随机种子、W&B 运行链接、沙箱配置和一个可端到端复现的 README。

## 使用示例

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付物

`outputs/skill-ai-scientist.md` 为交付件。给定一个种子想法 + 领域 + $30 预算，它运行完整流水线并输出一份可审阅的论文与可复现包。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Paper quality | 盲评量表对比已发表研讨会论文 |
| 20 | Experimental rigor | 基线、随机种子、消融；每一条主张都由结果表中的一格支持 |
| 20 | Cost and compute discipline | 每篇 $30 上限强制执行，Langfuse 追踪 |
| 20 | Safety | 沙箱红队通过；网络策略与杀死开关经验证 |
| 15 | Reproducibility | 一条命令在相同随机种子下可复现整篇论文 |
| **100** | | |

## 练习

1. 针对同一领域的三个不同种子想法运行流水线。比较树搜索的哪些部分重叠。识别重复浪费的计算。

2. 在执行实验前为估计超过 $5 的节点增加人工审批门控。衡量总成本下降了多少。

3. 将审稿集合替换为单一评审。对已知的坏论文保留集测量误接收率（false-accept rate）。

4. 引入一个网络外排红队测试：让智能体写出尝试 `curl` 外部地址的代码。确认 `--network=none` 策略阻止了请求。记录该尝试。

5. 将你的树搜索与平坦随机基线比较（相同预算，无扩展策略）。报告新颖性 × 质量的提升。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Tree search | "AB-MCTS-style expansion" | 在实验节点上进行的基于新颖性×质量×预算评分的 best-first 探索 |
| Sandbox | "Experiment isolation" | 无网络、受限 CPU/内存、固定随机种子、只读输入的容器 |
| Vision critique | "Render-then-read" | 将论文编译为 PDF，然后将 PDF 送入视觉大模型（VLM）以进行布局和论据-证据批评 |
| Reviewer ensemble | "Automated peer review" | 多个 LLM 评审使用 NeurIPS 量表为论文打分；加权聚合决定管道是否放行 |
| Novelty score | "Is this new?" | 一种启发式，惩罚与 50 篇文献缓存的接近性 |
| Cost ceiling | "$ budget" | 每篇论文的硬上限；由 Langfuse 计数器和预运行估计强制执行 |
| Red team | "Sandbox-escape audit" | 如果策略错误会导致沙箱逃逸的一组对抗性测试 |

## 延伸阅读

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 参考的生产级研究智能体  
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — 原始方法学  
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — 进化扩展  
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色研究实验室框架  
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — 编排层参考文档  
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文献检索  
- [E2B sandboxes](https://e2b.dev) — 实验隔离参考  
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — 审稿集合所编码的量表