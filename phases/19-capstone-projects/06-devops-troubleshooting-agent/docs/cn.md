# Capstone 06 — 用于 Kubernetes 的 DevOps 故障排查代理

> AWS 的 DevOps Agent 已进入 GA，Resolve AI 发布了其 K8s 操作手册，NeuBird 演示了语义监控，Metoro 将 AI SRE 与每服务 SLO 绑定。生产形态已定：告警 webhook 触发，代理读取遥测，遍历 K8s 对象图，排序根因假设，并在 Slack 发布摘要附带审批按钮。默认只读。任何修复动作都需人工审批。本结业项目即该代理，使用 20 个合成事故进行评估，并在三个共享用例上与 AWS 的 Agent 对比。

**Type:** 结业项目  
**Languages:** Python (agent), TypeScript (Slack 集成)  
**Prerequisites:** Phase 11 (LLM 工程), Phase 13 (工具 与 MCP), Phase 14 (agents), Phase 15 (自主), Phase 17 (基础设施), Phase 18 (安全)  
**Phases exercised:** P11 · P13 · P14 · P15 · P17 · P18  
**Time:** 30 小时

## 问题

2025–2026 年的 SRE 叙事变成：“AI 代理进行事故分流，人类审批修复。”AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产环境中交付这种形态。代理读取 Prometheus 指标、Loki 日志、Tempo 跟踪、kube-state-metrics，以及 K8s 对象的知识图谱。它在五分钟内生成带有遥测引用的排序根因假设。未经 Slack 的明确人工审批，绝不执行破坏性命令。

大部分难点在于范围与安全，而不是推理。代理需要默认只读的 RBAC 表面、加固的 MCP 工具服务器，以及每一个“考虑过”的命令与“实际执行”的审计日志。它需要知道何时超出能力范围并上报。此外它要足够廉价运行，避免 OOM-kill 级联产生高昂费用。

## 概念

代理在知识图上运行。节点是 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）以及遥测源（Prometheus 序列、Loki 流、Tempo 跟踪）。边表示所有权（Pod -> ReplicaSet -> Deployment）、调度（Pod -> Node）和观测（Pod -> Prometheus 序列）。图由 kube-state-metrics 同步保持新鲜，并在每次告警时重新采样。

当告警触发时，代理从受影响对象开始根因推断。它遍历边，拉取相关的遥测切片（最近 15 分钟），并起草一个假设。假设按证据排序：多少遥测引用支持它、多新近、多具体。排名前三的假设发送到 Slack，附带图路径可视化和用于修复操作的审批按钮。

修复操作受限。默认允许的操作为只读。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 审批；ArgoCD 回滚钩子需要代理永远不持有的认证令牌。审计日志记录代理“考虑过”的每一条命令 —— 不仅是执行过的 —— 以便复核过程能捕捉近失误。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI 接收器
           |
           v
   LangGraph 根因代理
           |
           +---- 只读 MCP 工具 ----+
           |                      |
           v                      v
   K8s 知识图                   遥测切片
     (Neo4j / kuzu)            Prometheus, Loki, Tempo
   所有权 + 调度                 最近 15 分钟，范围限定
           |
           v
   假设排序（证据权重）
           |
           v
   Slack 摘要 + 审批按钮
           |
           v (获批)
   ArgoCD 回滚钩子 / PagerDuty 升级
           |
           v
   审计日志：考虑过 vs 执行过的每条命令
```

## 技术栈

- 可观测性源：Prometheus、Loki、Tempo、kube-state-metrics  
- 知识图：托管 Neo4j 或 嵌入式 kuzu，表示 K8s 对象 + 遥测边  
- 代理：LangGraph，配以每个工具的 allow-list，默认只读  
- 工具传输：基于 StreamableHTTP 的 FastMCP；破坏性工具放在单独服务器并受审批门控  
- 模型：Claude Sonnet 4.7 用于根因推理，Gemini 2.5 Flash 用于日志摘要  
- 修复：ArgoCD 回滚 webhook、PagerDuty 升级、Slack 审批卡片  
- 审计：追加式结构化日志（considered、executed、approved、outcome）  
- 部署：K8s Deployment，具有严格窄化的 RBAC 角色；单独命名空间

## 实现步骤

1. Graph ingestion（图摄取）。每 30s 将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测覆盖边：OBSERVED_BY（一个 Pod 被某个 Prometheus 序列观测到）。

2. Alert receiver（告警接收器）。FastAPI 端点，接受 PagerDuty 或 Alertmanager webhook。提取受影响对象和 SLO 违规信息。

3. Read-only tool surface（只读工具面）。通过 FastMCP 封装 kubectl、Prometheus query、Loki logql、Tempo traceql。每个工具有窄化的 RBAC 动词（"get"、"list"、"describe"）。默认服务器不提供 "delete"、"exec"、"scale" 等。

4. Root-cause agent（根因代理）。LangGraph 包含三个节点：`sample` 拉取最近 15 分钟的遥测切片，`walk` 查询图中邻居对象，`hypothesize` 起草带遥测引用的排序根因候选。

5. Evidence scoring（证据评分）。每个假设得分 = 新近性 * 具体性 *（图路径长度的倒数）* 引用数。返回前 3。

6. Slack brief（Slack 摘要）。发布包含假设、图路径可视化（服务器端渲染的子图图片）和最多一个修复动作的审批按钮的附件。

7. Remediation gate（修复门控）。破坏性工具（缩容、回滚、删除）驻留在第二个 MCP 服务器后端并由审批令牌保护。代理仅在 Slack 卡片被人工批准后调用它们。

8. Audit log（审计日志）。追加式 JSONL：对于每个候选命令，记录是否被考虑、是否被执行、谁批准。每天上传至 S3。

9. Synthetic incident suite（合成事故套件）。构建 20 个场景：OOMKill 级联、DNS 波动、HPA 抖动、PVC 满、吵闹邻居、错误 sidecar、错误的 ConfigMap 发布、证书轮转、image-pull backoff 等。在根因准确度和生成假设的时间上评分代理。

## 使用示例

```
webhook: alert.pagerduty.com -> checkout-api SLO 违规，错误率 14%
[graph]   受影响：Deployment checkout-api（3 个 Pod，节点 ip-10-2-3-4）
[walk]    邻居：ReplicaSet checkout-api-abc，Service checkout-api，
           最近一次 rollout 14 分钟前
[sample]  prometheus error_rate 14%，上升趋势；loki /api/v2/pay 出现 500
[hypo]    #1 错误发布：最新镜像 checkout-api:v2.41 导致 /healthz 失败
          引用：deploy.yaml（rev 42）、prometheus errorRate、loki 500 堆栈
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          （需要审批；代理不会单方面回滚）
```

## 交付物

`outputs/skill-devops-agent.md` 是最终交付物。给定一个 K8s 集群和告警源，代理应生成排序的根因假设并提供 Slack 门控的修复流程。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | RCA accuracy on scenario suite | ≥80% 在 20 个合成事故中识别正确根因 |
| 20 | Safety | 在审计日志中，若未通过 Slack 审批，破坏性操作绝不触发 |
| 20 | Time-to-hypothesis | p50 从告警到 Slack 摘要在 5 分钟内 |
| 20 | Explainability | 每个假设具有图路径和遥测引用 |
| 15 | Integration completeness | PagerDuty、Slack、ArgoCD、Prometheus 全链路可用 |
| **100** | | |

## 练习

1. 在与 AWS 的 DevOps Agent 演示相同的三个事故上运行你的代理。发布并排对比。报告代理差异所在。

2. 添加“近失误”审计：标记任何代理“考虑过”的、在没有审批时将是破坏性的命令。统计一周内的近失误率。

3. 将假设模型从 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量 RCA 准确度差异和每次事故的成本变化（美元）。

4. 构建因果过滤器：区分相关的遥测突增与真实根因。基于 20 个场景标签训练一个小分类器。

5. 添加回滚 dry-run：在具有相同清单的预演集群上对 ArgoCD 回滚进行演练。在 Slack 审批按钮之前，在真实集群中验证回滚计划。

## 术语表

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| K8s knowledge graph | "Cluster graph" | 节点 = K8s 对象 + 遥测序列；边 = 所有权、调度、观测 |
| Read-only-by-default | "Scoped RBAC" | 代理的 service account 仅有 get/list/describe 动词；破坏性动词在单独服务器并由审批保护 |
| Audit log | "Considered vs executed" | 追加式记录每个候选命令，是否运行，谁批准 |
| Hypothesis ranking | "Evidence score" | 新近性 × 具体性 ×（图路径长度的倒数）× 引用数 |
| Slack approval card | "HITL gate" | 带修复按钮的交互式 Slack 消息；人工点击前代理无法继续 |
| Telemetry citation | "Evidence pointer" | 支持某项断言的 Prometheus 查询、Loki 选择器或 Tempo 跟踪 URL |
| MTTR | "Time to resolution" | 从告警触发到 SLO 恢复的实时时间 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年权威参考  
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品参考  
- [NeuBird semantic monitoring](https://www.neubird.ai) — 语义图方法  
- [Metoro AI SRE](https://metoro.io) — 以 SLO 为中心的生产框架  
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态来源  
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考的代理编排器  
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架  
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 门控修复目标