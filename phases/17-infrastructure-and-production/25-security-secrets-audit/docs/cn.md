# Security — Secrets, API Key Rotation, Audit Logs, Guardrails

> 通过集中化的密钥库（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥蔓延。绝不要在 VCS 的配置文件、环境文件或电子表格中存储凭证。优先使用 IAM 角色而非静态密钥；CI/CD 使用 OIDC。AI-gateway 模式是 2026 年的解决方案：apps → gateway → model provider，网关在运行时从 vault 拉取凭证。在 vault 中轮换，所有应用在几分钟内自动使用新密钥——无需重新部署，也无需在 Slack 上询问“谁拿到新密钥”。轮换策略 ≤90 天；在每次提交时用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任：MFA、SSO、RBAC/ABAC、短期令牌、设备安全态势。PII 清洗在转发前用实体识别屏蔽 PHI/PII；一致性标记化（Mesh 方法）将敏感值映射到稳定占位符，使 LLM 保持代码/关系语义不变。网络出口：将 LLM 服务放在专用 VPC/VNet 子网，仅白名单 `api.openai.com`、`api.anthropic.com` 等；阻止所有其它出站。2026 年的事件推动因素：Vercel 供应链攻击通过被泄露的 CI/CD 凭证窃取了数千个客户部署的环境变量。

**Type:** 学习  
**Languages:** Python（标准库，示例 PII 清洗器 + 审计日志写入器）  
**Prerequisites:** 第17阶段 · 19（AI 网关）、第17阶段 · 13（可观测性）  
**Time:** ~60 分钟

## Learning Objectives

- 列举四种密钥管理反模式（在 VCS 的配置文件、硬编码环境变量、电子表格、静态密钥）并指出它们的替代方案。
- 解释 AI-gateway 从 vault 拉取凭证的模式为何成为 2026 年的生产标准。
- 实现一个具有一致性标记化（相同值 → 相同占位符）功能的 PII 清洗器，以保留语义关系。
- 说明 2026 年 Vercel 供应链事件及其对 CI/CD 凭证卫生的教训。

## The Problem

一名实习生提交了包含 API 密钥的 `.env` 文件。虽然他们很快删除了该文件，但密钥已进入 git 历史——GitGuardian 扫描发现后，你们的轮换流程是“在 Slack 上通知团队，更新 40 个配置文件，重新部署所有服务”。8 小时后，一半服务已上线，另一半仍在等待部署窗口。

另外，用户提示包含 “My SSN is 123-45-6789.”。提示被转发到 OpenAI。虽然你们有 BAA（业务伙伴协议），但内部策略要求在转发前屏蔽 PII。你们没有这么做。

再者，你们的 EKS 集群中的 LLM pod 可以访问任意互联网主机。有人通过对攻击者控制的域发起 DNS 查询来窃取数据。没有任何阻止措施。

LLM 服务的安全必须覆盖这三条向量：基于 vault 的凭证、PII 清洗、网络出口过滤、审计日志。

## The Concept

### Centralized vault + IAM-role pull

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。作为单一可信源。

**IAM role**：应用/网关通过其 IAM 身份进行认证，而不是使用静态密钥。Vault 返回具有生存期的短期凭证。

**AI-gateway 模式**：网关在请求时从 vault 拉取 `OPENAI_API_KEY`。在 vault 中轮换；下一次请求会使用新密钥。无需重新部署。

### Rotation policy ≤ 90 days

所有 API 密钥、vault root token、CI/CD 凭证都应 ≤ 90 天轮换。尽可能实现自动轮换。手动轮换需记录并跟踪。

### Secret scanning

- **TruffleHog** — 基于正则和熵的提交扫描。  
- **GitGuardian** — 商业产品，高准确率。  
- **Gitleaks** — 开源，作为 CI 中的扫描工具。

在每次提交时运行。若检测到新密钥则阻止 PR。

### Zero-trust posture

- 所有账号必须启用 MFA。  
- 通过 SAML/OIDC 做 SSO。  
- 使用 RBAC（基于角色）或 ABAC（基于属性）实现精细化访问控制。  
- 使用短期令牌（以小时计，而不是天）。  
- 设备安全态势检查——仅允许公司设备且启用磁盘加密。

### PII / PHI scrubbing

在提示离开你们基础设施前：

1. 实体识别（spaCy NER、Presidio、或商业方案）。  
2. 屏蔽匹配到的实体：将 `"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。  
3. 一致性标记化（Mesh 方法）：相同值映射到相同占位符，以使 LLM 能保留关系语义。  
4. 可选：对 LLM 响应进行反向映射以恢复真实值。

静态正则用于捕捉基本模式；NER 捕获更多上下文。两者兼用。

### Input + output guardrails

输入：阻止已知越狱、禁止主题；对单用户进行速率限制。  
输出：对泄露密钥（API key 模式、电子邮件模式等）进行正则清洗；用分类器检测策略违规。

### Network egress whitelist

将 LLM 服务放置在专用子网：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、vault 端点。  
- 其它所有目标：丢弃。  
- DNS 使用只允许名单的解析器（以防止 DNS 隧道数据外泄）。

### Audit log

每次 LLM 调用的不可变日志，包含：
- 时间戳。  
- 用户 / 租户。  
- 提示哈希（为隐私起见不存原始提示）。  
- 模型 + 版本。  
- Token 计数。  
- 成本。  
- 响应哈希。  
- 任何触发的护栏事件。

根据合规要求保留（SOC 2 = 1 年，HIPAA = 6 年）。

### The 2026 Vercel incident

供应链攻击：被窃取的 CI/CD 凭证导致数千个客户部署的环境变量被外泄。教训：CI/CD 凭证等同于生产凭证。将其存储在 vault 中，限制作用域，并积极轮换。

### Numbers you should remember

- 轮换策略：≤ 90 天。  
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。  
- Vercel 2026：CI/CD 凭证被入侵 → 数千个客户环境变量泄露。  
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

## Use It

`code/main.py` 实现了一个带有一致性标记化和追加式审计日志的示例 PII 清洗器。

## Ship It

本课产出 `outputs/skill-llm-security-plan.md`。根据合规范围和当前状态，规划 vault 迁移、清洗器、出口策略和审计日志。

## Exercises

1. 运行 `code/main.py`。发送两个引用相同 SSN 的提示。确认两次都映射为相同的占位符。  
2. 为一个在 EKS 上运行的 vLLM 设计网络出口策略，该部署需要调用 OpenAI、Anthropic 和 Weaviate。  
3. 你在 git 历史中发现了一个两年前的密钥。正确的响应是什么——轮换密钥、擦除历史，还是两者都做？说明理由。  
4. 你的审计日志每天增长 10 GB。设计分级保留策略（hot 30 天、warm 12 个月、cold 6 年）。  
5. 论证是否值得实现反向标记化（在 LLM 响应中替换回真实值），相对于保持占位符可见，两者的复杂性与风险权衡。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Vault | "secrets store" | 集中化的凭证管理服务 |
| IAM role | "identity-based auth" | 应用假定的角色；返回短期凭证 |
| OIDC for CI/CD | "cloud-issued tokens" | CI 中不使用静态密钥——通过 OIDC 获取身份令牌 |
| TruffleHog / GitGuardian / Gitleaks | "secret scanners" | 提交时的密钥检测工具 |
| RBAC / ABAC | "access control" | 基于角色 vs 基于属性 的访问控制 |
| PII scrubbing | "data masking" | 删除或对敏感实体进行标记化 |
| Consistent tokenization | "stable placeholders" | 相同值 → 每次相同的占位符 |
| Mesh approach | "Mesh tokenization" | 保持语义的标记化模式 |
| Egress whitelist | "outbound allowlist" | 仅允许的域名可达 |
| Audit log | "immutable history" | 追加式不可变合规记录 |

## Further Reading

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)  
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)  
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)  
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)  
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII 检测与匿名化。  
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)