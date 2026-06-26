# Multimodal Agents and Computer-Use (Capstone)

> 2026 年的前沿产品是一个多模态代理，能够读取截图、点击按钮、导航 Web UI、填写表单并端到端完成工作流。SeeClick 和 CogAgent（2024）验证了 GUI 绑定原语。Ferret-UI 增加了移动端支持。ChartAgent 引入了针对图表的视觉工具使用。VisualWebArena 和 AgentVista（2026）是前沿模型追逐的基准——即便是 Gemini 3 Pro 和 Claude Opus 4.7 在 AgentVista 的难题上也仅得 ~30%。本结业项目汇聚了 Phase 12 的各条线：感知（高分辨率 VLM）、推理（带工具调用的 LLM）、定位（坐标输出）、长时记忆与评估。

**Type:** 结业项目  
**Languages:** Python（stdlib，action schema + agent loop skeleton）  
**Prerequisites:** Phase 12 · 05（LLaVA）、Phase 12 · 09（Qwen-VL JSON）、Phase 14（Agent Engineering）  
**Time:** ~240 分钟

## 学习目标

- 设计一个多模态代理循环：感知 → 推理 → 执行 → 观察 → 重复。
- 构建一个 GUI 定位输出 schema（点击坐标、输入文本、滚动、拖拽），使 VLM 能以 JSON 格式输出。
- 比较仅截图代理、可访问性树代理与混合代理的表现。
- 在一个小规模的 VisualWebArena 切片上搭建多模态代理基准评估。

## 问题描述

一个订票网站的工作流： “帮我找一趟 4 月 15 日飞东京的航班，靠过道座位，价格低于 800 美元，并帮我预订。”

一个多模态代理需要：

1. 获取浏览器截图。
2. 将截图 + URL + 目标解析为计划。
3. 输出结构化动作：点击（在 x,y）、输入 “Tokyo”（在元素 E）、向下滚动、选择（单选按钮）。
4. 将动作应用到浏览器。
5. 观察新状态（下一张截图）。
6. 重复，直到任务完成。

每一步都是一次多模态 VLM 调用。VLM 的输出必须是可解析的 JSON。错误会在步骤间累积，因此恢复能力很重要。

## 概念介绍

### GUI 定位 — 原语

GUI 定位是：给定一个截图和自然语言指令，输出要点击的（x, y）坐标（或其它动作）。

SeeClick（arXiv:2401.10935）是第一个大规模开源成果：在合成与真实 GUI 数据上微调 VLM，将坐标作为纯文本 token 输出。可行。

CogAgent（arXiv:2312.08914）引入了 1120x1120 的高分辨率编码以处理密集 UI。成绩：Web 导航约 84%。

Ferret-UI（arXiv:2404.05719）聚焦移动端 UI，并集成了 iOS 的可访问性数据。

输出格式通常是 JSON：

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` 有助于恢复：当截图间坐标漂移时，语义提示允许系统重新定位。

### 动作 schema

典型的动作 schema 包含 6–10 种动作类型：

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

代理每步发出一个动作。浏览器封装执行该动作并返回新状态。

### 仅截图 vs 可访问性树

两种输入模式：

- 仅截图：只有完整图像，无结构信息。最通用；适用于任意应用。
- 可访问性树：结构化的 DOM / iOS 可访问性信息。用于定位更可靠；适用于可提供树的场景。
- 混合：两者兼备，使用树来作为原子动作的可靠落点，用截图提供语义上下文。

生产环境的代理在可行时会使用混合模式。浏览器自动化（Selenium + accessibility）通常有可访问性树；桌面应用有时没有。

### 长时记忆

一个 20 步的工作流会产生 20 张截图。VLM 的上下文窗口很快被耗尽。三种压缩策略：

- Summary-chain：每 5 步摘要一次，丢弃旧截图。
- Skip-frame：保留首帧、末帧和每第 3 帧。
- 工具记录日志：执行动作并保留文本日志；不再回看旧截图。

Claude 的 computer-use API 使用日志模式。更简单、更可靠。

### 视觉工具使用

ChartAgent（arXiv:2510.04514）引入了针对图表的视觉工具使用：裁剪、放大、OCR、调用外部检测器。代理可以输出 “裁剪到区域 (100, 200, 300, 400) 然后调用 OCR” 作为一个工具调用。工具返回文本；VLM 继续推理。

该模式可泛化：集合提示（set-of-mark prompting）、区域标注与外部检测工具都符合 “输出一个工具调用，接收结构化响应” 的模式。

### 2026 年的基准

- ScreenSpot-Pro：针对 ~1k 网页截图的 GUI 定位。开源 SOTA Qwen2.5-VL-72B ~85%。前沿模型 ~90%。
- VisualWebArena：端到端网页任务（购物、论坛、分类信息）。开源 SOTA ~20%。Gemini 3 Pro ~27%。
- AgentVista（arXiv:2602.23166）：2026 年最难的基准。12 个领域的真实工作流。前沿模型得分 27–40%；开源模型 10–20%。
- WebArena / WebShop：较早的基准，被前沿模型充分覆盖。

### 为什么仍然困难

代理性能瓶颈：

1. 细粒度的视觉定位。在移动分辨率上，“点击小 X”经常失败。
2. 长时规划。执行 10 步后，代理常偏离目标。
3. 错误恢复。当点击失败（点错按钮）时，检测并恢复的训练样本稀缺。
4. 跨页面上下文。在标签间切换或长表单时会丢失状态。

研究方向：记忆架构、显式重规划、多模态验证（通过截图匹配判断动作成功）。

### 结业项目构建目标

结业项目任务：构建一个电脑使用代理，该代理需：

1. 读取订票网站模拟页面的 HTML + 截图。
2. 规划多步序列：搜索 → 选择 → 填表 → 提交。
3. 输出与动作 schema 匹配的 JSON 动作。
4. 在固定的 10 个任务切片上评估。

课程提供易于扩展为真实浏览器的脚手架代码。

## 使用方法

`code/main.py` 是结业项目的脚手架：

- 动作 schema 的 JSON 定义（10 个动作）。
- 作为字典的模拟浏览器状态。
- 代理循环骨架：接收状态、输出动作、应用动作、循环。
- 10 个任务的小型基准（合成页面），用于衡量端到端成功率。
- 动作失败时的错误恢复钩子。

## 交付物

本课程产出 `outputs/skill-multimodal-agent-designer.md`。给定一个电脑使用产品（领域、动作集、评估目标），设计完整的代理循环、记忆策略、定位模式以及预期基准得分。

## 练习

1. 用 `screenshot_region` 工具（裁剪 + 放大）扩展动作 schema。哪些任务受益？
2. 阅读 AgentVista（arXiv:2602.23166）。描述最困难的任务类别，以及为什么前沿模型仍然失效。
3. 长时记忆压缩：设计一个 summary-chain，最多保留 ≤4 张实时截图，其余任意数量记录在日志中。
4. 构建错误恢复钩子：在动作失败（未找到按钮）时，代理接下来做什么？
5. 比较仅截图的 Claude 4.7 与 混合截图 + 可访问性树的 Qwen2.5-VL，在 10 个网页任务上的表现。哪些任务各自占优？

## 关键词

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| GUI 定位 | “点击坐标” | 模型输出截图上指令目标的 (x,y) |
| Action schema | “工具定义” | 有效动作的 JSON 描述（click、type、scroll、drag） |
| 可访问性树 | “结构化 DOM” | 来自浏览器 / iOS API 的机器可读 UI 层级 |
| 混合代理 | “截图 + 树” | 同时使用图像与结构化信息；比单独使用更可靠 |
| 视觉工具使用 | “放大/裁剪/检测” | 代理在中途调用外部视觉工具（OCR、检测） |
| Summary-chain | “记忆压缩” | 定期用文本摘要替代长截图历史 |
| VisualWebArena | “端到端网页基准” | 2024 年用于端到端网页任务的基准 |
| AgentVista | “2026 年困难基准” | 涵盖 12 个领域的真实工作流；即便 Gemini 3 Pro 得分也仅 ~30% |

## 延伸阅读

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)  
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)  
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)  
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)  
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)  
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)