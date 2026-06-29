# 3D 生成

> 3D 是在 2D->3D 利用上最有优势的模态。2023 年的突破是 3D Gaussian Splatting。2024–2026 年的生成进展把多视图扩散与 3D 重建叠加在一起，能够从单个提示词或照片生成物体和场景。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 4（视觉）, Phase 8 · 07（潜在扩散）  
**Time:** ~45 分钟

## 问题

3D 内容创建很痛苦：

- **表示。** 网格（mesh）、点云、体素栅格、带符号距离场（SDF）、神经辐射场（NeRF）、3D 高斯（3D Gaussians）。每种表示都有权衡。
- **数据稀缺。** ImageNet 有 1400 万张图像。最大的干净 3D 数据集（Objaverse-XL，2023）约有 ~1000 万个对象，大多数质量较低。
- **内存。** 一个 512³ 的体素栅格有 1.28 亿个体素；一个有用的场景 NeRF 每条光线需要 100 万个采样点。生成比重建更难。
- **监督。** 对于 2D 图像你有像素。对于 3D 通常只有少量 2D 视图，需要将其提升到 3D。

到 2026 年的技术栈把这两个问题分开。首先，用扩散模型生成多视图 2D 图像。其次，将一种 3D 表示（通常是 Gaussian splatting）拟合到这些图像上。

## 概念

![3D 生成：多视图扩散 + 3D 重建](../assets/3d-generation.svg)

### 表示：3D Gaussian Splatting（Kerbl 等，2023）

将场景表示为大约 1M 个 3D 高斯云。每个高斯有 59 个参数：位置（3）、协方差（6，或四元数 4 + 缩放 3）、不透明度（1）、球谐色彩（degree 3 下为 48，degree 0 下为 3）。

渲染 = 投影 + alpha 合成。快速（在 4090 上 1080p 约 ~100 fps）。可微。通过对真实照片做梯度下降拟合。一个场景在消费级 GPU 上 5–30 分钟内即可拟合完成。

2023–2024 年的两项叠加创新：
- **生成式高斯斑点（Generative Gaussian splats）。** 像 LGM、LRM、InstantMesh 这样的模型可以直接从一张或几张图像预测高斯云。
- **4D Gaussian Splatting。** 对动态场景，高斯带有逐帧的偏移量（per-frame offsets）。

### 多视图扩散

将预训练的图像扩散模型微调，使其从文本提示或单张图片生成同一物体的一组一致视图。代表工作有 Zero123（Liu 等，2023）、MVDream（Shi 等，2023）、SV3D（Stability，2024）、CAT3D（Google，2024）。通常输出物体周围的 4–16 个视图，再通过 Gaussian splatting 或 NeRF 抬升到 3D。

### 文本到 3D 的流水线

| Model | Input | Output | Time |
|-------|-------|--------|------|
| DreamFusion (2022) | text | NeRF via SDS | ~1 hour per asset |
| Magic3D | text | mesh + texture | ~40 min |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1 min |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30 min |
| LRM (Meta, 2023) | image | triplane | ~5 s |
| InstantMesh (2024) | image | mesh | ~10 s |
| SV3D (Stability, 2024) | image | novel views | ~2 min |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1 min |
| TripoSR (2024) | image | mesh | ~1 s |
| Meshy 4 (2025) | text + image | PBR mesh | ~30 s |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60 s |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30 s |

2025–2026 的方向：直接输出适用于游戏引擎的带 PBR 材质的 text-to-mesh 模型。对于一般对象，多视图扩散作为中间步骤仍是表现最好的方法。

### NeRF（背景说明）

神经辐射场（Neural Radiance Field，Mildenhall 等，2020）。一个小型 MLP 接受 (x, y, z, 视角方向) 并输出 (颜色, 密度)。通过沿光线积分进行渲染。在质量上优于基于网格的新视图合成，但渲染慢 100–1000 倍。对于大多数实时应用已被 Gaussian splatting 取代，但在研究中仍占主导地位。

## 实现

`code/main.py` 实现了一个玩具的 2D “Gaussian splatting” 拟合：将一个合成目标图像（一个平滑的梯度）表示为若干 2D 高斯斑点之和。通过梯度下降优化位置、颜色和协方差以匹配目标。你将看到两个核心操作：前向渲染（splat + alpha 合成）和用梯度下降拟合。

### 步骤 1：2D 高斯斑点

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 步骤 2：通过叠加斑点渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的 3D Gaussian splatting 会根据深度排序高斯并按顺序做 alpha 合成。我们的 2D 玩具只是做相加。

### 步骤 3：用梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 陷阱

- **视图不一致（View inconsistency）。** 如果你独立生成 4 个视图且它们在物体结构上互相矛盾，3D 拟合会变得模糊。解决方法：使用共享注意力的多视图扩散。
- **背面幻觉（Back-side hallucination）。** 单图到 3D 必须“想象”未看到的一侧，质量差异很大。
- **高斯斑点爆炸（Gaussian splat explosion）。** 无约束训练会增长到 1000 万个斑点并过拟合。densification（增密）和修剪启发式（来自 3D-GS 原始论文）是必要的。
- **拓扑问题。** 来自隐式场（SDFs）的网格常有洞或自交。在发布前运行重网格化工具（例如 Blender 的体素重网格化）。
- **训练数据的许可。** Objaverse 的许可混杂；商业使用依模型而异。

## 使用场景

| Task | 2026 推荐 |
|------|-----------|
| 从照片重建场景 | Gaussian splatting（3DGS、Gsplat、Scaniverse） |
| 用于游戏的文本到 3D 对象 | Meshy 4 或 Rodin Gen-1.5（PBR 输出） |
| 图像到 3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 少量图片的新视图合成 | CAT3D、SV3D |
| 动态场景重建 | 4D Gaussian Splatting |
| 头像 / 穿衣人体 | Gaussian Avatar、HUGS |
| 研究 / SOTA | 刚刚发布的最新成果 |

要在游戏或电商流水线中部署生产级 3D：Meshy 4 或 Rodin Gen-1.5 能输出直接导入 Unity / Unreal 的 PBR 网格。

## 部署

保存为 `outputs/skill-3d-pipeline.md`。该 Skill 接受一个 3D 需求说明（输入：文本 / 一张图 / 几张图；输出：mesh / splat / NeRF；用途：渲染 / 游戏 / VR），并输出：流水线（多视图扩散 + 拟合，或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## 练习

1. **简单。** 用 4、16、64 个 Gaussians 运行 `code/main.py`。报告最终 MSE 与目标的差异。
2. **中等。** 扩展为彩色高斯（RGB）。确认重建匹配目标颜色模式。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片重建真实物体。报告拟合时间和在保留视图上的最终 SSIM。

## 关键术语

| Term | 人们通常怎么说 | 实际含义 |
|------|-----------------|---------|
| 3D Gaussian Splatting | "3DGS" | 将场景表示为一簇 3D 高斯；可微的 alpha 合成渲染。 |
| NeRF | "Neural radiance field" | 在三维点上输出颜色 + 密度的 MLP；通过光线积分渲染。 |
| Triplane | "Three 2-D planes" | 将 3D 因式分解为三张轴对齐的 2D 特征平面；比体积方法更廉价。 |
| SDS | "Score distillation sampling" | 使用 2D 扩散模型的 score 作为伪梯度来训练 3D 模型。 |
| Multi-view diffusion | "Many views at once" | 输出一批一致相机视图的扩散模型。 |
| PBR | "Physically-based rendering" | 含有反照率（albedo）、粗糙度、金属度、法线等通道的材质。 |
| Densification | "Grow splats" | 3DGS 的训练启发式：在高梯度区域拆分/克隆斑点以增密。 |

（注：提示词工程 = 提示词工程，RAG = RAG，Embeddings = 嵌入，Fine-tuning = 微调，Context window = 上下文窗口，few-shot = 少样本，chain-of-thought = 思维链，guardrails = 护栏，function calling = 函数调用，speculative decoding = 投机性解码，positional embeddings = 位置嵌入，self-attention = 自注意力，instruction tuning = 指令微调，distributed training = 分布式训练，Model Context Protocol = 模型上下文协议。）

## 生产说明：3D 还没有统一的共享底层运行时

不同于图像（latent diffusion + DiT）和视频（时空 DiT），到 2026 年 3D 仍然没有单一主导的运行时。生产决策树依表示方式而分叉：

- **NeRF / triplane。** 推理是光线行进（ray-marching）+ 每个采样点做一个 MLP 前向。512² 的渲染需要数百万次 MLP 前向。需要对光线采样做大批量合并；可用 SDPA/xformers 等优化。
- **多视图扩散 + LRM 重建。** 两阶段流水线。第一阶段（多视图 DiT）是一个类似第 07 课的扩散服务；第二阶段（LRM transformer）是对视图的一次性前向。整体延迟表现是“扩散 + 一次性前向”——据此为每个阶段选择合适的服务原语。
- **SDS / DreamFusion。** 每个资产做优化，而不是推理。把它们当作批处理作业，不是请求-响应处理器。

对于大多数 2026 年的产品，正确答案是“按需运行多视图扩散模型，异步重建为 3DGS，并提供 3DGS 以供实时查看”。这把工作负载在 GPU 推理服务器（快速）和离线优化器（慢）之间划分得很清晰。

## 深入阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — 多视图扩散。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。