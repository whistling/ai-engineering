# 3D Gaussian Splatting from Scratch

> 一个场景是由数百万个 3D 高斯点云组成。每个高斯有位置、方向、缩放、透明度，以及依赖于视角的颜色。对它们进行光栅化，并对光栅化过程进行反向传播，就完成了。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 13 (3D Vision & NeRF), Phase 1 Lesson 12 (Tensor Operations), Phase 4 Lesson 10 (Diffusion basics optional)
**Time:** ~90 分钟

## 学习目标

- 解释为什么 3D Gaussian Splatting 在 2026 年取代 NeRF 成为用于真实感 3D 重建的生产默认方法
- 说明每个高斯的六个参数（位置、旋转四元数、缩放、透明度、球谐函数颜色、可选特征）以及每项贡献了多少浮点数
- 从头实现一个 2D 高斯 splatting 光栅器，使用 `alpha` 合成，然后展示 3D 情况如何投影到相同的循环上
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 从 20-50 张照片重建一个场景并导出为 `KHR_gaussian_splatting` glTF 扩展或 OpenUSD 26.03 的 `UsdVolParticleField3DGaussianSplat` 模式

## 问题背景

NeRF 将场景存储为一个 MLP 的权重。每个渲染像素都需要沿射线进行数百次 MLP 查询。训练需要数小时，渲染需要数秒，并且权重不可编辑 —— 如果你想移动场景中的椅子，就必须重新训练。

3D Gaussian Splatting（Kerbl, Kopanas, Leimkühler, Drettakis，SIGGRAPH 2023）解决了这些问题。场景是一个显式的 3D 高斯集合。渲染是 GPU 光栅化，能达到 100+ fps。训练只需几分钟。编辑是直接的：平移一部分高斯就能移动椅子。到 2026 年，Khronos 已通过 glTF 的高斯 splat 扩展，OpenUSD 26.03 提供了高斯 splat 模式，Zillow 和 Apartments.com 使用它们渲染房地产，大部分新的 3D 重建研究论文都是基于 3DGS 的变体。

这个思维模型很简单，但数学有足够多的移动部件，以至于多数入门教程从光栅化开始并跳过投影和球谐函数。本课将完整构建整个流程——先做 2D 版本，然后扩展到 3D。

## 概念

### 一个高斯包含的参数

一个 3D 高斯是在空间中的参数化斑块，具有以下属性：

```
position         mu         (3,)    世界坐标系的中心
rotation         q          (4,)    表示方向的单位四元数
scale            s          (3,)    每个轴的对数缩放（渲染时求指数）
opacity          alpha      (1,)    经过 sigmoid 后的透明度 [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   视角依赖颜色（球谐系数）
```

旋转 + 缩放构建一个 3x3 协方差：`Sigma = R S S^T R^T`。这就是高斯在 3D 中的形状。球谐函数允许颜色随视角变化——高光、微妙的光泽、视角依赖的辉光——而不需要存储每视角的纹理。使用 3 阶球谐函数（degree 3），每个颜色通道有 16 个系数，单个高斯仅颜色部分就占 48 个浮点数。

一个场景通常包含 1-5 百万个高斯。每个大约存储 60 个浮点（3 + 4 + 3 + 1 + 48 + 其他）。五百万个高斯大约 240 MB —— 远小于带逐点纹理的点云，也比 NeRF 的 MLP 权重在高分辨率重渲染时所需内存小一个数量级。

### 光栅化，而非射线行进（ray marching）

```mermaid
flowchart LR
    SCENE["数百万个 3D 高斯<br/>(位置, 旋转, 缩放,<br/>透明度, SH 颜色)"] --> PROJ["投影到 2D<br/>(相机外参 + 内参)"]
    PROJ --> TILES["分配到瓦片<br/>(16x16 屏幕空间)"]
    TILES --> SORT["按深度排序<br/>每个瓦片"]
    SORT --> ALPHA["Alpha 合成<br/>从前到后"]
    ALPHA --> PIX["像素颜色"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五个步骤，全部适合 GPU。没有每像素的 MLP 查询。一张 RTX 3080 Ti 可在 147 fps 下渲染 600 万个 splat。

### 投影步骤

世界坐标中位置为 `mu`、协方差为 `Sigma` 的 3D 高斯投影为屏幕位置 `mu'`、二维协方差 `Sigma'` 的 2D 高斯：

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = 视图变换（相机的旋转 + 平移）
J = 在 mu' 处透视投影的雅可比矩阵
```

2D 高斯的投影足迹是一个椭圆，其轴线是 `Sigma'` 的特征向量。椭圆内的每个像素都接收该高斯的贡献，加权项为 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`。

### Alpha 合成规则

对于一个像素，覆盖它的高斯按从后到前排序（或者等价地按从前到后并使用反向公式）。颜色合成使用自 1980s 起所有半透明光栅化器相同的方程：

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       到第 i 个的透过率
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   局部贡献
c_i = eval_SH(SH_i, view_direction)    视角依赖颜色
```

这就是与 NeRF 的体积渲染相同的方程，只是现在是在显式稀疏的高斯集合上求积分，而不是在射线上进行密集采样。正因为这一恒等，渲染质量与 NeRF 可匹敌 —— 两者都在积分相同的辐射场方程。

### 为什么这是可微的

每一步——投影、瓦片分配、alpha 合成、SH 评估——相对于高斯参数都是可微的。给定一张真实图像，计算渲染像素损失，通过光栅器反向传播，用梯度下降更新所有 `(mu, q, s, alpha, c_lm)`。经过 ~30,000 次迭代后，高斯会找到正确的位置、缩放和颜色。

### 增密与剪枝

固定数量的高斯无法覆盖复杂场景。训练包含两个自适应机制：

- **克隆（Clone）**：当某个高斯的梯度幅值很大但其尺度很小时，在当前位置克隆一个高斯——该区域需要更多细节。
- **拆分（Split）**：当一个尺度很大的高斯梯度较大时，把它分成两个较小的高斯——一个大的高斯太平滑以至于无法拟合该区域。
- **剪枝（Prune）**：透明度降到阈值以下的高斯被删除——它们不再贡献。

增密每 N 次迭代运行一次。一个场景通常从 ~100k 的初始高斯（由 SfM 点初始化）增长到训练结束时的 1-5M。

### 球谐函数的一句话说明

视角依赖颜色是单位球面上的函数 `c(direction)`。球谐函数是球面的傅里叶基。在 degree `L` 截断后，每个通道有 `(L+1)^2` 个基函数。对新视角求颜色就是将学习到的 SH 系数与在该视角处评估的基函数做点乘。degree 0 = 一个系数 = 常量颜色。degree 3 = 16 个系数 = 足以捕捉朗伯（Lambertian）着色、高光和轻微反射。许多 SD Gaussian Splatting 论文默认使用 degree 3。

### 2026 年的生产栈

```
1. Capture         智能手机 / DJI 无人机 / 手持扫描仪
2. SfM / MVS       COLMAP 或 GLOMAP 得到相机位姿 + 稀疏点
3. Train 3DGS      nerfstudio / gsplat / inria official / PostShot (~10-30 分钟 在 RTX 4090 上)
4. Edit            SuperSplat / SplatForge（清理漂浮点，分割）
5. Export          .ply -> glTF KHR_gaussian_splatting 或 .usd (OpenUSD 26.03)
6. View            Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 与生成式变体

- **4D Gaussian Splatting** — 高斯随时间变化；用于体积视频（Superman 2026，A$AP Rocky 的 "Helicopter"）。
- **生成式 splats** — 文本到 splat 的模型（World Labs 的 Marble）能推想完整场景。
- **3D Gaussian Unscented Transform** — NVIDIA NuRec 在自动驾驶仿真的变体。

## 实现

### 第 1 步：一个 2D 高斯

我们先构建一个 2D 光栅器。3D 情况在投影后会简化为相同的循环。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      中心
    covs:   (G, 2, 2)   协方差矩阵
    points: (H, W, 2)   像素坐标
    returns: (G, H, W)  每个高斯在每个像素处的密度
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` 为每个（Gaussian, pixel）对计算二次型 `diff^T Sigma^-1 diff`。

### 第 2 步：2D splatting 光栅器

按从前到后的 Alpha 合成。二维中深度没有意义，所以我们使用一个可学习的每高斯标量来表示顺序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     在 [0, 1] 内
    depths:    (G,)     用于排序的每高斯标量
    image_size: (H, W)
    returns:   (H, W, 3) 渲染图像
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

速度不快 —— 真实实现使用基于瓦片的 CUDA 内核 —— 但数学是完全正确且可微的。

### 第 3 步：可训练的 2D splat 场景

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # 在 2D 中的单一角度
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是无约束参数，在渲染时通过合适的激活映射到有效域。这是所有 3DGS 实现的标准模式。

### 第 4 步：将 2D 高斯拟合到目标图像

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # 红色圆
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # 蓝色方块
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

经过 200 步，64 个高斯会稳定地拟合出两个形状。这就是整个想法 —— 对显式几何基元做梯度下降。

### 第 5 步：从 2D 到 3D

3D 扩展保留相同的循环。新增项：

1. 每个高斯的旋转由四元数代替单一角度。
2. 协方差为 `R S S^T R^T`，其中 `R` 从四元数构建，`S = diag(exp(log_scale))`。
3. 投影 `(mu, Sigma) -> (mu', Sigma')` 使用相机外参与在 `mu` 处透视投影的雅可比。
4. 颜色变为球谐展开；在视线方向处评估它。
5. 深度排序由真实相机空间 z 决定，而不是可学习标量。

每个生产实现（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）在 GPU 上用基于瓦片的 CUDA 内核精确地完成以上步骤。

### 第 6 步：球谐函数评估

到 degree 3 的 SH 基有每通道 16 项。评估如下：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   最后一个维度是 RGB 通道
    dirs:      (..., 3)       单位向量
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # 这里省略了 degree 3 的项以节省篇幅；完整的 16 系数版本在代码文件中
    return result
```

学习到的 `sh_coeffs` 存储了“在每个方向上的颜色”。在渲染时对当前视角方向求值，得到一个长度为 3 的 RGB 向量。

## 使用方法

对于真实的 3DGS 工作，请使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的 3DGS 训练器。典型场景在 RTX 4090 上运行 10-30 分钟。

到 2026 年重要的导出选项：

- `.ply` — 原始高斯云（可移植，文件最大）。
- `.splat` — PlayCanvas / SuperSplat 的量化格式。
- glTF `KHR_gaussian_splatting` — Khronos 标准，可在多浏览器/引擎间移植（2026 年 2 月 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat` — USD 原生，适用于 NVIDIA Omniverse 和 Vision Pro 管线。

对于 4D / 动态场景，`4DGS` 与 `Deformable-3DGS` 使用随时间变化的中心和透明度扩展了相同机制。

## 发布产物

本课产出：

- `outputs/prompt-3dgs-capture-planner.md` — 一个规划拍摄会话（照片数量、相机路径、灯光）的提示词，用于特定场景类型。
- `outputs/skill-3dgs-export-router.md` — 一个技能，根据下游查看器或引擎选择合适的导出格式（`.ply` / `.splat` / glTF / USD）。

## 练习

1. **(简单)** 在不同的合成图像上运行上面的 2D splat 训练器。将 `num_splats` 设为 `[16, 64, 256]` 并绘制每个设置下 MSE 随步骤的变化曲线。识别收益递减点。
2. **(中等)** 扩展 2D 光栅器以支持依赖于标量“视角”的每高斯 RGB 颜色，通过 degree-2 的谐函数来建模。在一对目标图像上训练并验证模型能同时重建两个视角。
3. **(困难)** 克隆 `nerfstudio` 并在你自己的 20 张照片采集中训练 `splatfacto`（例如：桌面、植物、人脸、房间）。导出为 glTF `KHR_gaussian_splatting` 并在查看器（Three.js 的 `GaussianSplats3D`、SuperSplat、Babylon.js V9）中打开。报告训练时间、高斯数量和渲染 fps。

## 术语表

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| 3DGS | "Gaussian splats" | 以数百万个 3D 高斯为显式场景表示，每个高斯具有位置、旋转、缩放、透明度、SH 颜色 |
| Covariance | "Shape of the Gaussian" | `Sigma = R S S^T R^T`；单个高斯的方向和各向异性缩放 |
| Alpha compositing | "Back-to-front blend" | 与 NeRF 的体积渲染相同的方程，现在作用于显式稀疏集合 |
| Densification | "Clone and split" | 在重建欠拟合的区域自适应地添加新高斯 |
| Pruning | "Delete low-opacity" | 删除在训练过程中透明度降到接近零的高斯 |
| Spherical harmonics | "View-dependent colour" | 球面的傅里叶基；将颜色作为视角方向的函数存储 |
| Splatfacto | "nerfstudio's 3DGS" | 2026 年训练 3DGS 的最简单路径 |
| `KHR_gaussian_splatting` | "glTF standard" | Khronos 2026 年的扩展，使 3DGS 在查看器和引擎间可移植 |

## 延伸阅读

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — 原始论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — 生产级 CUDA 光栅器
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — 参考训练流程
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026 年可移植格式
- [OpenUSD 26.03 release notes](https://openusd.org/release/) — `UsdVolParticleField3DGaussianSplat` 模式
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — 行业概述