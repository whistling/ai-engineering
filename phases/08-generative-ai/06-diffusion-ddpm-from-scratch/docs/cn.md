# Diffusion Models — DDPM from Scratch

> Ho, Jain, Abbeel (2020) 给这个领域一道无法忘怀的配方。在一千个小步骤中用噪声摧毁数据。训练一个神经网去预测噪声。在推断时反向运行该过程。如今每个主流的图像、视频、3D 和音乐模型都在这个循环上运行，可能还会在上面叠加 flow matching 或一致性技巧。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 3 · 02（反向传播）, Phase 8 · 02（VAE）  
**Time:** ~75 分钟

## The Problem

你想从 `p_data(x)` 采样。GAN 通过博弈（minimax）训练，常常会发散。VAE 的高斯解码器会生成模糊样本。你真正需要的是一个训练目标，它同时满足：(a) 一个稳定的单一损失（没有鞍点、没有博弈），(b) 对 `log p(x)` 的下界（能给出似然），以及 (c) 产生匹配 SOTA 质量的样本。

Sohl-Dickstein 等（2015）给出了一个理论答案：定义一个逐步加入高斯噪声的马尔可夫链 `q(x_t | x_{t-1})`，并训练一个反向链 `p_θ(x_{t-1} | x_t)` 做去噪。Ho, Jain, Abbeel（2020）证明了损失可以简化为一句话 —— 预测噪声 —— 并整理了数学推导。到 2020 年这还是一个趣闻；到 2021 年它产生了最先进的样本；到 2022 年它成为 Stable Diffusion；到 2026 年它成为生成模型的基石。

## The Concept

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**前向过程 `q`。** 在 T 个小步骤中加入高斯噪声。可闭式解 —— 使得数学可操作的原因 —— 是累积步骤仍为高斯分布：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)` 对于一组 `β_t` 的调度。通常在 T=1000 步上线性地把 `β_t` 从 1e-4 选到 0.02，此时 `x_T` 近似 `N(0, I)`。

**反向过程 `p_θ`。** 学习一个神经网 `ε_θ(x_t, t)` 来预测被加入的噪声。给定 `x_t`，按下式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 要么取 `sqrt(β_t)`，要么为学习得到的方差。公式看起来复杂，但只是代数 —— 解出后验 `q(x_{t-1} | x_t, x_0)` 并把 `x_0` 用其由噪声预测的估计替代。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，随机选取 `t`，采样 `ε ~ N(0, I)`，通过闭式解一次性构造噪声化的 `x_t`，并回归噪声。一个损失，无需博弈、无 KL 项、无额外的重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 迭代到 `1` 执行反向步。完成。

## Why it works

三条直观理解：

1. **去噪比生成容易。** 在 `t=T` 时数据变成纯噪声 —— 网络要解决的是一个平凡问题。在 `t=0` 时网络只需清理少量像素。在中间的 `t` 上问题较难，但来自不同噪声水平的梯度都会通过相同权重汇聚，从而共同训练网络。
2. **隐式的得分匹配。** Vincent（2011）证明了预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即*score*。反向 SDE 使用该 score 沿密度梯度上行 —— 朝向高概率区域的有指导随机游走。
3. **ELBO 简化为 MSE。** 完整的变分下界包含对每个时间步的 KL 项。用 DDPM 的参数化方式这些 KL 项可简化为对噪声预测的 MSE（带特定系数）；Ho 在“simple”损失中丢掉了那些系数，且质量反而*提升*了。

```figure
diffusion-denoise
```

## Build It

`code/main.py` 实现了一个 1 维的 DDPM。数据是一个双模混合。所用“网络”是一个接收 `(x_t, t)` 并输出噪声预测的小型 MLP。训练即那一句损失；采样则迭代反向链。

### Step 1: the forward schedule (closed form)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Step 2: sample `x_t` in one shot

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Step 3: one training step

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Step 4: reverse sampling

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于一个 1 维问题，T=40 步和一个 24 单元的 MLP，大约 200 个 epoch 就能学会该双模混合分布。

## Time conditioning

网络需要知道它当前在去噪哪一个时间步。有两种常见方案：

- **正弦嵌入（Sinusoidal embedding）。** 类似 Transformer 的位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过 MLP 投影后在网络中广播使用。
- **FiLM / group-norm 条件化。** 将嵌入投影为每通道的缩放/偏移（FiLM），在每个块中应用。

我们的玩具代码使用正弦嵌入然后 concat。生产级 U-Net 通常使用 FiLM。

## Pitfalls

- **调度非常关键。** 线性 `β` 是 DDPM 的默认，但余弦调度（Nichol & Dhariwal, 2021）在相同算力下能获得更好的 FID。如果质量停止提升，尝试切换调度。
- **时间步嵌入很脆弱。** 对于玩具 1 维问题直接传入原始 `t`（浮点）可能可行，但在图像上会失败；务必使用合适的嵌入。
- **V-prediction 与 ε-prediction。** 在极窄的区间（非常小或非常大的 t）下，`ε` 的信噪比很差。V-prediction（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 使用它。
- **Classifier-free guidance。** 在推断时同时计算有条件和无条件的 `ε`，然后用 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，通常 `w ≈ 3-7`。在 Lesson 08 中有介绍。
- **1000 步很多。** 生产上常用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。见 Lesson 12。

## Use It

| Role | Typical stack in 2026 |
|------|-----------------------|
| Image pixel-space diffusion (small, toy) | DDPM + U-Net |
| Image latent diffusion | VAE encoder + U-Net or DiT (Lesson 07) |
| Video latent diffusion | 时空 DiT（Sora, Veo, WAN） |
| Audio latent diffusion | Encodec + diffusion transformer |
| Science (molecules, proteins, physics) | 等变扩散（EDM, RFdiffusion, AlphaFold3） |

Diffusion 是通用的生成骨干。Flow matching（Lesson 13）是 2024–2026 年间在推理速度上通常胜出的竞争者，在相同质量下更快。

## Ship It

保存为 `outputs/skill-diffusion-trainer.md`。Skill 接受数据集 + 计算预算并输出：调度（linear/cosine/sigmoid）、预测目标（ε/v/x）、步数、guidance scale、采样器家族，以及评估协议。

## Exercises

1. **Easy.** 在 `code/main.py` 中把 T 从 40 改为 10。样本质量（输出的直方图可视化）如何退化？在多少步数下双模结构会崩塌？
2. **Medium.** 从 ε-prediction 切换到 v-prediction。重新推导反向步。比较最终样本质量。
3. **Hard.** 增加 classifier-free guidance。以类别标签 `c ∈ {0, 1}` 作为条件，训练时有 10% 的概率丢弃条件；采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。分别在 `w = 0, 1, 3, 7` 下测量条件模式命中率。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Forward process | "Adding noise" | 固定的马尔可夫链 `q(x_t \| x_{t-1})`，用于摧毁数据。 |
| Reverse process | "Denoising" | 学到的链 `p_θ(x_{t-1} \| x_t)`，用于重构数据。 |
| β schedule | "The noise ladder" | 每步方差；可选 linear、cosine 或 sigmoid 调度。 |
| α̅ | "Alpha bar" | 累乘 `∏(1 - β)`；用于从 `x_0` 得到闭式 `x_t`。 |
| Simple loss | "MSE on noise" | `\|\|ε - ε_θ(x_t, t)\|\|²`；所有变分推导都会归约到这一项。 |
| ε-prediction | "Predict noise" | 输出为被加入的噪声；标准 DDPM 做法。 |
| V-prediction | "Predict velocity" | 输出为 `α·ε - σ·x`；在不同 t 上更稳定的目标。 |
| DDPM | "The paper" | Ho 等 2020；线性 β，1000 步，U-Net。 |
| DDIM | "Deterministic sampler" | 非马尔可夫采样器，20-50 步，训练目标相同。 |
| Classifier-free guidance | "CFG" | 混合有条件与无条件的噪声预测以放大条件信息。 |

## Production note: diffusion inference is a step-count problem

DDPM 论文使用 T=1000 的反向步。没有人在生产中直接这样部署。每个真实的推理栈都会选择以下三种策略之一 —— 每一种都清晰对应于“延迟来自哪里”的生产化讨论：

1. **更快的采样器，使用相同模型。** DDIM（20-50 步）、DPM-Solver++（10-20 步）、UniPC（8-16）。作为反向循环的即插即用替代；训练得到的 `ε_θ` 权重不变。可把延迟降低 20–50×。
2. **蒸馏。** 训练一个学生模型以更少步数匹配教师：Progressive Distillation（2 → 1）、Consistency Models（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。再把延迟降低 ~5–10×，但需要重训。
3. **缓存与编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的 diffusion 后端、`xformers`/SDPA attention、bf16 权重。将每步延迟再降约 2×。可与（1）和（2）叠加使用。

对于生产级 diffusion 服务，预算的讨论与 LLM 文献中描述的情形相同：延迟为 `num_steps × step_cost + VAE_decode`，吞吐率为 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；从用户视角图像生成是“整体完成”的，所以 TPOT 等同于完整响应时间。

## Further Reading

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 先驱性的 diffusion 论文，超前于时代。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，少步采样。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — 余弦调度、学习方差。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — classifier guidance 相关工作。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 统一符号与最清晰的配方。