# 采样方法

> 采样是 AI 在可能性空间中探索的方式。

**Type:** 构建  
**Language:** Python  
**Prerequisites:** 第一期，课程 06-07（概率，贝叶斯定理）  
**Time:** ~120 分钟

## 学习目标

- 从零实现逆累积分布函数、拒绝采样和重要性采样，仅使用均匀随机数
- 为语言模型生成实现温度、top-k 和 top-p（核）采样
- 解释重参数化技巧以及为什么它使在 VAE 中通过采样进行反向传播成为可能
- 运行 Metropolis-Hastings MCMC，从未标准化的目标分布中采样

## 问题背景

语言模型在处理完你的提示后会产生一个长度为 50,000 的 logits 向量。词表中的每个 token 都对应一个 logit。现在它必须选择一个 token。如何选择？

如果总是选取概率最高的 token，那么每次回答都一样。确定性。无趣。如果完全均匀随机选择，输出会是废话。答案介于这两者之间，而这个“中间地带”由采样来控制。

采样并不限于文本生成。强化学习通过采样轨迹来估计策略梯度。VAE 通过从学习到的分布中采样并对随机性进行反向传播来学习潜在表示。扩散模型通过采样噪声并迭代去噪生成图像。蒙特卡罗方法用来估计没有解析解的积分。MCMC 算法在高维后验分布中探索，这些分布无法被枚举。

每一个生成式 AI 系统本质上都是一个采样系统。采样策略决定输出的质量、多样性和可控性。本课从均匀随机数出发，从头构建所有主要采样方法，直至现代 LLM 和生成模型中使用的技术。

## 概念要点

### 为什么采样重要

在 AI 和机器学习中，采样扮演以下四个基础角色：

- 生成。语言模型、扩散模型和 GAN 都通过采样产生输出。采样算法直接控制创造性、一致性和多样性。温度、top-k 与核采样是工程师每天调整的旋钮。
- 训练。随机梯度下降采样小批量。Dropout 采样要失活的神经元。数据增强采样随机变换。重要性采样通过重加权样本来降低强化学习（PPO、TRPO）中的梯度方差。
- 估计。许多 ML 中的量没有解析解。比如对数据分布的期望损失、能量模型的配分函数、贝叶斯推理中的证据。蒙特卡罗估计通过对样本取平均来近似这些量。
- 探索。MCMC 算法在贝叶斯推理中探索后验分布。进化策略采样参数扰动。Thompson 采样在 bandit 问题中平衡探索与利用。

核心挑战：你只能直接从简单分布（均匀、正态）采样。对于其他分布，需要把简单样本转换为目标分布样本的方法。

### 均匀随机采样

每种采样方法都从这里开始。均匀随机数生成器在 [0, 1) 上产生值，每个等长子区间具有相同概率。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个离散项中均匀采样，生成 U 并返回 floor(n * U)。要从连续区间 [a, b] 均匀采样，计算 a + (b - a) * U。

关键点：一个均匀随机数恰好包含将任意分布产生单个样本所需的随机性。问题在于找对变换。

### 逆 CDF 方法（逆变换采样）

累积分布函数（CDF）把数值映射为概率：

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

逆 CDF 把概率映射回数值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 就服从目标分布。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

指数分布示例：

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

当你能写出 F_inverse 的解析形式时，这个方法非常好用。对于正态分布，没有解析逆 CDF，因此我们使用其他方法（Box-Muller，或数值近似）。

离散版：对于离散分布，构建累积和 CDF，生成 U，找到第一个使累积和超过 U 的索引。这就是第 06 课中 `sample_categorical` 的做法。

### 拒绝采样

当你无法对 CDF 求逆但能（至多到比例常数）计算目标 PDF 时，拒绝采样适用。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

M 越紧，接受率越高。在低维（1-3维）时拒绝采样效果良好。在高维时，接受率呈指数下降，因为大多数 proposal 体积会被拒绝。这是拒绝采样的维度灾难。

示例：从截断正态分布采样，可在截断区间上使用均匀 proposal。包络常数 M 为该区间内正态 PDF 的最大值。

示例：从半圆采样。在包围矩形内均匀提议，若点落在半圆内则接受。这就是 Monte Carlo 计算 π 的方法：接受率等于面积比 π/4。

### 重要性采样

有时你并不需要从目标分布 p(x) 直接采样，而是需要在 p(x) 下估计期望，而你有来自另一个分布 q(x) 的样本。

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

这在强化学习中非常关键。在 PPO（Proximal Policy Optimization）中，你在旧策略 pi_old 下收集轨迹，但想优化新策略 pi_new。重要性权重为 pi_new(a|s) / pi_old(a|s)。PPO 会裁剪这些权重以防新策略偏离旧策略过远。

重要性采样估计器的方差取决于 q 与 p 的相似性。如果 q 与 p 差异很大，少数样本将获得巨大的权重并主导估计。自归一化重要性采样（self-normalized importance sampling）通过除以权重之和来缓解这个问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 蒙特卡罗估计

蒙特卡罗估计通过对随机样本取平均来近似积分。大数定律保证收敛。

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

误差率与维度无关。这就是为什么在高维情况下，蒙特卡罗方法优于基于网格的积分。

估计 π：

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

估计期望值：

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### 马尔可夫链蒙特卡罗（MCMC）：Metropolis-Hastings

MCMC 构造一个马尔可夫链，其平稳分布为目标分布 p(x)。经过足够多步后，链上的样本（近似）服从 p(x)。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

对于对称的 proposal（q(x'|x) = q(x|x')），比值简化为 p(x')/p(x)。这就是原始的 Metropolis 算法。

为什么有效：接受规则保证了详细平衡（detailed balance）：处于 x 并移动到 x' 的概率等于处于 x' 并移动到 x 的概率。详细平衡意味着 p(x) 是该链的平稳分布。

实际注意事项：
- burn-in：在链达到平衡前丢弃早期样本
- thinning：每隔 k 个样本保留一个，以减少自相关
- proposal 尺度：太小链移动慢（高接受率但探索慢）；太大大多数 proposal 被拒绝（低接受率，链被卡住）
- 在高维中，对于高斯 proposal 的最优接受率约为 0.234

### Gibbs 采样

Gibbs 采样是多变量分布下 MCMC 的特例。它不是一次在所有维度上提议移动，而是一次更新一个变量，按其条件分布采样。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs 采样要求能够从每个条件分布 p(x_i | x_{-i}) 中采样。很多模型满足这一点：
- 贝叶斯网络：条件分布由图结构给出
- 高斯混合：条件分布为高斯
- Ising 模型：每个自旋的条件仅依赖于其邻居

接受率始终为 1（每个提议都会被接受），因为从精确条件分布采样会自动满足详细平衡。

局限性：当变量高度相关时，Gibbs 混合速度慢，因为一次更新一个变量无法在分布中做大的对角方向移动。

### 温度采样（在 LLM 中使用）

语言模型输出词表中每个 token 的 logits z_1, ..., z_V。Softmax 将它们转换为概率。温度在 softmax 前对 logits 进行缩放：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

为什么有效：对 logits 除以 T < 1 会放大 logits 之间的差异。例如 z_1 = 2, z_2 = 1，除以 T = 0.5 后变为 4 与 2，使得 gap 更大。经过 softmax 后，最高 logit 的 token 获得更多概率质量。

实际建议：
- T = 0.0：贪心解码，适用于事实问答
- T = 0.3-0.7：略有创造性，适合代码生成
- T = 0.7-1.0：平衡，适合一般对话
- T = 1.0-1.5：富有创造性的写作、头脑风暴
- T > 1.5：越来越随机，通常不常用

温度不会改变可能被选中的 token，它只改变分配给每个 token 的概率质量。

### Top-k 采样

Top-k 采样将候选集合限制为概率最高的 k 个 token，然后对该限制集合重新归一化并从中采样。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k 防止模型选择极不可能的 token（拼写错误、无意义的词），这些通常位于词表的长尾中。问题在于 k 是固定的，不根据上下文调整。当模型很有信心（某个 token 概率为 95%）时，k = 40 仍允许 39 个备选；当模型不确定（概率分散到 1000 个 token）时，k = 40 可能截断很多合理选项。

### Top-p（核采样）

Top-p 动态调整候选集合大小。不保留固定数量的 token，而是保留累积概率超过 p 的最小集合。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

当模型自信时，核采样只保留很少的 token（可能 2-3 个）。当模型不确定时，会保留很多（可能 200 个）。这种自适应行为是核采样通常比 top-k 生成更好文本的原因。

常见组合：
- 温度 0.7 + top-p 0.9：通用良好设置
- 温度 0.0（贪心）：适用于确定性任务
- 温度 1.0 + top-k 50：Fan 等人（2018）原论文设置

Top-k 和 top-p 可以组合使用：先应用 top-k，再对剩余集合使用 top-p。

### 重参数化技巧（VAE 中使用）

变分自编码器（VAE）通过将输入编码为潜空间中的分布、从该分布采样并解码来学习。问题是：无法对采样操作进行反向传播。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

重参数化技巧将随机性与参数分离：

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 分布相同。关键在于把随机性移动到与参数无关的源（epsilon），然后把样本表示为参数的可微变换。

在 VAE 的训练循环中：
1. 编码器为每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重建输入
5. 通过步骤 4、3、2、1 反向传播（因为步骤 3 是可微的）

没有重参数化技巧，VAE 无法用标准反向传播训练。这个单一洞见使得 VAE 成为可行的方法。

### Gumbel-Softmax（可微的分类采样）

重参数化技巧适用于连续分布（高斯）。对于离散的分类分布，需要另一种方法。Gumbel-Softmax 提供了对分类采样的可微近似。

Gumbel-Max 技巧（不可微）：

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

Gumbel-Softmax（可微近似）：

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax 产生离散样本的连续松弛。输出是概率向量（软 one-hot），而不是硬 one-hot。梯度可通过 softmax 流动。在训练的前向传播中，你可以使用 straight-through 估计器：前向用硬 argmax，反向使用软 Gumbel-Softmax 的梯度。

应用：
- VAE 中的离散潜变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 带离散动作的强化学习

### 分层采样（Stratified Sampling）

标准的蒙特卡罗采样可能会在样本空间留下空隙。分层采样通过将空间划分为若干层（strata）并在每层中采样来强制覆盖均匀性。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

分层采样的方差总是小于或等于标准蒙特卡罗：

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

应用：
- 数值积分（准蒙特卡罗）
- 训练数据切分（确保每个 fold 中类别平衡）
- 与重要性采样结合的分层（hybrid techniques）
- NeRF（Neural Radiance Fields）在相机光线上使用分层采样

### 与扩散模型的关联

扩散模型通过一个采样过程生成图像。前向过程在 T 步中向图像加入高斯噪声，直到变为纯噪声；逆过程学习去噪，逐步恢复原图。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

与本课方法的联系：
- 每个去噪步骤使用重参数化技巧（先采样噪声，再应用确定性变换）
- 噪声时间表 {alpha_t} 控制一种温度退火
- 训练使用蒙特卡罗估计来近似 ELBO（证据下界）
- 扩散模型中的祖先采样（ancestral sampling）是一个马尔可夫链（每一步只依赖当前状态）

整个图像生成过程是迭代采样：从噪声开始，每一步采样一个条件于当前状态的稍微低噪声版本，直到恢复出数据。

```figure
monte-carlo-pi
```

## 实践构建

### 步骤 1：均匀与逆 CDF 采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数分布样本并验证均值约为 1/lambda。

### 步骤 2：拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用拒绝采样从截断正态分布中抽样。通过直方图检验样本形状。

### 步骤 3：重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

用均匀 proposal 估计正态分布下 E[X^2]。与已知答案（mu^2 + sigma^2）比较。

### 步骤 4：蒙特卡罗估计 π

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 步骤 5：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布（两个高斯的混合）中采样。可视化链的轨迹。

### 步骤 6：Gibbs 采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 步骤 7：温度采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

演示温度如何改变一组 token logits 的输出分布。

### 步骤 8：Top-k 与 Top-p 采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 步骤 9：重参数化采样

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示重参数化样本可以让梯度通过，而直接采样则不能。

### 步骤 10：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低温度如何使输出接近 one-hot 向量。

完整实现和所有可视化位于 `code/sampling.py`。

## 在实践中使用

使用 NumPy 和 SciPy 时的生产级版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

在大规模 MCMC 中，使用专门库：
- PyMC：带有 NUTS（自适应 HMC）的完整贝叶斯建模
- emcee：基于集群的 MCMC 采样器
- NumPyro/JAX：GPU 加速的 MCMC

你已经从零构建了这些方法。现在你知道库函数背后在做什么。

## 练习

1. 为柯西分布实现逆 CDF 采样。其 CDF 为 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本并将直方图与真实 PDF 对比。注意重尾（远离中心的极端值）。
2. 使用拒绝采样，用 Uniform(0, 1) 作为 proposal，从 Beta(2, 5) 生成样本。将接受的样本与真实 Beta PDF 绘图比较。理论接受率是多少？
3. 使用蒙特卡罗在 0 到 π 上估计 sin(x) 的积分，采样量分别为 1,000、10,000 和 100,000。比较每一级的误差，验证误差符合 O(1/sqrt(N))。
4. 实现 Metropolis-Hastings，从二维分布 p(x, y) ∝ exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 中采样。绘制样本和链的轨迹。尝试不同的 proposal 标准差。
5. 构建一个完整的文本生成演示：给定 10 个单词的词表和对应的 logits，生成长度为 20 的序列，分别使用 (a) 贪心、(b) 温度=0.7、(c) top-k=3、(d) top-p=0.9。在 5 次运行中比较输出的多样性。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sampling | "Drawing random values" | 根据概率分布生成值。所有生成式 AI 的机制核心 |
| Uniform distribution | "All equally likely" | [a, b] 上每个值的概率密度为 1/(b-a)。所有采样方法的出发点 |
| Inverse CDF | "Probability transform" | F_inverse(U) 将均匀样本转换为已知 CDF 的任意分布样本。精确且高效 |
| Rejection sampling | "Propose and accept/reject" | 从简单 proposal 生成样本，按目标/提议比率接受。精确但浪费样本 |
| Importance sampling | "Reweight samples" | 使用来自 q(x) 的样本并按 p(x)/q(x) 加权以估计 p(x) 下的期望。RL 中 PPO 的核心 |
| Monte Carlo | "Average random samples" | 将积分近似为样本平均。误差为 O(1/sqrt(N))，与维度无关 |
| MCMC | "Random walk that converges" | 构造平稳分布为目标分布的马尔可夫链。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | "Accept uphill, sometimes downhill" | 提议移动，按密度比接受。详细平衡保证收敛到目标分布 |
| Gibbs sampling | "One variable at a time" | 每次从条件分布更新一个变量。接受率为 100% |
| Temperature | "Confidence knob" | 在 softmax 前将 logits 除以 T。T<1 使分布更尖锐（更自信），T>1 使分布更平缓（更有多样性） |
| Top-k sampling | "Keep the k best" | 仅保留概率最高的 k 个 token，归一化后采样。候选集大小固定 |
| Nucleus sampling (top-p) | "Keep the probable ones" | 保留累积概率超过 p 的最小 token 集合。候选集大小自适应 |
| Reparameterization trick | "Move randomness outside" | 写作 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。使采样可微。VAE 训练的关键 |
| Gumbel-Softmax | "Soft categorical sampling" | 使用 Gumbel 噪声 + 带温度的 softmax，对分类采样做可微近似 |
| Stratified sampling | "Forced coverage" | 将样本空间划分为层，在每层中采样。方差总不超过朴素蒙特卡罗 |
| Burn-in | "Warm-up period" | MCMC 的初始样本，在链到达平稳分布前被丢弃 |
| Detailed balance | "Reversibility condition" | p(x) * T(x->y) = p(y) * T(y->x)。马尔可夫链平稳分布的一种充分条件 |
| Diffusion sampling | "Iterative denoising" | 从噪声开始，通过学习的去噪步骤逐步生成数据。每一步都是条件采样 |

## 拓展阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - 关于 MCMC 基础的详细教程  
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 原始论文  
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 核采样（top-p）论文  
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 引入重参数化技巧的 VAE 论文  
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - 将采样与图像生成联系起来的 DDPM 论文