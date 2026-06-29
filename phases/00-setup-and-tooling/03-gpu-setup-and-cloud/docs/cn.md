# GPU 设置与云

> 在学习阶段使用 CPU 可以，但真正的训练需要 GPU。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** 第0阶段，第01课  
**Time:** ~45 分钟

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 可用性
- 配置 Google Colab 并选择 T4 GPU 以进行免费的云端实验
- 对比 CPU 与 GPU 的矩阵乘法基准测试并测量加速比
- 使用 fp16 经验法则估算可放入显存（VRAM）的最大模型大小

## 问题

在第 1-3 阶段的大多数课程在 CPU 上运行良好。但一旦你开始训练 CNN、transformers 或 LLM（第 4 阶段及以后），就需要 GPU 加速。一次在 CPU 上运行需要 8 小时的训练，在 GPU 上可能只需 10 分钟。

你有三种选择：本地 GPU、云端 GPU，或 Google Colab（免费）。

## 概念

```
你的选项：

1. 本地 NVIDIA GPU
   成本：$0（你已经有了）
   设置：安装 CUDA + cuDNN
   适用场景：常规使用、大数据集

2. Google Colab（免费版）
   成本：$0
   设置：无需
   适用场景：快速实验、家里没有 GPU 的情况

3. 云端 GPU（Lambda, RunPod, Vast.ai）
   成本：$0.20-2.00/小时
   设置：SSH + 安装
   适用场景：严肃训练、大模型
```

## 实践

### 选项 1：本地 NVIDIA GPU

检查是否有 GPU：

```bash
nvidia-smi
```

安装带 CUDA 支持的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选项 2：Google Colab

1. 访问 [colab.research.google.com](https://colab.research.google.com)  
2. Runtime > Change runtime type > T4 GPU  
3. 运行 `!nvidia-smi` 来验证

将本课程的笔记本直接上传到 Colab。

### 选项 3：云端 GPU

对于 Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？没问题。

大多数课程可以在 CPU 上完成。需要 GPU 的课程会明确说明并附上 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 实践：GPU vs CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上面的基准测试并比较 CPU 与 GPU 的时间  
2. 如果你没有 GPU，请在 Google Colab 上运行并比较结果  
3. 检查你的 GPU 显存有多少，并估算你能放入的最大模型（经验法则：fp16 每个参数约占 2 字节）

## 关键术语

| Term | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA 的并行计算平台，允许你在 GPU 上运行代码 |
| VRAM | "GPU memory" | GPU 上的视频内存，与系统内存分离。限制模型大小。 |
| fp16 | "Half precision" | 16 位浮点数，相对于 fp32 占用一半内存且通常精度损失很小 |
| Tensor Core | "Fast matrix hardware" | 专用的 GPU 核心，用于矩阵乘法，比普通核心快 4-8 倍 |