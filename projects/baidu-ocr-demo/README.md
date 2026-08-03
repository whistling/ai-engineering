# Baidu Unlimited-OCR Demo

> One-shot long-horizon document parsing with [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR).

## Overview

**Unlimited-OCR** 是百度开源的 3.3B 参数文档解析模型，采用 MoE 架构（每个 token 仅激活 ~500M 参数），
支持单次前向传播处理多页文档。

本 demo 提供两种推理方式：

| 方式 | 脚本 | 硬件 | 显存/内存 |
|------|------|------|-----------|
| **MLX (Apple Silicon)** ⭐ | `main_mlx.py` | M1/M2/M3/M4 Mac | ~5GB (Int4) |
| **CUDA (NVIDIA GPU)** | `main.py` | NVIDIA GPU | ~7.3GB (FP16) |

## Apple Silicon 用户（M4 16GB ✅）

### 量化模型选择

| 变体 | 模型 ID | 磁盘大小 | 峰值内存 | 推荐 |
|------|---------|---------|---------|------|
| **Int4** | `sahilchachra/unlimited-ocr-4bit-mlx` | ~2.3 GB | ~5 GB | ⭐ 16GB Mac 首选 |
| **MXFP4** | `sahilchachra/unlimited-ocr-mxfp4-mlx` | ~2.2 GB | ~5 GB | 轻量备选 |
| **Int8** | `sahilchachra/unlimited-ocr-8bit-mlx` | ~3.7 GB | ~6 GB | 更高精度 |
| **MXFP8** | `sahilchachra/unlimited-ocr-mxfp8-mlx` | ~3.6 GB | ~5 GB | 速度/精度平衡 |

### 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv && source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements_mlx.txt

# 3. 运行 OCR（首次会自动下载模型）
python main_mlx.py --image your_image.jpg

# 使用 8bit 模型（更高精度）
python main_mlx.py --image your_image.jpg --model_variant 8bit

# PDF 文档
python main_mlx.py --pdf your_document.pdf

# 多张图片
python main_mlx.py --images page1.png page2.png page3.png

# 自定义 prompt
python main_mlx.py --image your_image.jpg --prompt "Extract all tables from this document."
```

### 输出

结果保存在 `--output_dir`（默认 `./output`）：
- 每张图片一个 `.md` 文件
- 多页模式额外生成 `combined_output.md`

---

## NVIDIA GPU 用户

### 环境要求

- NVIDIA GPU ≥ 8GB VRAM
- CUDA 12.9+
- Python 3.12+

### 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 单张图片 (gundam 配置，默认)
python main.py --image your_image.jpg

# 单张图片 (base 配置)
python main.py --image your_image.jpg --mode base

# 多张图片
python main.py --images page1.png page2.png page3.png

# PDF 文档
python main.py --pdf your_document.pdf
```

### 单图配置说明

| Config | image_size | crop_mode | 适用场景 |
|--------|-----------|-----------|---------|
| **gundam** (默认) | 640 | ✅ | 单页文档，质量最高 |
| **base** | 1024 | ❌ | 大尺寸输入，不裁切 |

---

## 项目结构

```
baidu-ocr-demo/
├── README.md               # 本文件
├── main_mlx.py             # Apple Silicon 推理 (mlx-vlm)
├── main.py                 # NVIDIA GPU 推理 (transformers)
├── download_model.py       # 预下载 GPU 版模型
├── requirements_mlx.txt    # MLX 依赖
└── requirements.txt        # CUDA 依赖
```

## References

- GitHub: https://github.com/baidu/Unlimited-OCR
- HuggingFace: https://huggingface.co/baidu/Unlimited-OCR
- MLX models: https://huggingface.co/sahilchachra
- Paper: https://arxiv.org/abs/2606.23050
