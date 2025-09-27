---
layout: posts
title: "ComfyUI Qwen-Image-Edit-2509使用指南"
subtitle: ""
description: "Qwen-Image-Edit-2509 Window部署指南"
excerpt: ""
date: 2025-09-22 12:00:00
author: "rickyang"
image: "/images/posts/7.jpg"
published: true
tags:
  - ComfyUI
  - Qwen-Image-Edit-2509
URL: "/2025/09/22/comfyui-qwen-edit-2509"
categories:
  - LLM
is_recommend: true
---


# ComfyUI Qwen-Image-Edit-2509 使用指南

本指南将详细介绍如何通过 Pinokio 安装 ComfyUI 并配置 Qwen-Image-Edit-2509 模型。

## 系统要求

- **显存要求**: 至少 22GB VRAM（FP8版本可以在2080Ti 22G魔改卡上运行）
- **操作系统**: Windows、macOS 或 Linux
- **网络**: 建议配置代理以访问国外资源

## 第一步：安装 Pinokio

### 1.1 下载 Pinokio

1. 访问 [Pinokio 官网](https://pinokio.co/)
2. 下载 **Pinokio Setup 3.9.0.exe**（Windows 版本）
3. 运行安装程序完成安装

Pinokio 被誉为"AI 应用的 Steam"，能够简化 AI 应用的查找、安装和运行过程 <mcreference link="https://www.xda-developers.com/pinokio-how-to/" index="5">5</mcreference>。它为每个应用管理虚拟环境和 Python 依赖项，支持 ComfyUI 等热门 AI 图像生成应用的"一键"安装 <mcreference link="https://www.xda-developers.com/pinokio-how-to/" index="5">5</mcreference>。

### 1.2 配置代理（国内用户必需）

由于许多资源托管在国外服务器，国内用户需要配置代理：

1. 打开 Pinokio
2. 进入设置页面
3. 配置以下环境变量：
   - `HTTP_PROXY`: 你的代理地址（如：http://127.0.0.1:7890）
   - `HTTPS_PROXY`: 你的代理地址（如：http://127.0.0.1:7890）

## 第二步：安装 ComfyUI

### 2.1 一键安装 ComfyUI

1. 打开 Pinokio
2. 导航到 **Discovery** 页面
3. 搜索 **ComfyUI**
4. 点击 ComfyUI 进行一键安装

Pinokio 会自动处理 ComfyUI 的安装过程，包括所有依赖项和虚拟环境的配置 <mcreference link="https://comfyui-wiki.com/en/install/install-comfyui" index="3">3</mcreference>。

### 2.2 更新 ComfyUI

为确保获得最新功能和兼容性：

1. 在 Pinokio 中打开 ComfyUI
2. 导航到 **Manager** 标签页
3. 选择 **Update All** 更新所有组件

定期更新 ComfyUI 可确保访问最新功能和节点，以及重要的错误修复和兼容性增强 <mcreference link="https://www.nextdiffusion.ai/tutorials/how-to-use-qwen-multi-image-editing-in-comfyui-a-step-by-step-guide" index="2">2</mcreference>。

## 第三步：下载工作流文件

### 3.1 获取 Qwen 工作流

1. 访问 [Hugging Face 仓库](https://huggingface.co/datasets/stablediffusiontutorials/Qwen_Image_Workflows/tree/main)
2. 下载 `Qwen_Image_Edit_2509_Multi_Editing.json` 工作流文件

### 3.2 安装工作流

将下载的 `Qwen_Image_Edit_2509_Multi_Editing.json` 文件放置到以下目录：

```
pinokio\api\comfy.git\app\user\default\workflows\
```

或者，你也可以直接将工作流文件拖拽到 ComfyUI 界面中加载 <mcreference link="https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit" index="4">4</mcreference>。

## 第四步：下载模型文件

Qwen-Image-Edit-2509 需要以下模型文件。所有模型都可以在 [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) 或 [Comfy-Org/Qwen-Image-Edit_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI) 找到 <mcreference link="https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit" index="4">4</mcreference>。

### 4.1 主要模型文件

#### 扩散模型（必需）
- **文件名**: `qwen_image_edit_fp8_e4m3fn.safetensors` 或 `qwen_image_edit_bf16.safetensors`
- **存放路径**: `ComfyUI/models/diffusion_models/`
- **说明**: 这些版本需要至少 24GB VRAM。如果显存不足，请使用 GGUF 变体

#### 文本编码器（必需）
- **文件名**: `qwen_2.5_vl_7b_fp8_scaled.safetensors`
- **存放路径**: `ComfyUI/models/text_encoders/`

#### VAE 模型（必需）
- **文件名**: `qwen_image_vae.safetensors`
- **存放路径**: `ComfyUI/models/vae/`

#### LoRA 模型（可选）
- **文件名**: `Qwen-Image-Lightning-4steps-V1.0.safetensors`
- **存放路径**: `ComfyUI/models/loras/`
- **说明**: 用于加速推理的可选模型

### 4.2 目录结构

完整的模型目录结构应如下所示：

```
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   └── qwen_image_edit_fp8_e4m3fn.safetensors
│   ├── 📂 loras/
│   │   └── Qwen-Image-Lightning-4steps-V1.0.safetensors
│   ├── 📂 vae/
│   │   └── qwen_image_vae.safetensors
│   └── 📂 text_encoders/
│       └── qwen_2.5_vl_7b_fp8_scaled.safetensors
```

### 4.3 重启 ComfyUI

下载完所有模型文件后：

1. 重启 ComfyUI
2. 刷新界面以使更改生效

## 第五步：使用 Qwen-Image-Edit-2509

### 5.1 启动工作流

1. 启动 ComfyUI
2. 加载 `Qwen_Image_Edit_2509_Multi_Editing` 工作流
3. 在提示框中编写你的编辑指令
4. 点击生成按钮开始图像编辑



## 高级配置

### 指定 GPU 设备

如果你有多个 GPU 并希望指定特定的 GPU 运行 ComfyUI：

1. 编辑文件：`pinokio\api\comfy.git\app\comfy\cli_args.py`
2. 修改以下行：
   ```python
   parser.add_argument("--cuda-device", type=int, default=1, metavar="DEVICE_ID", 
                      help="Set the id of the cuda device this instance will use. All other devices will not be visible.")
   ```
3. 将 `default=1` 改为你想使用的 GPU ID（通常从 0 开始）

### 共享模型文件夹

Pinokio 使用共享模型文件夹来节省磁盘空间。模型、LoRA、嵌入等文件下载并存储在 `.\pinokio\drive\drives\peers\d1704581225212\` 中，可以被 Automatic1111、Fooocus 和 ComfyUI 共同使用，无需移动或复制模型 <mcreference link="https://github.com/6Morpheus6/pinokio-wiki" index="4">4</mcreference>。

## 故障排除

### 常见问题

1. **白屏问题**: 如果遇到白屏，可能是 Pinokio 版本过旧。删除相应的应用文件夹（如 `.\pinokio\api\comfy.git`）并重新安装 <mcreference link="https://github.com/6Morpheus6/pinokio-wiki" index="4">4</mcreference>。

2. **显存不足**: 如果显存少于 24GB，请使用 GGUF 量化版本的模型。

3. **网络连接问题**: 确保代理配置正确，特别是在下载模型文件时。

4. **模型加载失败**: 检查模型文件是否放置在正确的目录中，并确保文件完整下载。

### Refs

- [ComfyUI 官方文档](https://docs.comfy.org/)
- [Qwen Image Edit - GGUF/Fp8/BF16/LoRA Support in ComfyUI](https://www.stablediffusiontutorials.com/2025/08/qwen-image-edit.html)
- [Qwen Image Edit 2509 GGUF/fp8/Bf16 Multi Image Editing](https://www.stablediffusiontutorials.com/2025/09/qwen-image-edit-2509.html)
