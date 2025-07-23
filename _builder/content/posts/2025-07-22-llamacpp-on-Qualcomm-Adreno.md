---
layout: posts
title: "在高通Adreno GPU上使用OpenCL运行llama.cpp"
subtitle: ""
description: "高性能移动端大语言模型部署指南"
excerpt: ""
date: 2025-07-22 12:00:00
author: "rickyang"
image: "/images/posts/3.jpg"
published: true
tags:
  - llamacpp
  - qualcomm
URL: "/2025/07/22/llamacpp-on-qualcomm-adreno"
categories:
  - LLM
is_recommend: true
---



## 前言

随着大语言模型（LLM）在移动设备上的应用需求日益增长，如何在Android设备上高效运行这些模型成为了开发者关注的焦点。本文将详细介绍如何在搭载高通骁龙芯片的Android设备上，通过OpenCL技术在Adreno GPU上运行llama.cpp，实现高性能的大语言模型推理。

## OpenCL在Adreno GPU上的优势

### OpenCL技术简介

OpenCL（开放计算语言）是由Khronos Group开发的广泛采用的行业标准，它允许开发者编写高效且可移植的并行编程代码，可在CPU、GPU、NPU、FPGA等各种设备上运行，而无需深入了解这些设备的具体实现细节。OpenCL在GPU上的应用特别强大，使开发者能够充分利用现代GPU的并行计算能力，应用于图像/视频/视觉信号处理以及卷积神经网络（CNN）和大语言模型（LLM）等AI工作负载。

### 高通在OpenCL标准化中的贡献

作为Khronos Group中OpenCL工作组的重要成员，高通技术公司积极参与OpenCL标准化工作。作为移动GPU上OpenCL标准的早期采用者之一，高通在各种SoC设备上支持OpenCL，包括：

- 高端、中端和低端Android智能手机
- IoT设备（如无人机）
- 汽车平台
- Windows on Snapdragon (WoS)设备

高通还提供了全面的工具集（Snapdragon Profiler）、OpenCL SDK示例和OpenCL编程指南，帮助开发者在Adreno GPU上开始使用OpenCL。

### llama.cpp OpenCL后端的关键特性

**性能提升**：新的后端显著提升了llama.cpp在Adreno GPU上的性能，实现更快的计算和更高效的处理。

**更广泛的兼容性**：该后端针对Adreno GPU进行了高度优化，同时也可在所有支持OpenCL 3.0标准（带子组支持）的GPU上运行，确保更广泛的兼容性和可访问性。


## 支持的模型和平台

### 测试验证的模型

经过严格测试，以下大语言模型已确认在llama.cpp OpenCL后端上表现良好：

- **llama系列**：包括llama 2和3模型，参数量为70亿（7B）和80亿（8B）等
- **Gemma系列**：Gemma 1&2 2B模型、Phi3 mini
- **Mistral系列**：Mistral 7B模型
- **DeepSeek R1**：蒸馏模型
- **其他**：Qwen 1&2 7B、百川7B

### 支持的硬件平台

该后端已在多款搭载骁龙SoC的高端设备上进行测试：

- **Windows 11笔记本电脑**：搭载骁龙X Elite和骁龙X Plus芯片
- **Android智能手机**：搭载骁龙8 Gen 1、2、3和最新的骁龙8 Elite

## Android平台构建步骤详解

### 环境准备

在开始构建之前，您需要准备以下软件和硬件环境：

**软件要求**：
- Ubuntu 22.04
- Python3、CMake、Make和Ninja
- C/C++编译器
- Android NDK版本26.3.11579264

**硬件要求**：
- 搭载高通骁龙8 Gen 1、2、3或Elite移动平台的Android设备

### 步骤1：安装Android NDK

首先，我们需要安装Android NDK。以下是`install-ndk.sh`脚本的内容：

```bash
cd ~ 
wget https://dl.google.com/android/repository/commandlinetools-linux-8512546_latest.zip && \ 
unzip commandlinetools-linux-8512546_latest.zip && \ 
mkdir -p ~/android-sdk/cmdline-tools && \ 
mv cmdline-tools latest && \ 
mv latest ~/android-sdk/cmdline-tools/ && \ 
rm -rf commandlinetools-linux-8512546_latest.zip 
 
yes | ~/android-sdk/cmdline-tools/latest/bin/sdkmanager "ndk;26.3.11579264"
```

这个脚本执行以下操作：
1. 下载Android命令行工具
2. 解压并组织目录结构
3. 使用SDK管理器安装指定版本的NDK

### 步骤2：安装OpenCL头文件和ICD加载器

由于NDK发行版中没有直接提供运行OpenCL所需的文件，我们需要从官方Khronos OpenCL仓库下载OpenCL头文件和ICD加载器。`install-opencl.sh`脚本完成这个任务：

```bash
mkdir -p ~/dev/llm
cd ~/dev/llm

# 下载并安装OpenCL头文件
git clone https://github.com/KhronosGroup/OpenCL-Headers
cd OpenCL-Headers
cp -r CL ~/android-sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include

cd ~/dev/llm

# 下载并构建OpenCL ICD加载器
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
cd OpenCL-ICD-Loader

# 修复缺失的stdlib.h包含
sed -i '/#include <unistd.h>/a #include <stdlib.h>' loader/linux/icd_linux_envvars.c

mkdir -p build_ndk26
cd build_ndk26

# 设置编译器标志以处理Android NDK问题
export CFLAGS="-D_Nonnull= -D_Nullable= -D__BIONIC_COMPLICATED_NULLNESS= -D__INTRODUCED_IN= -D__BIONIC_AVAILABILITY= -D__BIONIC_FORTIFY_INLINE= -D__BIONIC_FORTIFY_VARIADIC= -Wno-nullability-completeness -Wno-availability -Wno-attributes -Wno-builtin-declaration-mismatch -Wno-implicit-function-declaration -Wno-int-conversion"
export CXXFLAGS="-D_Nonnull= -D_Nullable= -D__BIONIC_COMPLICATED_NULLNESS= -D__INTRODUCED_IN= -D__BIONIC_AVAILABILITY= -D__BIONIC_FORTIFY_INLINE= -D__BIONIC_FORTIFY_VARIADIC= -Wno-nullability-completeness -Wno-availability -Wno-attributes -Wno-builtin-declaration-mismatch -Wno-implicit-function-declaration -Wno-int-conversion"

# 使用CMake配置构建
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$HOME/android-sdk/ndk/26.3.11579264/build/cmake/android.toolchain.cmake \
  -DOPENCL_ICD_LOADER_HEADERS_DIR=$HOME/android-sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=24 \
  -DANDROID_STL=c++_shared \
  -DCMAKE_C_FLAGS="${CFLAGS}" \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS}"

ninja
cp libOpenCL.so ~/android-sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
```

这个脚本的关键步骤包括：
1. **下载OpenCL头文件**：从Khronos官方仓库获取最新的OpenCL头文件
2. **构建ICD加载器**：编译OpenCL ICD加载器库
3. **处理Android NDK兼容性**：通过编译器标志解决Android NDK的特定问题
4. **交叉编译配置**：针对ARM64架构进行交叉编译

### 步骤3：构建llama.cpp

最后，使用`build-llamacpp.sh`脚本构建带有Adreno OpenCL后端的llama.cpp：

```bash
cd ~/dev/llm 
 
git clone https://github.com/ggerganov/llama.cpp && \ 
cd llama.cpp
mkdir -p build-android && cd build-android 
 
cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$HOME/android-sdk/ndk/26.3.11579264/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_OPENCL=ON \
  -DLLAMA_CURL=OFF 
 
ninja
```

构建配置说明：
- **GGML_OPENCL=ON**：启用OpenCL支持
- **ANDROID_ABI=arm64-v8a**：目标架构为ARM64
- **ANDROID_PLATFORM=android-28**：最低Android API级别
- **BUILD_SHARED_LIBS=OFF**：构建静态库以简化部署


### 调试技巧

- 使用`adb logcat`查看Android设备上的运行时日志
- 通过Snapdragon Profiler分析GPU使用情况
- 检查OpenCL设备枚举是否正确

## 扩展资料

- [官方原文](https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu)
