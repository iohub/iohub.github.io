---
layout: posts
title: "pytorch多机分布式训练"
subtitle: ""
description: "使用nccl在RDMA网卡执行训练"
excerpt: ""
date: 2024-06-05 12:00:00
author: "rickyang"
image: "/images/posts/6.jpg"
published: true
tags:
  - torch
  - nccl
URL: "/2024/06/05/torch-nccl"
categories:
  - 模型微调
  - AI
is_recommend: true
---

## 0. 环境准备

```sh
pip install transformers datasets peft tensorboard
# 安装flash-attation
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 1. Master节点

```sh
# 配置RDMA网卡
epoxrt NCCL_SOCKET_FAMILY=AF_INET
epoxrt NCCL_SOCKET_IFNAME=eno3p1
export NCCL_IB_HCA=mlx5_1
# 配置nccl日志
export NCCL_DEBUG=INFO
# 在master执行训练
torchrun --nproc_per_node=2 \ # 每个节点的进程数，即每个节点的GPU数
    --nnodes=2 \ # 执行训练的节点总数
    --master_addr="25.25.3.150" \
    --master_port=7200 \
    --node_rank=0 \ # rank: 0 master, 1..N slaver
    finetune.py xxx
    

```


## 2. Slaver节点

```sh
# 配置RDMA网卡
epoxrt NCCL_SOCKET_FAMILY=AF_INET
epoxrt NCCL_SOCKET_IFNAME=eno2p1
export NCCL_IB_HCA=roce01
# 配置nccl日志
export NCCL_DEBUG=INFO
# 在master执行训练
torchrun --nproc_per_node=2 \ # 每个节点的进程数，即每个节点的GPU数
    --nnodes=2 \ # 执行训练的节点总数
    --master_addr="25.25.3.150" \
    --master_port=7200 \
    --node_rank=1 \ # rank: 0 master, 1..N slaver 仅需改变rank参数
    finetune.py xxx
    

```