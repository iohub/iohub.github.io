---
layout: posts
title: "starcoder2量化效果评估"
subtitle: ""
description: "使用human-eval评测代码模型"
excerpt: ""
date: 2024-06-18 12:00:00
author: "rickyang"
image: "/images/posts/8.jpg"
published: true
tags:
  - quantization
  - starcoder
URL: "/2024/06/18/starcoder2-eval"
categories:
  - quantization
is_recommend: true
---

## TL;DR

| 模型 | 量化方式 | human-eval pass@1 | human-eval pass@10 | 显存占用 |
|-------|-------|-------|-------|-------|-------|
| starcoder2-15b | - | 22.5% | 67.7% | 30G |
| starcoder2-15b | Q8_0 | 23.2% | 66.4% | 15G |
| starcoder2-15b | Q5_K_M | 15.5% | 56.7% | 12G |
| starcoder2-15b | AWQ | 14.3% | 52.4% | 8G | 

**结论**： 
1. 对模型进行 8-bit量化（Q8_0）显存占用减半，human-eval评测效果与未量化模型接近，量化损失较小。
2. 对模型进行5-bit & 6-bit 混合量化，相对8-bit量化损失较大，节省的显存空间也不多，条件允许建议优先使用8-bit量化。

- PS: DeepSeek-Coder-V2-Instruct 量化效果
| 模型 | 量化方式 | human-eval pass@1 | human-eval pass@10 | 显存占用 |
|-------|-------|-------|-------|-------|-------|
| DeepSeek-Coder-V2-Instruct-16b | - | 62.2% | 84.1% | 32G |
| DeepSeek-Coder-V2-Instruct-16b | Q8_0 | 61.6% | 84.7% | 17G |
| DeepSeek-Coder-V2-Instruct-16b | Q5_K_M | 60.2% | 84.1% | 13G |

**参考**
- **llama.cpp量化方式**

    In the existing ggml quantization types we have "type-0" (Q4_0, Q5_0) and "type-1" (Q4_1, Q5_1). In "type-0", weights w are obtained from quants q using w = d * q, where d is the block scale. In "type-1", weights are given by w = d * q + m, where m is the block minimum. I use this to describe the quantizations being added by this PR.

    The following new quantization types are added to ggml:

    `GGML_TYPE_Q2_K` - "type-1" 2-bit quantization in super-blocks containing 16 blocks, each block having 16 weight. Block scales and mins are quantized with 4 bits. This ends up effectively using 2.5625 bits per weight (bpw)
    
    `GGML_TYPE_Q3_K` - "type-0" 3-bit quantization in super-blocks containing 16 blocks, each block having 16 weights. Scales are quantized with 6 bits. This end up using 3.4375 bpw.
    
    `GGML_TYPE_Q4_K` - "type-1" 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits. This ends up using 4.5 bpw.
    
    `GGML_TYPE_Q5_K` - "type-1" 5-bit quantization. Same super-block structure as GGML_TYPE_Q4_K resulting in 5.5 bpw
    
    `GGML_TYPE_Q6_K` - "type-0" 6-bit quantization. Super-blocks with 16 blocks, each block having 16 weights. Scales are quantized with 8 bits. This ends up using 6.5625 bpw
    
    `GGML_TYPE_Q8_K` - "type-0" 8-bit quantization. Only used for quantizing intermediate results. The difference to the existing Q8_0 is that the block size is 256. All 2-6 bit dot products are implemented for this quantization type.
    This is exposed via llama.cpp quantization types that define various "quantization mixes" as follows:

    `LLAMA_FTYPE_MOSTLY_Q2_K` - uses GGML_TYPE_Q4_K for the attention.vw and feed_forward.w2 tensors, GGML_TYPE_Q2_K for the other tensors.
    
    `LLAMA_FTYPE_MOSTLY_Q3_K_S` - uses GGML_TYPE_Q3_K for all tensors
    
    `LLAMA_FTYPE_MOSTLY_Q3_K_M` - uses GGML_TYPE_Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else GGML_TYPE_Q3_K
    
    `LLAMA_FTYPE_MOSTLY_Q3_K_L` - uses GGML_TYPE_Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else GGML_TYPE_Q3_K
    
    `LLAMA_FTYPE_MOSTLY_Q4_K_S` - uses GGML_TYPE_Q4_K for all tensors
    
    `LLAMA_FTYPE_MOSTLY_Q4_K_M` - uses GGML_TYPE_Q6_K for half of the attention.wv and feed_forward.w2 tensors, else GGML_TYPE_Q4_K
    
    `LLAMA_FTYPE_MOSTLY_Q5_K_S` - uses GGML_TYPE_Q5_K for all tensors
    
    `LLAMA_FTYPE_MOSTLY_Q5_K_M` - uses GGML_TYPE_Q6_K for half of the attention.wv and feed_forward.w2 tensors, else GGML_TYPE_Q5_K
    
    `LLAMA_FTYPE_MOSTLY_Q6_K`- uses 6-bit quantization (GGML_TYPE_Q8_K) for all tensors

## 1. 验证步骤

- 安装huam-eval

```shell

git clone https://github.com/openai/human-eval.git
pip install -e human-eval

```

- 编写评估脚本

```python
from human_eval.data import write_jsonl, read_problems
import requests

# 使用requests实现HTTP推理请求，返回模型生成的代码片段
def generate_one_completion(prompt):
    pass

problems = read_problems()

num_samples_per_task = 200
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)
```

- 填坑
openai提交的代码有点随意，官方的评估脚本上有语法错误，需手工修复。

未修复代码 human_eval/execution.py
```python
def check_correctness(problem: Dict, completion: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    def unsafe_execute():

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            # Disable functionalities that can make destructive changes to the test.
            reliability_guard()

            # Construct the check program and run it.
            check_program = (
                problem["prompt"] + completion + "\n" +
                problem["test"] + "\n" +
                f"check({problem['entry_point']})"
            )

            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
# WARNING
# This program exists to execute untrusted model-generated code. Although
# it is highly unlikely that model-generated code will do something overtly
# malicious in response to this test suite, model-generated code may act
# destructively due to a lack of model capability or alignment.
# Users are strongly encouraged to sandbox this evaluation suite so that it 
# does not perform destructive actions on their host or network. For more 
# information on how OpenAI sandboxes its code, see the accompanying paper.
# Once you have read this disclaimer and taken appropriate precautions, 
# uncomment the following line and proceed at your own risk:
#                         exec(check_program, exec_globals)
                result.append("passed") # result未定义
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")

```

已代码 human_eval/execution.py
```python
def check_correctness(problem: Dict, completion: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    result = []
    def unsafe_execute():

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            # Disable functionalities that can make destructive changes to the test.
            reliability_guard()

            # Construct the check program and run it.
            check_program = (
                problem["prompt"] + completion + "\n" +
                problem["test"] + "\n" +
                f"check({problem['entry_point']})"
            )

            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                        exec(check_program, exec_globals)
                        result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")
```