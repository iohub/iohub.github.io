---
layout: posts
title: "修复Seq2SeqTrainer训练奔溃问题"
subtitle: ""
description: "修复GLM4训练错误：AttributeError: 'NoneType' object has no attribute 'to'"
excerpt: ""
date: 2024-06-20 12:00:00
author: "rickyang"
image: "/images/posts/6.jpg"
published: true
tags:
  - bugfix
  - transformers
URL: "/2024/06/20/index"
categories:
  - transformers
is_recommend: true
---

## TL;DR

修改transformer代码，`src/transformers/tokenization_utils_base.py`
```diff
        if isinstance(device, str) or is_torch_device(device) or isinstance(device, int):
-            self.data = {k: v.to(device=device) for k, v in self.data.items()}
+            self.data = {k: v.to(device=device) for k, v in self.data.items() if v is not None}
        else:
            logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
```

## Root Cause

如`data_collator.py`代码所示 (https://github.com/iohub/transformers/blob/b7672826cad31e30319487af876e608d8af7d37b/src/transformers/data/data_collator.py#L664)

```python
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                import torch

                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None # 问题根因1
```

当数据集未包含`labels`数据时，`DataCollatorForSeq2Seq`会填充为`None`的默认值。正是这个默认值导致程序奔溃。

```diff
        if isinstance(device, str) or is_torch_device(device) or isinstance(device, int):
             # 问题根因2
-            self.data = {k: v.to(device=device) for k, v in self.data.items()}
+            self.data = {k: v.to(device=device) for k, v in self.data.items() if v is not None}
        else:
            logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
```
