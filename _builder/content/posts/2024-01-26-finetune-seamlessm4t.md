---
layout: posts
title: "SeamlessM4T语音翻译微调之路"
subtitle: ""
description: "粤语微调示例"
excerpt: ""
date: 2023-12-13 12:00:00
author: "rickyang"
image: "/images/posts/10.jpg"
published: true
tags:
  - python
  - huggingface
URL: "/2024/01/26/finetune-seamlessm4t"
categories:
  - 模型微调
  - AI
is_recommend: true
---



## TL;DR

 [训练脚本及推理WebUI](https://github.com/iohub/SeamlessM4T-finetune)



## 环境安装

```sh
git clone https://github.com/facebookresearch/seamless_communication
cd seamless_communication
pip install .

```


## 数据处理

- 下载`google/fleurs`语料

```sh
cd /media/ssd/
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/google/fleurs
cd fleurs
# pull 英文语料
git lfs pull --include  data/en_us/audio/*.tar.gz
# pull 粤语语料
git lfs pull --include  data/yue_hant_hk/audio/*.tar.gz
```

- 处理训练语料

**注意**：seamless_communication代码`未直接支持`粤语微调，需要`魔改代码`适配粤语处理。

1. 修改`cli/m4t/finetune/dataset.py`  `UNITY_TO_FLEURS_LANG_MAPPING`  语言映射字典，添加`cmn`中文语言，并指向粤语的语料路径。

```py
UNITY_TO_FLEURS_LANG_MAPPING = {
    "eng": "en_us",
    ...,
    "cmn": "yue_hant_hk", # 新增粤语语料
}
```

2. 修改`datasets/huggingface.py` 添加本地语料加载逻辑。

```python
    
# 添加tsv语料元数据加载函数
# 构建 {'1032454.wav': {'id': 语料ID, 'transcription': 翻译文本} } 数据字典
def load_tsv_meta(self, filename: str) -> Dict:
    meta = {}
    with open(filename, 'r') as fobj:
        for line in fobj:
            buf = line.strip().split('\t')
            meta[buf[1]] = {'id': buf[0], 'transcription': buf[2]}
    return meta
    
# 修改函数支持加载本地语料
def iterate_lang_audio_samples(self, lang: str) -> Iterable[MultimodalSample]:
    # 写死语料根路径
    rpath = '/media/ssd/data/fleurs/data'
    path = f'{rpath}/{lang}'
    ds = load_dataset(
        # self.HF_FLEURS_DATASET_NAME,
        # lang,
        path=path, # 指定加载路径
        split=self.split,
        cache_dir=self.dataset_cache_dir,
        streaming=False,
        trust_remote_code=True,
    )
    # 写死加载train元数据
    meta = self.load_tsv_meta(f'{rpath}/{lang}/train.tsv')
    for item in ds:
        # audio_path = os.path.join(os.path.dirname(item["path"]), item["audio"]["path"])
        audio_path = item["audio"]["path"]
        wavfile = audio_path.split('/')[-1] # 抽取wav文件名
        if wavfile not in meta:
            continue
        mitem = meta[wavfile]
        (sample_id, audio_local_path, waveform, sampling_rate, text) = (
            mitem["id"], # 设置语料ID
            audio_path,
            item["audio"]["array"],
            item["audio"]["sampling_rate"],
            mitem["transcription"], # 设置翻译文本
        )
        yield self._prepare_sample(
            sample_id=sample_id,
            audio_local_path=audio_local_path,
            waveform_npy=waveform,
            sampling_rate=sampling_rate,
            text=text,
            lang=lang,
        )    
```

重新源码安装seamless_communication，然后执行数据处理

```sh
# 安装修改后的函数库
cd seamless_communication
pip install .

# 执行train数据处理
m4t_prepare_dataset --source_lang eng --target_lang cmn --split train --save_dir /media/ssd/m4t-traindata
```

3. 修改`datasets/huggingface.py` 加载`dev.tsv` 校验集元数据。

```python
# 写死加载dev元数据 
meta = self.load_tsv_meta(f'{rpath}/{lang}/dev.tsv')
```

   重新源码安装seamless_communication，然后执行数据处理

```sh
# 安装修改后的函数库
cd seamless_communication
pip install .

# 执行train数据处理
m4t_prepare_dataset --source_lang eng --target_lang cmn --split validation --save_dir /media/ssd/m4t-traindata
```

4. 编写脚本`过滤units过长的训练样本`

    (不过滤，训练会crash报错，提示输入样本的向量长度大于模型的最大长度)

```python
# truncate-fleurs-corpus.py
import sys
import json

in_filename = sys.argv[1]
max_length = int(sys.argv[2])

valid_lines = []
with open(in_filename, 'r') as file:
   for line in file:
       try:
           data = json.loads(line.strip())
           if len(data['target']['units']) < max_length:
               valid_lines.append(line)
           else:
               print('drop sample')
       except Exception as e:
           print(e)

with open(in_filename, 'w') as file:
   for line in valid_lines:
       file.write(line)
```

  执行脚本过滤样本

```sh
python3 truncate-fleurs-corpus.py /media/ssd/m4t-traindata/train_manifest.json 2030 
python3 truncate-fleurs-corpus.py /media/ssd/m4t-traindata/validation_manifest.json 2030 
```



## 执行微调

```sh
#!/bin/bash

DATASET_DIR=/media/ssd/m4t-traindata

m4t_finetune \
   --mode TEXT_TO_SPEECH \
   --train_dataset $DATASET_DIR/train_manifest.json  \
   --eval_dataset $DATASET_DIR/validation_manifest.json \
   --learning_rate 1e-6 \
   --warmup_steps 100 \
   --max_epochs 10 \
   --patience 5 \
   --model_name seamlessM4T_large \
   --save_model_to $DATASET_DIR/checkpoint.pt
```

