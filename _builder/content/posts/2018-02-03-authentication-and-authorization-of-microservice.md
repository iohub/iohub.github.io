---
layout: posts
title: "解决python自签名证书的 ssl.SSLError问题"
subtitle: ""
description: ""
excerpt: ""
date: 2023-12-13 12:00:00
author: "rickyang"
image: "/images/posts/7.jpg"
published: true
tags:
  - python
  - huggingface
URL: "/2018/05/22/authentication-and-authorization-of-microservice"
categories:
  - 填坑日常
is_recommend: true
---

## 问题现象

使用huggingface下载模型数据和第三方依赖时，由于公司网络使用自签名证书，导致https校验失败无法，完成数据下载。报错信息 : ` (Caused by SSLError(SSLError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:833)'),))`


## 解决方法

在python执行数据下载前，开启全局无校验context。
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

