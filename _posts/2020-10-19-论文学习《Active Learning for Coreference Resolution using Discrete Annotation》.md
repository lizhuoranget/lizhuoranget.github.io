---
layout:     post
title:      论文学习《Active Learning for Coreference Resolution using Discrete Annotation》
subtitle:  基于离散标注的共指消解主动学习
date:       2020-10-19
author:     Zhuoran Li
header-img: img/post-bg-posthk-web.jpg
catalog:    true
tags:
      - 共指消解
---

## Active Learning for Coreference Resolution using Discrete Annotation

Belinda Z.Li	Gabriel Stanovsky	Luke Zettlemoyer

University of Washington

Allen Institute for AI

Facebook

#### 摘要

本文针对共指消解标注数据集少的问题，使用了主动学习的方法，对一对实体$<j, i>$，向标注者询问是否共指？如果是，则判断下一对；否则，则提出新的问题，哪个实体是$i$的第一个先行词，并标注出来。通过一个这样的改变，相比传统标注方法节省了大量的成本，同时更加有效，提高了模型结果。

#### 代码
https://github.com/belindal/discrete-active-learning-coref

由于主动学习的研究需要标注成本，所以暂未继续仔细研读该文...
