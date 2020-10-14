---
layout:     post
title:      CorefQA_Coreference Resolution as Query-based Span Prediction 论文学习笔记
subtitle:   Wei Wu, Fei Wang, Arianna Yuan, Fei Wu and Jiwei Li
date:       2020-10-14
author:     Zhuoran Li
header-img: img/post-bg-posthk-web.jpg
catalog:    true
tags:
      - 共指消解

---

## CorefQA: Coreference Resolution as Query-based Span Prediction

Wei Wu, Fei Wang, Arianna Yuan, Fei Wu and Jiwei Li

Department of Computer Science and Technology, Zhejiang University

Computer Science Department, Stanford University

ShannonAI

#### 摘要

本文针对共指消解提出了一种新的扩展研究—CorefQA。本文可形式化为一个类似问答系统的范围预测任务：对每一个候选实体词基于其上下文生成一个问题，然后范围预测模块用于提取共指的文本范围。这种形式有以下三点优势：（1）范围预测策略可以发现实体标注阶段遗漏的实体；（2）基于问答框架思路，编码实体和其上下文信息到一个问题中，使得问题具有更深层的共指线索信息；（3）现有大量的问答数据集可用于数据增强，以提高模型的泛化能力。实验结果显示出了对之前模型的效果提升，在CoNLL-2012上F1为83.1(+3.5)，在GAP上F1是87.5(+2.5)。

#### 模型

<div contenteditable="plaintext-only"><center class="half">
    <img src="https://lizhuoranget.github.io/images/20201014CorefQA/archi.png" width="100%" >
    <p>图1 CorefQA模型整体架构图</p>
</center></div>

##### 1 描述

给出一个文档的字符序列$X={x_1, X_2, …, x_n}$，$n$是文档的长度。那么$X$中所有可能的文本范围数是$N=n*(n+1)/2$。$e_i$表示第$i$个文本范围，$1 \leq i \leq N$，起始索引和结尾索引分别是FIRST(i)和LAST(i)。$e_i=\{x_{FIRST(i)}, x_{FIRST(i+1)}, …, x_{LAST(i-1)}, x_{LAST(i)}\}$。

共指消解任务是从所有可能的范围中确定先行词。如果一个范围$e_i$不表示一个实体，或者不存在共指的实体，那么为其分配一个虚假先行词$\epsilon$。所有可能范围$e$的链接定义最终的类。

