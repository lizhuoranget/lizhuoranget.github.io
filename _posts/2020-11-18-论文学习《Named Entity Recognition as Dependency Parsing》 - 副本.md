---
layout:     post
title:      论文学习《Named Entity Recognition as Dependency Parsing》
subtitle:   实体识别作为依存句法分析
date:       2020-11-18
author:     Zhuoran Li
header-img: img/post-bg-posthk-web.jpg
catalog:    true
tags:
      - 命名实体识别
---

## Named Entity Recognition as Dependency Parsing

**Juntao Yu♣, Bernd Bohnet♠ and Massimo Poesio♣** 

*♣* Queen Mary University

*♠* Google Research 

#### 摘要

命名实体识别是一个NLP基本任务，目的是识别出参考实体的文本范围。NER往往只关注平面式实体识别，忽略了嵌入式如[Bank of [China]]。本文基于图的依存句法分析的思想，提出了通过双放射模型提供全局视角的模型。双放射模型对句子中开始和结尾字符进行成对打分，这样探索所有的范围，最终模型准确预测出命名实体。通过8个语料库实验显示了模型在嵌入式和平面式上都能达到SoTA表现，准确率最高提升了2.2个点。

<div contenteditable="plaintext-only"><center class="half">
    <img src="https://lizhuoranget.github.io/images/20201118NER_BiAffine/1.png" width="100%" >
    <p>图1 本文系统的网络架构</p>
</center></div>

#### 代码

https://github.com/juntaoy/biaffine-ner 

#### Discuss

1. 计算所有的span，复杂度很大，如何避免过多的计算成本？
2. 双放射效果真的这么好？原因？

