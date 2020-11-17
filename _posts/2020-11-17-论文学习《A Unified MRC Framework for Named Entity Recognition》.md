---
layout:     post
title:      论文学习《A Unified MRC Framework for Named Entity Recognition》
subtitle:   一个统一命名实体识别的阅读理解框架
date:       2020-11-17
author:     Zhuoran Li
header-img: img/post-bg-posthk-web.jpg
catalog:    true
tags:
      - 命名实体识别
---

## A Unified MRC Framework for Named Entity Recognition

**Xiaoya Li***♣***, Jingrong Feng***♣***, Yuxian Meng***♣***, Qinghong Han***♣***, Fei Wu***♠* **and Jiwei Li***♣* 

*♠* Department of Computer Science and Technology, Zhejiang University 

*♣* Shannon.AI

#### 摘要

NER命名实体识别通常分为嵌入式和平面式两种。模型也通常是按照独立的任务进行设计，因为平面式的一个字符标记为一个标签，这样就不适用于嵌入式。

本文提出一个统一框架，解决平面式和嵌入式。本文不是将NER问题作为一个序列打标签任务，而是一个机器阅读理解（问答）任务。例如，提取一个PERSON标签实体，本文即形式化为query"which person is mentioned in the text"的answer。这解决了嵌入式NER的实体重叠问题：提取两个不同类的重叠实体，对应回答两个不同问题的答案。另外，由于query包含了丰富的知识信息，使得不管平面式还是嵌入式的结果都更好。

本文在嵌入式和平面式数据集上都进行了实验。结果证明了有效性。在当前嵌入式的SOTA模型上有了大的提升，ACE04 +1.28，ACE05 +2.55， GENIA +5.44， KBP17 +6.37。在平面式数据集一样提升了，English CoNLL 2003 +0.24, English OntoNotes 5.0 + 1.95, Chinese MSRA +0.21, Chinese OntoNotes 4.0 +1.49。

<div contenteditable="plaintext-only"><center class="half">
    <img src="https://lizhuoranget.github.io/images/20201117/1.png" width="100%" >
    <p>图1 GENIA和ACE04中的嵌入式实体例子</p>
</center></div>

#### 代码

https://github.com/ShannonAI/mrc-for-flat-nested-ner

#### Discuss

1. 如果重叠实体是同一类型的，比如“小明和小红”，那么是不是两个实体对应的query一样，导致无法识别？
2. 本文提出将更多的同义和示例包含到了query中，对构建query方法进行了深入分析，具体是怎么实现的？

