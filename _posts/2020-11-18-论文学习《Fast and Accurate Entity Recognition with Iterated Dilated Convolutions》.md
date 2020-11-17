---
layout:     post
title:      论文学习《Fast and Accurate Entity Recognition with Iterated Dilated Convolutions》
subtitle:   
date:       2020-11-18
author:     Zhuoran Li
header-img: img/post-bg-posthk-web.jpg
catalog:    true
tags:
      - 命名实体识别
---

## Fast and Accurate Entity Recognition with Iterated Dilated Convolutions

**Emma Strubell♣, Patrick Verga♣, David Belanger♣, Andrew McCallum♣**

*♣* College of Information and Computer Sciences University of Massachusetts Amherst

#### 摘要

如今许多学者基于整个互联网和大数据上研究NLP任务，更快的方法可以节约精力和时间。GPU硬件的优势最近使得双向LSTM将标记输入的任务如NER，作为一个得到字符向量表示的标准方法（经常跟随一个线性链CRF作为预测）。尽管具有准确性和可表示性，但是这些方法没有完全利用GPU的并行能力。本文提出了一个更快的Bi-LSTM的可替代方法用于NER：**Iterated Dilated Convolutional Neural Networks (ID-CNNs)**，对大文本和结构预测相比传统CNN有更好的能力。不像LSTM，对长度N的句子作序列处理在并行机制下页需要*O(N)*时间，ID-CNNs允许固定深度的卷积在整个问答上并行运行。本文描述了一个网络结构，参数共享和训练过程的特殊结合，使得测试时间加速了14-20倍，同时保证了和Bi-LSTM-CRF相当的准确率。而且，ID-CNNs从整个问答聚合上下文训练是更准确的，同时保留8倍的加速。



<div contenteditable="plaintext-only"><center class="half">
    <img src="https://lizhuoranget.github.io/images/20201118NER_ID-CNNs/1.png" width="100%" >
    <p>图1 </p>
</center></div>

#### 代码



#### Discuss

1. 

