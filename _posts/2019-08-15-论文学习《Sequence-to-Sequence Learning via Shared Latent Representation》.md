---
layout:     post
title:      论文学习《Sequence-to-Sequence Learning via Shared Latent Representation》
subtitle:   支持不同模态信息进行转换的星状模型
date:       2018-08-15
author:     Zhuoran Li
header-img: img/post-bg-posthk-web.jpg
catalog:    true
tags:
- 多模态
---

![](https://images.ifanr.cn/wp-content/uploads/2018/06/WWDC-10.jpg)

[原文](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16071/15955)

这篇文献主要讲了一个支持不同模态信息进行转换的星状模型，该模型可对文本、视频、语音等不同序列信息进行相互转换，模型主要分为三个部分，多源编码器、星型学习模型、多源解码器。其中，编码器对不同的源可能采用不同的合适的模型编码，视频使用了CNN，文本使用了LSTMs。星型学习器使用SLR（Shared Latent Representation）模型。多源解码器使用LSTM模型。

![](https://img3.doubanio.com/view/status/l/public/ef2fa1864e3a98f.jpg)

SLR学习器分为全模型学习FL、部分模型学习PL、测试三个阶段test，FL阶段全部输入输出通道打开，将随机的数据对（Si,Sj）输入模型，通过损失函数训练出模型，为了防止SLR模型过拟合，PL阶段随机关闭一些输入通道，分别按照0.5、0.25、0.25的概率选择（Si,Sj）、(Si,0)、(0,Sj)三种数据对进行训练。测试阶段打开对应的输入输出通道，输入Si，生成Sj。
模型损失函数基于VAE（Variational Auto-Encoder）进行修改：

![](https://img3.doubanio.com/view/status/l/public/0f56682bcb342b2.jpg)

Llike是期望重构错误率，Lprior是后验正则式。这里仅将Llike改为

![](https://img3.doubanio.com/view/status/l/public/4c8dbc55a25ec02.jpg)


该文献仍要做的研究有：文本信息到视频的转换生成，因为文本内容包含的有效信息很少，很难生成好的视频，所以这里的实验结果看起来相当模糊，仍是研究下一步重点要解决的问题。
这里，考虑可以自己去做一个图像生成模型，通过文本的描述，生成对应的画面结果，可以实现人工智能对不同信息的联想能力。实验数据可以选取具有画面感的文本数据集，或者是描述具体的视频数据集。
