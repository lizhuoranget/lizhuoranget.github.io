---
layout:     post
title:      论文学习《End-to-end Neural Coreference Resolution》
subtitle:   问题理解以及代码阅读
date:       2020-09-28
author:     Zhuoran Li
header-img: img/post-bg-posthk-web.jpg
catalog:    true
tags:
      - 共指消解

---

## 论文

#### 核心思路

本文介绍了一个端到端的共指消解模型，没有使用语义解析器和手工实体检测。**关键思想是找到文档内所有的可能实体，学习每个实体的先行词概率分布。**模型结合发现中心词的上下文边界表示计算跨度词嵌入。训练时通过共指簇的标准先行词最大化边界似然函数，并剪枝一些代词。这是第一个没有使用外部资源成功训练的模型。实验在*OntoNotes*上提升了*1.5 F1*。

## 代码实现知识

1. tqdm 

   tqdm是一个快速，**可扩展的Python进度条**，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。

   https://blog.csdn.net/qq_33472765/article/details/82940843

2. *args和**kwargs

   这两个是python中的可变参数。*args表示任何多个无名参数，它是一个tuple；**kwargs表示关键字参数，它是一个dict。

   https://www.cnblogs.com/fengmk2/archive/2008/04/21/1163766.html

3. combinations

   Python的itertools库中提供了`combinations`方法可以轻松的实现排列组合。

   https://blog.csdn.net/cloume/article/details/76399093

4. torch.nn.functional.normalize

   本质上就是按照某个维度计算范数，p表示计算p范数（等于2就是2范数），dim计算范数的维度（这里为1，一般就是通道数那个维度）

   https://blog.csdn.net/u013066730/article/details/95208287

5. torch.nn.Embedding

   在pytorch里面实现`word embedding`是通过一个函数来实现的:`nn.Embedding`，只需要调用 torch.nn.Embedding(m, n) 就可以了，m 表示单词的总数目，n 表示词嵌入的维度

   https://www.cnblogs.com/lindaxin/p/7991436.html

   如果从使用已经训练好的词向量，则采用

   ```python
   # GLoVE
   self.glove = nn.Embedding(glove_weights.shape[0], glove_weights.shape[1])
   self.glove.weight.data.copy_(glove_weights)
   ```

   https://blog.csdn.net/david0611/article/details/81090371

6. torch.nn.Conv1d

   https://blog.csdn.net/sunny_xsc1994/article/details/82969867

   torch.nn.Conv1d 与 torch.nn.Conv2d

   https://www.jianshu.com/p/45a26d278473

7. torch.nn.Linear

   `nn.Linear（）`是用于设置网络中的**全连接层的**.

   https://blog.csdn.net/qq_42079689/article/details/102873766

8. torch.nn.Sequential

   一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。

   https://blog.csdn.net/dss_dssssd/article/details/82980222

   Sequential是一个特殊的module，它包含几个子Module，前向传播时会将输入一层接一层的传递下去。ModuleList也是一个特殊的module，可以包含几个子module，可以像用list一样使用它，但不能直接把输入传给ModuleList。

   https://blog.csdn.net/e01528/article/details/84397174

9. torch.cat

   torch.cat是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接，联系在一起。

   https://blog.csdn.net/qq_39709535/article/details/80803003

10. cached_property

    缓存属性 (`cached_property`) 是一个非常常用的功能，很多知名 Python 项目都自己实现过它。

    https://www.dongwm.com/post/cached-property/

11. torch.mul()和torch.mm()的区别

    torch.mul(a, b)是矩阵a和b**对应位相乘**，a和b的**维度必须相等**，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵

    torch.mm(a, b)是矩阵a和b**矩阵相乘**，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵

    https://blog.csdn.net/Real_Brilliant/article/details/85756477





## 代码结构

<center class="half">
    <table>
         <tr>
             <td><p>coref.py</p></td>
            <td><p>loader.py</p></td>
            <td><p>utils.py</p></td>
        </tr>
        <tr>
            <td><img src="https://lizhuoranget.github.io/images/20200928coref/coref.png" width="100%" ></td>
            <td><img src="https://lizhuoranget.github.io/images/20200928coref/loader.png" width="100%" ></td>
            <td><img src="https://lizhuoranget.github.io/images/20200928coref/utils.png" width="100%" ></td>
        </tr>
    </table>
</center>

### coref.py

类：





## 代码错误

1. When I run the coref.py, I have a error is  "list index out of range in utils.py line73, in pack" .
   Is the code in loader.py, line 75 only considers the English but no Chinese and Arabic? I know the separate tokens of Chinese are ['。',  '！', '?']，so I append these. When I done these, the error "list index out of range in utils.py line73, in pack" don't occur.
   But I don't know the Arabic's separate tokens, I hope someone could fix it.

   <center class="half">
       <img src="https://lizhuoranget.github.io/images/20200928coref/error_outoflist.png" width="100%" >
       <p>图 错误</p>
        <img src="https://lizhuoranget.github.io/images/20200928coref/debug_outoflist.png" width="65%" >
       <p>图 代码修改</p>
   </center>

2. I have a error in coref.py, line 551, TypeError: log() got an unexpected keyword argument 'dim'.
   Should dim=0 be removed or move to other place?

   <center class="half">
       <img src="https://lizhuoranget.github.io/images/20200928coref/error_logUnexpectedArg.png" width="100%" >
       <p>图 错误</p>
   </center>
   
3. I have a error AttributeError: module 'torch.utils.data' has no attribute 'IterableDataset'      

   I use the torch=0.4.1, torchtext=0.7.0, when I downgrade torchtext=0.6.0, it fixed.

   <center class="half">
       <img src="https://lizhuoranget.github.io/images/20200928coref/error_IterableDataset.png" width="100%" >
       <p>图 错误</p>
   </center>

