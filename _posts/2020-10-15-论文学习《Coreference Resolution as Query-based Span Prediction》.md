---
layout:     post
title:      论文学习《Coreference Resolution as Query-based Span Prediction》
subtitle:   共指消解看作基于查询的范围预测
date:       2020-10-15
author:     Zhuoran Li
header-img: img/post-bg-posthk-web.jpg
catalog:    true
tags:
      - 共指消解
---

## CorefQA: Coreference Resolution as Query-based Span Prediction

Wei Wu, Fei Wang, Arianna Yuan, Fei Wu and Jiwei Li

Department of Computer Science and Technology, Zhejiang University

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

##### 2 输入表示

本文使用SpanBERT得到每个字符$x_i$的表示$x_i$ 。之前的方法通常是把speaker信息用二元表示，即连个实体是否来自同一讲话人。然而，本文使用speaker姓名。该思想来自最近的使用个人信息表示speaker的个人对话模型(Li, 2016; Zhang, 2018; Mazare, 2018)。

为了使SpanBERT能够适应长文本，本文使用一个T大小的滑动窗口，每次移动T/2。

##### 3 实体提出

该模型类似(Lee, 2017)，考虑不超过L长度的所有的文本范围。为了提升计算效率，在训练和估计阶段进一步基于贪婪思想对低分实体剪枝。得分计算基于以下：

$$s_m(i)=FFNN_m([x_{FIRST(i),} x_{LAST(i)}])$$

本文只保留得分较高的$\lambda n$ 个实体。

##### 4 范围预测链接实体

对于保留的实体，该文基于问答框架计算出得分$s_a(i,j)$，判断$e_i$和$e_j$是否共指。其在一个三元组上计算{context(X), query(q), answers(a)}。**context** $X$是输入文档。**query** $q(e_i)$是由实体$e_i$所在的句子，对$e_i$加上\<mention\>\</mention\>标签构成。**answer** $a$是$e_i$共指的实体。

基于Devlin, 2019，本文将问题和上下文表示为一个packed sequence。基于Li, 2019对每个字符标记*BIO*。分别表示一个共指代词的*beginning, inside, outside*。如果*X*的字符都是*O*，那么这个问题是不可回答的，也是没有意义的。以下两种情况的问题是不可回答的：（1）范围$e_i$不表示一个实体；（2）$e_i$表示一个实体但在X里没有共指。

BIO分配计算公式：

$$p^{tag}_i=softmax(FFNN(x_i))$$

下一步，将字符级分数扩展到范围级。j是实体i的答案概率得分，从B字符及后面连续的所有I字符进行计算：

$$s_a(j|i)=\frac {1}{|e_j|}[\log p^B_{FIRST(j)}+\sum^{k=LAST(j)}_{k=FIRST(j)+1}logp^I_k]$$

上面公式仅能代表i到j的共指，是单向的，但是共指具有双向性，所以优化公式如下：

$$s_a(i,j)=\frac{1}{2}(s_a(j|i)+s_a(i|j))$$

又由于$e_i$和$e_j$共指需要满足：（1）都是实体（2）共指。所以结合以上公式得：

$$s(i,j)=s_m(i)+s_m(j)+s_a(i,j)$$

##### 5 先行词剪枝

由于n个字符的文档的范围数量级是$O(n^2)$，公式(5)的计算复杂度是$O(n^4)$。由于对于$e_i$考虑所有的$e_j$也是计算复杂的，因为每次还要基于问答模块计算$s_a(i|j)$。所以，这里对$q(e_i)$，仅基于$s_a(j|i)$得分筛选出C个候选范围。

##### 6 训练

本文通过优化边缘对数似然函数，基于标准簇标签从以上剪枝后的C个范围，预测$e_i$的正确先行词。根据Lee, 2017，本文为C增加一个$\epsilon $。根据以下计算出$e_i$的共指分布$P(·)$：

$$P(e_j)=\frac {e^{s(i,j)}}{\sum_{j'\in C}e^{s(i,j')}}$$

##### 7 推断

给出一个文档，根据得分公式得到一个无向图，每一个节点代表一个实体。根据公式(6)找到最大边进行剪枝。如果一个节点的最邻近节点是$\epsilon$，则舍弃。最终，从该图得到实体簇。

##### 8 使用问答数据集增强数据

本文假设问答需要的推理知识（如同义词，世界知识，语法变化以及多句推理）对共指消解也是必不可少的。问答数据集因为没有那么高的语法要求，所以数据集更丰富。本文的共指消解和现有的问答数据集格式相同。因此，本文使用Quoref (Dasigi, 2019)和SQuAD (Rajpurkar, 2016)进行数据增强，预训练实体链接网络。

##### 9 总结和讨论

本文模型比现有的模型具有更强的实体发现能力，能够找到实体提出阶段遗漏的实体。但是不是所有的遗漏实体都能够被再次找到的。例子所示，对共指实体{}，如果部分{}被遗漏了，可以通过未遗漏的实体问题{themselves}找到。如果簇的所有实体都遗漏了，则没有问题了，所以就无法补全遗漏的实体了。但是实体提出阶段能够找到大量的实体，所以整个簇被遗漏的情况很少。这是该模型的优越性。然而，如何在共指消解中完全去除实体提出模块仍然是一个问题。

