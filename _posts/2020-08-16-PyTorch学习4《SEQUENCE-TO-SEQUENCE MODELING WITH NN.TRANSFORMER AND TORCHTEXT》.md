---
layout:     post
title:      PyTorch学习4《SEQUENCE-TO-SEQUENCE MODELING WITH NN.TRANSFORMER AND TORCHTEXT》
subtitle:   SEQUENCE-TO-SEQUENCE模型和nn.Transformer和TorchText
date:       2020-08-16
author:     Zhuoran Li
header-img: img/post-bg-pytorch.jpg
catalog: true
tags:
- PyTorch
---

这是一个如何用 [nn.Transformer](https://pytorch.org/docs/master/nn.html?highlight=nn%20transformer#torch.nn.Transformer) 模块训练sequence-to-sequence模型的教程。

PyTorch 1.2 release 包括一个标准的Transoformer模块，它基于论文 [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)。Transformer模型已被证明对于更具并行性的sequence-2-sequence的问题上，在效果上是表现优异的。这个 `nn.Transformer`模块完全依赖注意力机制（另一个最近实现的模块为 [nn.MultiheadAttention](https://pytorch.org/docs/master/nn.html?highlight=multiheadattention#torch.nn.MultiheadAttention)）绘制全局输入和输出之间的依赖。`nn.Transformer`模块已经高度模块化，使得这样的单个组件（如教程里的[nn.TransformerEncoder](https://pytorch.org/docs/master/nn.html?highlight=nn%20transformerencoder#torch.nn.TransformerEncoder) ）可以简单的调整/组合。

![https://pytorch.org/tutorials/_images/transformer_architecture.jpg](https://pytorch.org/tutorials/_images/transformer_architecture.jpg)

## 模型定义

这篇教程中，我们在语言模型任务上训练 `nn.TransformerEncoder` 模型。语言模型任务是对一个单词序列后的一个单词（或一序列单词）分配一个可能概率。首先一个单词序列传入嵌入层，后面是一个位置编码层，用于考虑单词的顺序（更多细节可在下一段看到）。 `nn.TransformerEncoder` 由多个 [nn.TransformerEncoderLayer](https://pytorch.org/docs/master/nn.html?highlight=transformerencoderlayer#torch.nn.TransformerEncoderLayer)层构成。输入序列的同时，需要一个平方级的注意力掩码，因为 `nn.TransformerEncoder` 的自我注意力层只允许注意到序列中单词之前的位置。对于语言建模任务，未来位置上的任何单词都应该被屏蔽。为了获得真实单词， `nn.TransformerEncoder` 模型的输出被发送到最后的线性层，其后还有一个log-Softmax函数。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
```

`PositionalEncoding` 模块添加一些词在序列中相对或绝对位置信息。positional encodings与embeddings具有相同的维数，因此可以将两者相加。这里，我们使用不同频率的 `sine` 和 `cosine`函数。

```python
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
```

## 加载和batch数据

训练过程使用来自`torchtext`的Wikitext-2数据集。vocab对象基于训练数据集建立，并数值化为tensor类型。从序列数据开始，`batchify()`函数将数据集重排成列，在将数据划分为批大小为batch_size之后，去掉所有剩余的标记。例如，使用字母表作为序列（总长度为26），batch size大小为4，我们将字母表分成4个长度为6的序列：
$$
\begin{split}\begin{bmatrix} \text{A} & \text{B} & \text{C} & \ldots & \text{X} & \text{Y} & \text{Z} \end{bmatrix} \Rightarrow \begin{bmatrix} \begin{bmatrix}\text{A} \\ \text{B} \\ \text{C} \\ \text{D} \\ \text{E} \\ \text{F}\end{bmatrix} & \begin{bmatrix}\text{G} \\ \text{H} \\ \text{I} \\ \text{J} \\ \text{K} \\ \text{L}\end{bmatrix} & \begin{bmatrix}\text{M} \\ \text{N} \\ \text{O} \\ \text{P} \\ \text{Q} \\ \text{R}\end{bmatrix} & \begin{bmatrix}\text{S} \\ \text{T} \\ \text{U} \\ \text{V} \\ \text{W} \\ \text{X}\end{bmatrix} \end{bmatrix}\end{split}
$$
模型将这些列视为独立的，这意味着G和F之间的前后连接依赖关系不能被学习，但这样可以实现高效的批处理。

```python
import torchtext
from torchtext.data.utils import get_tokenizer
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)
```

Out:

```
downloading wikitext-2-v1.zip
extracting
```

### 生成输入和目标序列的函数

`get_batch()`函数产生transformer的输入和目标序列。它将源数据细分为`bptt`长度块。对于语言模型任务，模型需要后续的单词作为`Target`。例如，`bptt`为2，我们得到两个`i=0`的变量如以下：

![https://pytorch.org/tutorials/_images/transformer_input_target.png](https://pytorch.org/tutorials/_images/transformer_input_target.png)

需要注意的是，块沿着维度0，与Transformer模型中的`S`维度一致。batch维度`N`沿着维度1。

```python
bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
```

## 初始化实例

模型是用下面的超参数建立的。vocab大小等于vocab对象的长度。

```python
ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
```

## 运行模型

[CrossEntropyLoss](https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss) 用于跟踪损失， [SGD](https://pytorch.org/docs/master/optim.html?highlight=sgd#torch.optim.SGD) 实现了随机梯度下降方法作为优化器。初始的学习率设置为5.0。 [StepLR](https://pytorch.org/docs/master/optim.html?highlight=steplr#torch.optim.lr_scheduler.StepLR)用于通过epoch调整学习率。训练期间，我们用 [nn.utils.clip_grad_norm_](https://pytorch.org/docs/master/nn.html?highlight=nn%20utils%20clip_grad_norm#torch.nn.utils.clip_grad_norm_) 函数把所有梯度放在一起防止爆炸增长。

```python
criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)
```

循环epoch。如果验证损失是我们迄今为止看到的最好的，请保存模型。在每个epoch后调整学习率。

```python
best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
```

Out:

<pre style="background-color: #fafae2;
    border: 0;
    max-height: 30em;
    overflow: auto;
    padding-left: 1ex;
    margin: 0px;
    word-break: break-word;">
| epoch   1 |   200/ 2981 batches | lr 5.00 | ms/batch 29.40 | loss  8.05 | ppl  3134.23
| epoch   1 |   400/ 2981 batches | lr 5.00 | ms/batch 28.41 | loss  6.80 | ppl   895.33
| epoch   1 |   600/ 2981 batches | lr 5.00 | ms/batch 28.53 | loss  6.37 | ppl   581.93
| epoch   1 |   800/ 2981 batches | lr 5.00 | ms/batch 28.56 | loss  6.23 | ppl   508.02
| epoch   1 |  1000/ 2981 batches | lr 5.00 | ms/batch 28.57 | loss  6.12 | ppl   453.72
| epoch   1 |  1200/ 2981 batches | lr 5.00 | ms/batch 28.57 | loss  6.08 | ppl   439.08
| epoch   1 |  1400/ 2981 batches | lr 5.00 | ms/batch 28.48 | loss  6.04 | ppl   418.19
| epoch   1 |  1600/ 2981 batches | lr 5.00 | ms/batch 28.60 | loss  6.05 | ppl   423.81
| epoch   1 |  1800/ 2981 batches | lr 5.00 | ms/batch 28.56 | loss  5.95 | ppl   383.52
| epoch   1 |  2000/ 2981 batches | lr 5.00 | ms/batch 28.56 | loss  5.96 | ppl   387.89
| epoch   1 |  2200/ 2981 batches | lr 5.00 | ms/batch 28.58 | loss  5.84 | ppl   345.31
| epoch   1 |  2400/ 2981 batches | lr 5.00 | ms/batch 28.58 | loss  5.89 | ppl   363.01
| epoch   1 |  2600/ 2981 batches | lr 5.00 | ms/batch 28.55 | loss  5.89 | ppl   362.94
| epoch   1 |  2800/ 2981 batches | lr 5.00 | ms/batch 28.61 | loss  5.80 | ppl   331.35
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 88.75s | valid loss  5.68 | valid ppl   291.92
-----------------------------------------------------------------------------------------
| epoch   2 |   200/ 2981 batches | lr 4.51 | ms/batch 28.77 | loss  5.80 | ppl   330.42
| epoch   2 |   400/ 2981 batches | lr 4.51 | ms/batch 28.61 | loss  5.78 | ppl   323.73
| epoch   2 |   600/ 2981 batches | lr 4.51 | ms/batch 28.61 | loss  5.61 | ppl   272.92
| epoch   2 |   800/ 2981 batches | lr 4.51 | ms/batch 28.58 | loss  5.64 | ppl   281.10
| epoch   2 |  1000/ 2981 batches | lr 4.51 | ms/batch 28.59 | loss  5.58 | ppl   265.93
| epoch   2 |  1200/ 2981 batches | lr 4.51 | ms/batch 28.63 | loss  5.62 | ppl   275.78
| epoch   2 |  1400/ 2981 batches | lr 4.51 | ms/batch 28.62 | loss  5.62 | ppl   277.17
| epoch   2 |  1600/ 2981 batches | lr 4.51 | ms/batch 28.59 | loss  5.66 | ppl   287.37
| epoch   2 |  1800/ 2981 batches | lr 4.51 | ms/batch 28.63 | loss  5.59 | ppl   268.30
| epoch   2 |  2000/ 2981 batches | lr 4.51 | ms/batch 28.58 | loss  5.62 | ppl   275.64
| epoch   2 |  2200/ 2981 batches | lr 4.51 | ms/batch 28.59 | loss  5.52 | ppl   249.40
| epoch   2 |  2400/ 2981 batches | lr 4.51 | ms/batch 28.59 | loss  5.58 | ppl   266.10
| epoch   2 |  2600/ 2981 batches | lr 4.51 | ms/batch 28.62 | loss  5.59 | ppl   268.40
| epoch   2 |  2800/ 2981 batches | lr 4.51 | ms/batch 28.62 | loss  5.52 | ppl   248.61
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 88.77s | valid loss  5.60 | valid ppl   269.20
-----------------------------------------------------------------------------------------
| epoch   3 |   200/ 2981 batches | lr 4.29 | ms/batch 28.75 | loss  5.55 | ppl   257.16
| epoch   3 |   400/ 2981 batches | lr 4.29 | ms/batch 28.62 | loss  5.55 | ppl   258.39
| epoch   3 |   600/ 2981 batches | lr 4.29 | ms/batch 28.62 | loss  5.37 | ppl   214.39
| epoch   3 |   800/ 2981 batches | lr 4.29 | ms/batch 28.54 | loss  5.43 | ppl   227.01
| epoch   3 |  1000/ 2981 batches | lr 4.29 | ms/batch 28.62 | loss  5.38 | ppl   216.50
| epoch   3 |  1200/ 2981 batches | lr 4.29 | ms/batch 28.61 | loss  5.42 | ppl   225.33
| epoch   3 |  1400/ 2981 batches | lr 4.29 | ms/batch 28.56 | loss  5.43 | ppl   228.89
| epoch   3 |  1600/ 2981 batches | lr 4.29 | ms/batch 28.65 | loss  5.47 | ppl   238.54
| epoch   3 |  1800/ 2981 batches | lr 4.29 | ms/batch 28.58 | loss  5.41 | ppl   222.85
| epoch   3 |  2000/ 2981 batches | lr 4.29 | ms/batch 28.60 | loss  5.44 | ppl   229.49
| epoch   3 |  2200/ 2981 batches | lr 4.29 | ms/batch 28.64 | loss  5.32 | ppl   205.17
| epoch   3 |  2400/ 2981 batches | lr 4.29 | ms/batch 28.57 | loss  5.41 | ppl   223.93
| epoch   3 |  2600/ 2981 batches | lr 4.29 | ms/batch 28.63 | loss  5.41 | ppl   224.31
| epoch   3 |  2800/ 2981 batches | lr 4.29 | ms/batch 28.59 | loss  5.35 | ppl   209.62
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 88.77s | valid loss  5.53 | valid ppl   251.20
-----------------------------------------------------------------------------------------
</pre>


## 使用测试数据集评估模型

用测试数据集检查最好的模型。

```python
test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
```

Out:

```
=========================================================================================
| End of training | test loss  5.42 | test ppl   226.79
=========================================================================================
```