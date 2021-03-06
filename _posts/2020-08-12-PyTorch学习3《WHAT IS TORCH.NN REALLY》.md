---
layout:     post
title:      PyTorch学习3《WHAT IS TORCH.NN REALLY?》
subtitle:   什么是torch.nn?
date:       2020-08-12
author:     Zhuoran Li
header-img: img/post-bg-pytorch.jpg
catalog: true
tags:
- PyTorch
---

PyTorch提供了优美的设计模块和类 [torch.nn](https://pytorch.org/docs/stable/nn.html) ,[torch.optim](https://pytorch.org/docs/stable/optim.html) , [Dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset) , 和 [DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader) ，帮助你创建和训练神经网络。为了让你能够充分的利用并自定义他们来解决自己的问题，你需要对他们实际在做什么有真正准确的理解。为了帮助你理解，我们将在MNIST数据集上训练一个基本的神经网络，该网络没有用任何特征；我们初始将仅用PyTorch Tensor最基本的函数。然后，我们将从 `torch.nn`, `torch.optim`, `Dataset`, 和 `DataLoader` 增加一个特征，展示每一块在做什么，以及如何使得代码更简洁更灵活。

**这篇向导假设你已经安装了PyTorch，并且熟悉tensor的基本操作。**（如果你熟悉Numpy操作，你会发现tensor操作几乎完全相同）。

## MNIST data setup

我们将使用经典的MNIST数据集，其由手写数字（0～9）的黑白图像构成。

我们使用pathlib解决路径（Python3的标准库部分），我们将使用requests下载数据集。我们将只在使用他们时导入，所以你能够看到每一点将用到什么。

```python
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
```

这个数据集是numpy数组格式，并且已经使用pickle存储，pickle是python序列化数据的特殊格式。

```python
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
```

每个图像是28x28像素，并且存储在长度784（=28x28）的扁平化行中。让我们看看其中一个，我们首先需要重塑为2d。

```python
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)
```

![https://pytorch.org/tutorials/_images/sphx_glr_nn_tutorial_001.png](https://pytorch.org/tutorials/_images/sphx_glr_nn_tutorial_001.png)

输出:

```
(50000, 784)
```

PyTorch使用 `torch.tensor`，而不是numpy数组，所以我们需要转换数据。

```python
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())
```

Out:

```
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]]) tensor([5, 0, 4,  ..., 8, 4, 8])
torch.Size([50000, 784])
tensor(0) tensor(9)
```

## Neural net from scratch (no torch.nn)

我们首先只使用tensor操作创建一个模型。我们假设你已经熟悉了基本的神经网络。（如果你不熟悉，你可以学习它 [course.fast.ai](https://course.fast.ai/)）。

PyTorch提供创建随机的或填0的tensor，我们将用其创建简单线性模型中的权重和偏置。这些都是正常的tensor，带有一个非常特殊的条件：我们告诉PyTorch他们需要一个梯度。这将使PyTorch记录tensor上的所有操作，所以在反向传播时可以自动计算出梯度!

对于权重，**我们初始化后设置** `requires_grad`，因为我们不想在梯度中包含这一步。（注意：加一个`_`表示该操作是in-place类型执行）

（译者注：in-place类型是指，改变一个tensor的值的时候，不经过复制操作，而是直接在原来的内存上改变它的值。可以理解为原地操作符。）

* 注意

我们初始化权重Xavier 初始化（通过 1/sqrt(n)的倍增）。

```python
import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)
```

由于PyTorch具有自动计算梯度的能力，我们能用任何标准Python函数（或可调用对象）作为一个模型！所以让我们写一个普通矩阵乘法和广播加法来创建一个简单的线性模型。我们也需要一个激活函数，所以我们将写一个*log_softmax*并使用它。记住：尽管PyTorch提供大量的写好的损失函数，激活函数等等，我们可以使用Python编写自己的函数。PyTorch甚至可以自动创建快速GPU和矢量化CPU代码用于你的函数。

```python
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)
```

以上，`@`代表点乘操作。我们将在一个batch的数据（这个例子中是64张图）上调用自己函数。这是一个前向传递。注意，我们目前没有发现相比随机数更好的预测效果，所以我们从随机权重开始。

```python
bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)
```

Out:

```
tensor([-2.0604, -2.4936, -2.5671, -2.7284, -1.8556, -2.5823, -1.7412, -2.0155,
        -2.9039, -2.9454], grad_fn=<SelectBackward>) torch.Size([64, 10])
```

如上，我们看到`preds`tensor不仅包含tensor值，还有一个梯度函数。我们将在后面的反向传播中用到。

让我们实现负对数似然作为损失函数（我们可以只用纯Python）：

```python
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll
```

让我们用我们的随机模型测试损失函数，这样我们能够看到我们是否有提升在一个反向传递之后。

```python
yb = y_train[0:bs]
print(loss_func(preds, yb))
```

Out:

```
tensor(2.3357, grad_fn=<NegBackward>)
```

让我们实现一个计算模型准确率的函数。对于每一个预测，如果最大值的索引和目标值匹配，则该预测是正确的。

```python
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()
```

让我们使用随机模型检验准确率，所以我们能够看到我们的准确率是否随着损失提升而提升。

```python
print(accuracy(preds, yb))
```

Out:

```
tensor(0.1562)
```

我们现在可以运行一个训练循环。对于每一次迭代，我们将：

* 选择一个mini-batch数据（bs大小）
* 用模型对其做预测
* 计算损失
* `loss.backward()` 更新模型梯度，这个示例中是权重和偏置

我们现在用这些梯度更新权重和偏置。我们在`torch.no_grad()` 的作用域下做这个，因为我们不想对下次梯度计算做这些行为的记录。你可以阅读更多关于PyTorch Autograd记录操作的[资料](https://pytorch.org/docs/stable/notes/autograd.html)。

我们然后设置梯度为0，这样我们可以准备下次循环。否则，我们的梯度将记下所有的已经运行的操作记录（比如 `loss.backward()` 是增加梯度在无论已经存储什么值的情况下，而不是进行替代）。

* 小技巧

你可以用标准的python debugger停止PyTorch的代码，这允许你检查每一步的变量值。去掉注释`set_trace()` 尝试一下。

```python
from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
```

现在，我们已经从零开始完全创建并训练了一个迷你的神经网络（这种例子，一个逻辑回归，由于我们没有隐藏层）！

让我们检查一下损失和准确率，并和我们之前的比较。我们期望损失减少同时准确率提升，我们已经做到了。

```python
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
```

Out:

```
tensor(0.0824, grad_fn=<NegBackward>) tensor(1.)
```

## 使用torch.nn.functional

我们将重构我们的代码，虽然它和之前做了相同的事，只是我们将利用PyTorch的nn类的优势使它更简洁灵活。从这里开始的每一步，我们都将使我们一个或多个的代码更短、更容易理解、更灵活。

首先，最简单的一步是通过 `torch.nn.functional`（通常导入后使用 `F` 作为命名空间）替换我们手工写的激活或损失函数使得代码更短。这个模块包含了 `torch.nn` 库的所有函数（然而该库的其他部分还包含类）。除了大量的损失和激活函数，你还可以找到用于方便创建神经网络的函数，如池函数。（也有卷积层和线性层等的函数，但是我们将看到，用库的其他部分通常能更好的解决问题。）

如果你正在用负对数似然损失和对数softmax激活函数，那么PyTorch提供一个结合二者的函数`F.cross_entropy` 。所以我们能够从我们的模型中移除激活函数。

```python
import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias
```

注意我们不再调用 `log_softmax`函数。让我们像之前一样确认我们的损失和准确率。

```python
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
```

Out:

```
tensor(0.0824, grad_fn=<NllLossBackward>) tensor(1.)
```

## 使用nn.Module重构

下面，我们用 `nn.Module` 和 `nn.Parameter`使得我们的循环更加简洁清晰。使用`nn.Module`的子类（它本身是一个能够跟踪状态的类）。这个例子中，我们想要创建一个类，包括权重、偏置和前向传递的方法。 `nn.Module` 中的大量属性和方法（比如 `.parameters()` 和 `.zero_grad()`）将被用到。

* 注意

`nn.Module` (大写M)是PyTorch的一个特殊概念，也是我们将大量使用的类。`nn.Module` 不要和Python中module（小写m）混淆，小写是一个Python代码文件，是可以被导入的。

```python
from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias
```

因为我们现在使用的是一个对象，而不是仅仅使用一个函数，我们首先要实例化我们的模型。

```python
model = Mnist_Logistic()
```

现在我们可以像之前一样计算损失。注意`nn.Module`对象被当作函数调用（如他们是可调用的），但其实PyTorch将自动调用 `forward` 方法。

```python
print(loss_func(model(xb), yb))
```

Out:

```
tensor(2.3974, grad_fn=<NllLossBackward>)
```

在之前的训练循环中，我们必须按名称更新每个参数的值，并分别手动将每个参数的梯度归零，如下所示：

```python
with torch.no_grad():
    weights -= weights.grad * lr
    bias -= bias.grad * lr
    weights.grad.zero_()
    bias.grad.zero_()
```

现在我们可以利用`model.parameters()`和`model.zero_grad()` （它们都定义在PyTorch的`nn.Module`)使这些步骤更简洁，并且更不容易忘记我们具体的参数，特别是我们的模型更复杂时：

```python
with torch.no_grad():
    for p in model.parameters(): p -= p.grad * lr
    model.zero_grad()
```

我们将小的训练循环封装在`fit`函数中，所以我们可以在之后运行它。

```python
def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()
```

让我们再检查一下我们的损失是否减少了：

```python
print(loss_func(model(xb), yb))
```

Out:

```
tensor(0.0837, grad_fn=<NllLossBackward>)
```

## 使用nn.Linear重构

我们继续重构我们的代码。代替之前手工定义和初始化 `self.weights` 和 `self.bias`，以及计算`xb  @self.weights + self.bias`，我们将用PyTorch的类 [nn.Linear](https://pytorch.org/docs/stable/nn.html#linear-layers)作为一个线性层，它可以做到我们所有要做的。PyTorch许多预定义好的层可以极大的简化代码，通常也能提高运行速度。

```python
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)
```

我们实例化我们的模型，并用与之前相同的方式计算损失：

```python
model = Mnist_Logistic()
print(loss_func(model(xb), yb))
```

Out:

```
tensor(2.3426, grad_fn=<NllLossBackward>)
```

我们还可以像之前一样使用相同的`fit`方法。

```python
fit()

print(loss_func(model(xb), yb))
```

Out:

```
tensor(0.0824, grad_fn=<NllLossBackward>)
```

## 使用optim重构

PyTorch有一个包含不同优化算法的包`torch.optim`。我们可以用`step`方法进行一步前向传递，代替手工更新每一个参数。

这需要我们替换我们之前手工优化的代码：

```python
with torch.no_grad():
    for p in model.parameters(): p -= p.grad * lr
    model.zero_grad()
```

而是使用：

```python
opt.step()
opt.zero_grad()
```

（`optim.zero_grad()` 重置梯度为0，我们需要在计算下一个minibatch的梯度前调用它。）

```python
from torch import optim
```

我们将定义一个小函数用于创建我们的模型和优化器，这样我们以后可以重复使用它。

```python
def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
```

Out:

```
tensor(2.2520, grad_fn=<NllLossBackward>)
tensor(0.0816, grad_fn=<NllLossBackward>)
```

## 使用Dataset重构

PyTorch有一个抽象的Dataset类。Dataset可以是任何有`__len__`函数（通过python的标准len函数调用）和一个通过`__getitem__`函数作为索引的类。这篇向导完成了一个创建自定义`FacialLandmarkDataset`类，他是`Dataset`的子类。

PyTorch的TensorDataset是一个封装Dataset的数据集。通过定义长度和索引方式，提供给我们遍历、索引和沿着第一维度切片的操作。这使我们更容易在同一行中访问自变量和因变量。

```python
from torch.utils.data import TensorDataset
```

`x_train`和`y_train`可以被结合到一个`TensorDataset`，这将会更容易遍历和切片。

```python
train_ds = TensorDataset(x_train, y_train)
```

之前，我们不得不分别遍历x和y通过minibatch:

```python
xb = x_train[start_i:end_i]
yb = y_train[start_i:end_i]
```

现在，我们一起完成这两步：

```python
xb,yb = train_ds[i*bs : i*bs+bs]
```

```python
model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
```

Out:

```
tensor(0.0802, grad_fn=<NllLossBackward>)
```

## 使用DataLoader重构

PyTorch的`DataLoader`负责管理batch。你可以从任何`Dataset`中创建`DataLoader`。`DataLoader`使得遍历batch更简单。不像之前不得不用`train_ds[i*bs : i*bs+bs]`，DataLoader自动给我们minibatch。

```python
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)
```

之前，我们像这样循环遍历batches(xb,yb)：

```python
for i in range((n-1)//bs + 1):
    xb,yb = train_ds[i*bs : i*bs+bs]
    pred = model(xb)
```

现在，我们的循环更简洁，因为(xb,yb)已经从data loader自动加载：

```python
for xb,yb in train_dl:
    pred = model(xb)
```

```python
model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
```

Out:

```
tensor(0.0819, grad_fn=<NllLossBackward>)
```

由于PyTorch的 `nn.Module`, `nn.Parameter`, `Dataset`, 和`DataLoader`，我们的训练循环现在已经大大的缩减，并且更容易理解。现在让我们尝试增加在实际中创建有效模型需要的基本特性。

## 增加验证

第一节中，我们只是用我们的训练数据尝试做一个合理的训练循环。实际上，你也**必须**有一个验证集，为了验证是否过拟合。

洗乱数据对于防止batch之间的相关性和过拟合是相当重要的。另一方面，不管我们是否洗乱验证集，验证损失都是一样的。由于洗乱是有额外时间开销的，所以洗乱验证数据是没有意义的。

我们将用一个batch size大小是训练集两倍的验证集。这是因为验证集不需要反向传播，因此占用更少内存（不需要存储梯度）。我们利用这个优势，用一个更大的batch并更快的计算损失。

```python
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
```

我们将计算验证损失并在每一个epoch的结尾打印。

（注意我们总是在训练前需要调用`model.train()`，推理前调用`model.eval()`，因为这些用在 `nn.BatchNorm2d` 和 `nn.Dropout` 这样的层，确保不同阶段的适当表现。）

```python
model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))
```

Out:

```
0 tensor(0.2989)
1 tensor(0.3405)
```

## 创建fit()和get_data()

我们现在将做一点重构。由于我们对训练集和验证集做了两次相似的计算损失操作，我们将这些弄到自己的函数`loss_batch`，它计算一个batch的损失。

我们传递给训练集一个优化器，用于进行反向传播。对验证集，因为没有反向传播，我们不需要传递优化器。

```python
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)
```

`fit`运行必要的操作，去训练我们的模型，并对每一个epoch计算训练和验证损失。

```python
import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)
```

`get_data`返回训练集和验证集的dataloader。

```python
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
```

现在，我们整个获得数据加载器和拟合模型的过程能够通过3行代码运行：

```python
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

Out:

```
0 0.32266821167469023
1 0.29154807721972464
```

你可以用这3行代码训练大量不同的模型。让我们看一下是否可以用它训练一个卷积神经网络(CNN)!

## 转换到CNN

我们现在通过3个卷积层建立我们的神经网络。因为之前章节的函数没有对模型有任何假设，我们将不做任何修改训练一个CNN模型。

我们将用PyTorch预训练的Conv2d类作为我们的卷积层。我们定义一个3层卷积的CNN。每一个卷积都在ReLU之后。结尾，我们执行一个平均化池。（注意`view`是numpy的`reshape`的PyTorch版本）

```python
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

lr = 0.1
```

[Momentum](https://cs231n.github.io/neural-networks-3/#sgd) 是随机梯度下降的变化，考虑了之前的更新，通常训练会更快。

```python
model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

Out:

```
0 0.3211389096498489
1 0.24728197736740112
```

## nn.Sequential

`torch.nn`有另外一个方便的类可以用于简化代码：[Sequential](https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential) 。一个`Sequential`对象运行每一个包含在里面的模块以一个序列形式。这是一个简单写神经网络的方式。

利用这个优势，我们需要从一个已有函数定义一个**自定义层**。例如，PyTorch没有view层，我们需要为我们的网络创建一个。`Lambda`将创建一个层，然后我们在通过`Sequential`定义网络时可以用到。

```python
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)
```

模型创建`Sequential`是简单的：

```python
model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

Out:

```
0 0.34388444738388063
1 0.2368821501970291
```

## 封装DataLoader

我们的CNN相当简洁，但是只能用于MNIST，因为：

* 它假设输入是一个28*28的长向量
* 它假设最后的CNN格子是4*4（由于这是我们用的平均池核大小）

让我们去掉这两个假设，因此我们的模型适用于任何二维单通道图像。首先，我们可以移除初始Lambda层，但将数据预处理移动到生成器中。

```python
def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
```

下一步，我们使用 `nn.AdaptiveAvgPool2d`替换 `nn.AvgPool2d` ，它允许我们定义我们想要的输出tensor的大小，而不是我们有的输入tensor。结果是，我们的模型可以运行在任何大小的输入。

```python
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```

让我们输出看一下：

```
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

Out:

```
0 0.42986514909267426
1 0.21305105301141739
```

## 使用你的GPU

如果你幸运能够访问有能力的CUDA GPU(你可以从云服务商租一个大概0.5$/h)你可以用它加速你的代码。首先检查你的GPU工作在PyTorch：

```python
print(torch.cuda.is_available())
```

Out:

```
True
```

然后对它创建一个设备对象：

```python
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
```

让我们更新`preprocess`，移动batch到GPU：

```python
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
```

最后，将我们的模型移到GPU：

```python
model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```

现在你会发现它运行的更快：

```python
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

Out:

```
0 0.26053627222776415
1 0.17719570652544497
```

## 结束语

我们现在有了一个通用的数据pipeline和训练循环，您可以使用Pytorch来训练许多不同类型的模型。为了了解现在训练一个模型有多简单，可以看看*mnist_sample*笔记。

当然，有许多事情你想要去增加，比如数据扩充、超参数调整、监控训练、迁移学习，等等。这些特性可以在fastai库中找到，该库是使用本教程中展示的相同设计方法开发的，为希望进一步开发模型的从业者提供了一个自然的下一步开发。

我们在这个教程的开始已经解释了 `torch.nn`, `torch.optim`, `Dataset`, 和 `DataLoader`。所以我们总结一下我们看到的：

* **torch.nn**
  *  `Module`:创建一个类似函数的可调用对象，但是包含状态（例如神经网络层权重）。它知道包含了什么 `Parameter` 并且能够清零他们所有的梯度，循环更新他们的权重等。
  *  `Parameter` ：一个Tensor的封装器，告诉一个`Module`它有反向传播时需要更新的权重。只有设置*requires_grad*属性的需要更新。
  * `functional`：一个包含激活函数，损失函数等等，以及没有状态版本的层，例如卷积或线性层的模块，（通常导入后使用`F`重命名）。

* `torch.optim`：包含优化器例如`SGD`，在反向传播步骤更新 `Parameter` 的权重。
* `Dataset`：一个包含`__len__` 和 `__getitem__`对象的抽象接口，包括PyTorch提供的类例如 `TensorDataset`
* `DataLoader`：对任何`Dataset`创建一个迭代器返回数据的batch。