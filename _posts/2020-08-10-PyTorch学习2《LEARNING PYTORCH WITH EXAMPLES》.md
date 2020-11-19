---
layout:     post
title:      PyTorch学习2《LEARNING PYTORCH WITH EXAMPLES》
subtitle:   通过例子学习PyTorch
date:       2020-08-10
author:     Zhuoran Li
header-img: img/post-bg-pytorch.jpg
catalog: true
tags:
- PyTorch
---

这篇向导介绍了PyTorch的基本概念

PyTorch包含两个主要的特征：

* 一个n维的Tensor，类似于Numpy，但是能够运行在GPU

* 自动求导，用于建立和训练神经网络

我们将用一个全联接的ReLU网络作为我们的运行示例。这个网络有一个隐藏层，通过梯度下降法训练，通过**最小化网络输出和真实输出的欧式距离**拟合数据。

## Tensors 

### Warm-up: numpy 热身

我们首先使用numpy实现一个网络。

numpy提供一个n维数组以及许多操作数组的函数。它是一个科学计算的通用框架，但它不能用于图计算、深度学习以及梯度。但是，我们可以用numpy拟合一个两层的网络，该网络基于随机的数据，通过numpy的操作手工实现网络的前向和后向操作：

```python
# -*- coding: utf-8 -*-
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# N 是 batch size；D_in 是输入数据的数目
# H 是 隐藏层维度；D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
# 随机建立输入和输出数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
# 随机初始化权重
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    # 前向：预测y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    # 计算并打印 loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    # 后向：计算loss对w1和w2的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```
### PyTorch: Tensors

Numpy是一个很强的框架，但是它不能利用GPU加速计算。对于现代的深度神经网络，GPU经常能提供50倍以上的加速，所以numpy并不满足现代的深度学习。

这里我们介绍最基础的PyTorch概念：**Tensor**。Tensor类似于numpy：一个tensor是一个n维数组，PyTorch也提供许多对tensor的操作函数。从表面看，Tensor可以跟踪图计算和迭代，但是他们作为科学计算工具也是很有用的。

不像numpy，PyTorch Tensor能够利用GPU加速计算。为了在GPU上运行Tensor，要将numpy转化成一个新的数据类型。

这里我们基于随机数据用Tensor拟合一个两层网络。就像numpy示例一样，我们需要手动实现网络的前向和后向传递：

```python
 -*- coding: utf-8 -*-

import torch


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU
								 									 # 不加注释在GPU上运行

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```


## 自动求导

### PyTorch：Tensors 和 自动求导

以上示例中，我们都是手工实现神经网络的前向和后向传递。手工实现后向传递在简单的两层神经网络中不是问题，但是对于复杂的多层网络很容易就会变得非常难。

幸运的是，我们能够自动计算神经网络的后向传递。PyTorch的autograd包正好提供这个功能。当使用自动求导，前向传递将定义一个**计算图**，节点都是Tensor，边是一个函数，输入和输出都是Tensor。反向传播通过这个图可以方便的计算出梯度。

听起来很不好理解，实际用起来很简单。计算图中，每一个Tensor表示一个节点。如果`x `是一个Tensor并且 `x.requires_grad=True` ，然后`x.grad` 是另外一个包含`x`梯度的由标量构成的Tensor。

**这里用PyTorch的Tensor和autograd实现一个两层的神经网络，不再需要手工实现神经网络的后向传递。**

```python
# -*- coding: utf-8 -*-
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU
								  # 不注释表示在GPU运行

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# N 是 batch size；D_in 是输入数据的数目
# H 是 隐藏层维度；D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
# 随机建立输入和输出数据在Tensor中
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
# 设置 requires_grad=False 表示在后向传递过程我们不需要计算这些梯度
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
# 随机创建权重在Tensor中
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
# 设置 requires_grad=True 表示在后向传递过程我们需要计算这些梯度
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    # 前向传递：使用Tensor的操作计算预测值y，Tensor的操作与我们之前定义的操作都相同，但是我们不需要保留中间值，因为我们没有手工实现后向传递。
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Tensors.
    # 使用Tensor的操作计算损失函数并打印
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    # 现在损失是一个(1,)的Tensor，loss.item()可以得到损失里的标量值。
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # 使用自动求导计算后向传递。这个调用将计算 requires_grad=True 的Tensor的损失梯度。
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    # 调用后， w1.grad 和 w2.grad 分别是带有w1和w2的梯度的Tensor。
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # 使用梯度下降手动更新权重。在 torch.no_grad() 语句下，因为权重设置 requires_grad=True，但是我们在自动求导时不需要跟踪它。
    # An alternative way is to operate on weight.data and weight.grad.data.
    # 一个可选的方式是操作 weight.data 和 weight.grad.data 。
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # 调用 tensor.data 得到一个和tensor共享存储值的tensor，但是不跟踪历史。
    # You can also use torch.optim.SGD to achieve this.
    # 你也可以使用 torch.optim.SGD 实现这个操作。
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        # 更新完权重后手工对梯度清零
        w1.grad.zero_()
        w2.grad.zero_()
```

### PyTorch: 定义新的自动求导函数

其实，每一个自动求导操作都是两个对Tensor的函数。**forward**函数根据输入计算输出Tensor。**backward**函数得到关于标量值的输出Tensor，并计算输入Tensor的关于标量值的梯度。

PyTorch可以简单的自定义自动求导操作，通过定义一个`torch.autograd.Function`的子类，并实现 `forward` 和 `backward` 函数。然后我们可以使用它，通过构造一个实例，像调用函数一样调用它，将包含输入数据的Tensor传递进去。

下面的示例，我们定义了自己的自动求导函数，用于ReLU非线性函数，并用它实现了我们的两层网络。

```python
 -*- coding: utf-8 -*-
import torch


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
	"""
	我们能够通过torch.autograd.Function子类并实现前向、后向传递，实现自己的自动求导函数。
	"""
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        """
        前向传递接收一个输入Tensor，返回一个输出Tensor。ctx是一个上下文对象，可以为后向计算存储信息。通过ctx.save_for_backward方法可以缓存任何在后向传递用到的对象。
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        """
        后向传递我们接收一个包含loss关于输出的梯度Tensor，需要计算loss关于输入的梯度值。
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    # 为了应用我们的函数，我们用Function.apply方法，重新命名为relu。
    relu = MyReLU.apply

    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    # ReLU 使用我们的自动求导操作。
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
```

## nn module

### PyTorch: nn

计算图和autograd是一个定义复杂算子和获取导数的非常典型的示例。然而对于大型神经网络，autograd可能太低级了。

当建立神经网络时，我们通常考虑将计算分为若干层，其中一些层有一些在学习期间可被优化的学习参数。

在TensorFlow，类似 [Keras](https://github.com/fchollet/keras), [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)和 [TFLearn](http://tflearn.org/) 包提供对神经网络有用的比计算图高级的概念。

在PyTorch中，nn包有相同的用途。nn包定义了一系列**Modules**，大概等同于神经网络的层。一个Module接收输入Tensor并计算输出Tensor，但也可以保持内部状态例如可学习参数。nn包也定义了一系列有用的常用于训练神经网络的损失函数。

这个示例中，我们用nn包实现了两层神经网络。

```python
# -*- coding: utf-8 -*-
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# 使用nn包定义我们的模型为一个层序列。nn.Sequential是一个包含其他模块的模块，并会按顺序产生输出。每个线性模块利用线性函数产生输出，并保存权重和偏置内部Tensor。
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
# nn包也包含了常用的损失函数。这个示例中，我们使用MSE损失函数。
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    # 前向传递： 通过传递给模型的x计算预测y。Module对象会重写__call__操作，因此你可以像调用函数一样调用他们。当你这样做时，你传递一个输入Tensor并产生一个输出Tensor。
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    # 计算打印损失。传递包含预测和真值的Tensor，返回一个包含loss的Tensor。
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    # 运行后向传递前清零梯度
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    # 后向传递： 计算所有关于学习参数的梯度。每个带有requires_grad=True的模块参数都会存储在Tensor，所以这个调用会计算模型内所有可学习参数的梯度。
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    # 使用梯度下降更新权重。每一个参数是一个Tensor，所以我们可以像之前一样访问它的梯度。
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
```

### PyTorch：optim

目前为止，我们已经通过手工的方式更新了模型权重，即手工改变可学习参数Tensor（使用`torch.no_grad()` 或 `.data`避免autograd中跟踪历史 ）。虽然对于简单的优化算法随机梯度下降不是什么大负担，但实际上我们经常用更优化的优化器训练神经网络，比如AdaGrad，RMSProp，Adam，等等。

optim包基于常用的优化算法理论进行了实现。

在这个示例中，我们像之前一样用nn包定义我们的模型，但是我们将用optim包提供的Adam算法优化模型。

```python
# -*- coding: utf-8 -*-
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
# 使用optim包定义一个更新模型权重的优化器。这里我们将用Adam；optim包含很多其他的优化算法。Adam构造函数的第一个参数是应当被更新的Tensor。
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    # 后向传递前，用优化器清零用于更新的梯度（即模型的可学习权重）。这是因为默认情况下，梯度当调用.backward()时是累积在缓存中（不是覆盖）。更多细节可查看torch.autograd.backward。
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    # 调用优化器的step函数更新参数
    optimizer.step()
```

### PyTorch：自定义的nn模块

有时你想要比现有的序列模型更复杂的模型，这种情况你需要定义自己的模块作为nn.Module的子类，并且使用其他的modules或其他的autograd操作，定义一个接收输入Tensor产生输出Tensor的前向传递。

这个示例中，我们实现一个两层的网络作为一个自定义Module的子类。

```python
# -*- coding: utf-8 -*-
import torch


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        """
        构造器中实例化两个nn.Linear模型，并分配他们作为成员变量。
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        """
        前向函数接收一个输入数据Tensor，必须返回一个输出Tensor。我们能够用构造器中的模块作为Tensor的操作。
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
# 通过实例化定义的类构造我们的模型
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
# 构造我们的损失函数和优化器。SGD中调用model.parameters()将包含两个nn.Linear成员函数的可学习参数。
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    # 清零梯度，执行一个后向传递，更新权重。
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### PyTorch：控制流+共享权重

作为一个动态图和共享权重的示例，我们实现一个非常陌生的模型：一个全连接的ReLU网络，每一个前向传递随机从1-4选择一个数，并使用许多隐藏层，重复使用相同的权重多次计算内部的隐藏层。

对于这个模型我们能够用Python流控制实现循环，并且可以在定义前向传递时通过重复多次相同的模块在内部层实现权重共享。

我们可以容易的实现这个模型，作为一个Module的子类。

```python
# -*- coding: utf-8 -*-
import random
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        """
        在构造器构造三个nn.Linear实例，我们将在前向传递中使用。
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        """
        对前向传递，我们随机选择0，1，2，3然后重复middle_linear多次计算隐藏层表示。
        由于每一个前向传递建立一个动态计算图，当定义前向传递时，我们能够用Python控制流操作，如循环或条件语句。
        这里我们也看到当定义一个计算图时，这样重复相同的Module很多次是很安全的。
        这对于Lua Torch每个Module只能用一次是一个很大的提升。
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
# 通过实例化以上定义的类构造模型
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
# 构造我们的损失和优化器。使用vanilla随机梯度下降训练陌生的模型是困难的，所以我们用momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```