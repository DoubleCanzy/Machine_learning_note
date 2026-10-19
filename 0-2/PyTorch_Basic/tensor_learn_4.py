import torch


#广播机制
a = torch.tensor([1, 2, 3])
b = torch.tensor(10)

print(a.shape)
print(b.shape)
print(a + b)

#从两个张量形状的最右侧开始比较，每一维必须满足以下条件之一：
#1. 两个维度相等。
#2. 其中一个维度是 1。
#3. 其中一个张量已经没有对应的维度，可以视为在前面补 1。

a = torch.tensor([1, 2, 3])  # shape: (3,)
b = torch.tensor(10)         # shape: ()

c = a + b
print(c.shape)  # torch.Size([3])


a = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])  # shape: (2, 3)

b = torch.tensor([10, 20, 30])  # shape: (3,)

print(a + b)

#因为就像剥洋葱一样，从外到内，所以说是不够的在左边加上1维度，让维度匹配

#如果你的目的是让 b 中的两个值分别加到矩阵的两行上，需要把 b 变成 (2, 1)：
a = torch.randn(2, 3)
b = torch.randn(2)
c = a + b.unsqueeze(1)
print(c.shape)  # torch.Size([2, 3])S
#unsqueeze(1) 的作用是增加一个长度为 1 的维度：
#(2,) → (2, 1)






#nn.Linear：神经网络的线性层
#nn.Linear 是 PyTorch 提供的一个神经网络层，用来对输入数据进行线性变换（实际包含偏置时是仿射变换）

import torch
import torch.nn as nn

layer = nn.Linear(in_features=3, out_features=2)

x = torch.tensor([1.0, 2.0, 3.0])

y = layer(x)

print(y.shape)

#输出torch.Size([2])——因为 nn.Linear 将输入的 3 个特征映射到 2 个输出特征。
#查看权重就明白为什么输入3输出2了，权重的形状是(out_features, in_features)
layer = nn.Linear(3, 2)

print(layer.weight.shape)
print(layer.bias.shape)

#可以手动设定
import torch
import torch.nn as nn

layer = nn.Linear(3, 2)

with torch.no_grad():
    layer.weight.copy_(
        torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
    )

    layer.bias.copy_(torch.tensor([0.5, 1.0]))

x = torch.tensor([1.0, 2.0, 3.0])

y = layer(x)

print(y)
#补充说明：torch.no_grad() 用来暂时关闭这段代码中的梯度跟踪，方便我们手动设置参数，不让这些赋值操作进入梯度计算图。

#广播和 nn.Linear 有什么关系？
# 前面两个概念其实会在同一个线性层中同时出现。
# 看公式：
# \[
# y=xW^T+b
# \]

# 当输入有多个样本时：
# - \(xW^T\) 是矩阵乘法。
# - \(b\) 是偏置向量。
# - 偏置向量 \(b\) 会通过广播加到每个样本的输出上。

# 矩阵乘法负责把输入特征转换成输出特征。
# 广播负责让形状兼容的张量参与逐元素运算，例如将偏置加到每个样本上。

#nn.Linear 做的是仿射变换，nn.ReLU 引入非线性。