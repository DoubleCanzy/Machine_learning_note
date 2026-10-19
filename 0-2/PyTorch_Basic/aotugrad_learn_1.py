# 在神经网络中，模型通常包含大量可训练参数。训练时，我们需要知道损失函数对每个参数的梯度，才能调整参数，让损失逐渐减小。
# 如果手动求每个参数的导数，会非常繁琐。PyTorch 的 Autograd 可以自动完成这项工作。

#4.2 第一个核心概念：requires_grad=True
#在 PyTorch 中，我们可以通过 requires_grad=True 告诉系统：这个 Tensor 参与的运算需要被跟踪，以便后续计算梯度。

import torch

x = torch.tensor(3.0, requires_grad=True)

y = x ** 2

print(y)

# 注意两点：
# - requires_grad=True：允许 PyTorch 跟踪与 x 有关的运算。
# - grad_fn：显示这个结果由什么运算产生，是计算图的一部分。
# 这里，PyTorch 记录了 y = x ** 2 这项运算，后续就能计算 \(y\) 对 \(x\) 的导数。

y.backward()

print(x.grad)

# 代码	作用
# requires_grad=True	让 PyTorch 跟踪相关计算
# y.backward()	从 y 开始反向计算梯度
# x.grad	查看计算得到的、关于 x 的梯度

#这里 y 是一个标量，所以可以直接调用 backward()。对于非标量输出，通常需要提供梯度参数，或者先将输出聚合为标量。