# 之前学习的是：
# \[ y=x^2 \]

# 现在来看一个更接近神经网络的例子：
# \[ y=2x+3w \]

# 这里，y 同时依赖变量 x 和 w。我们可以分别计算输出对这两个变量的梯度。

import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(4.0, requires_grad=True)

y = 2 * x + 3 * w

y.backward()

print(x.grad)  # tensor(2.)
print(w.grad)  # tensor(3.)

#y.backward() 会计算 y 对所有相关叶子张量的梯度，并将结果存入它们的 .grad 属性。
#x 和 w 都是叶子张量（leaf tensors），所以这里可以直接查看它们的 .grad。


#理解计算图中的链式法则
x = torch.tensor(3.0, requires_grad=True)

a = 2 * x
y = a ** 2

y.backward()

print(x.grad)  # tensor(24.)

#这就是 Autograd 的核心机制：记录运算之间的依赖关系，再从输出开始反向应用链式法则，计算各个变量的梯度。



#梯度不等于参数更新
w = torch.tensor(4.0, requires_grad=True)
loss = w ** 2

loss.backward()

print(w)       # tensor(4., requires_grad=True)
print(w.grad)  # tensor(8.)
# 调用 backward() 后，w 仍然是 4，梯度为 8。
# 也就是说：
# - backward()：计算梯度。
# - optimizer.step()：使用梯度更新模型参数。
# 两者不是同一件事。后面我们会结合优化器具体学习参数如何更新。

#loss.backward() 的作用是计算梯度并将梯度累积到相应张量的 .grad 属性中，它不会自动更新模型参数。