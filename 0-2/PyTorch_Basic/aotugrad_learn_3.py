# loss.backward() 的作用是计算梯度并将梯度累积到相应张量的 .grad 属性中，它不会自动更新模型参数。
# 一次典型的训练步骤包含三个操作：

# optimizer.zero_grad()  # 清空上一次的梯度
# loss.backward()       # 计算当前梯度
# optimizer.step()      # 根据梯度更新参数

# 这三个操作各司其职。接下来我们学习第三步背后的机制：优化器如何根据梯度更新参数。

# 我们的目标是让损失 \(L\) 尽可能小。
# - 梯度告诉我们参数朝哪个方向变化会让损失增大或减小。
# - 优化器根据梯度和学习率，决定怎样更新参数。

#见笔记

#PyTorch 提供了现成的优化器，例如 torch.optim.SGD 和 torch.optim.Adam。先用 SGD 理解最基本的过程。

import torch

# 一个可训练的参数
w = torch.tensor(2.0, requires_grad=True)

# 输入和真实值
x = torch.tensor(2.0)
y = torch.tensor(6.0)

# 创建优化器
optimizer = torch.optim.SGD([w], lr=0.1)
#学习率 lr 控制每次参数更新的幅度。
#学习率过小，训练可能很慢；学习率过大，参数可能越过合适的位置，甚至导致训练不稳定。注意，表格展示的是一次更新，不是整个训练过程的结果。


# 前向计算
prediction = w * x
loss = (prediction - y) ** 2

# 反向传播：计算梯度
#这里的梯度不是y关于x的梯度，而是损失函数关于参数 w 的梯度。
optimizer.zero_grad()
loss.backward()

print("更新前的梯度：", w.grad.item())

# 更新参数
optimizer.step()

print("更新后的参数：", w.item())