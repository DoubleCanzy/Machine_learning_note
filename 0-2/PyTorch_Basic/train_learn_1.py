# 本课目标：把前面学过的张量、自动求导和优化器组合起来，理解一个神经网络是如何反复训练的。
# 1. 为什么需要训练循环？
# 前面我们只更新了一次参数。但真实任务通常有成百上千个样本，需要反复执行：
# 1. 把数据输入模型，得到预测结果。
# 2. 计算预测结果与真实答案之间的损失。
# 3. 反向传播，计算梯度。
# 4. 使用优化器更新参数。
# 5. 重复以上过程，让模型逐渐学会任务。
# 这个反复执行的过程就是训练循环。


# Epoch：训练轮次
# 模型完整遍历一次训练集，称为一个 epoch。如果训练集有 1,000 条数据，训练 10 个 epoch，意味着模型会经历 10 轮完整的数据遍历。

# Batch：批次
# 一次送入模型处理的一小批数据。例如训练集有 1,000 条数据，batch_size=100，那么每轮通常有 10 个 batch。

# Iteration：迭代步骤
# 在常规训练中，一个 batch 通常对应一次参数更新，也就是一次 iteration。因此上面的例子中，每个 epoch 通常有 10 次参数更新。



#下面用一个简单的线性回归任务演示。模型学习从输入 \(x\) 预测目标值 \(y\)。

import torch
from torch import nn

# 1. 准备数据：y = 2x + 1
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# 2. 定义模型：输入 1 个特征，输出 1 个数
model = nn.Linear(1, 1)

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练 100 轮
for epoch in range(100):
    # 前向传播：计算预测结果
    prediction = model(X)

    # 计算损失
    loss = criterion(prediction, y)

    # 清空梯度、反向传播、更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 5. 查看训练后的预测
model.eval()
with torch.no_grad():
    print("预测结果：", model(X))

#     另外，代码中还有两个值得先认识的操作：
# - model.eval()：把模型切换到评估模式。对于线性层本身没有明显影响，但对于 Dropout、BatchNorm 等层很重要。
# - torch.no_grad()：在评估时关闭梯度记录，减少不必要的计算和内存开销。