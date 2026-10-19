#本课目标：理解 PyTorch 如何组织数据、按批次读取数据，以及如何把数据送进训练循环。

# 前面训练时，我们直接写了：
# prediction = model(X)
# 这适合很小的数据集。但如果有几十万条样本，通常不会一次把所有数据都送进模型，而是分批读取。这就是 Dataset 和 DataLoader 要解决的问题。

# 可以把它们理解为：
# - Dataset：定义每条数据是什么、怎么取。
# - DataLoader：定义每次取多少条、按什么顺序取。




import torch
from torch.utils.data import TensorDataset, DataLoader

X = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0],
    [9.0, 10.0],
])

y = torch.tensor([0, 1, 0, 1, 0])

dataset = TensorDataset(X, y)

#这里的 TensorDataset(X, y) 会将 X 和 y 按索引配对。
# 例如，索引为 1 的样本是： dataset[1]
# (tensor([3., 4.]), tensor(1))
#非常重要：输入和标签必须保持对应关系。 如果输入数据被打乱了，而标签没有同步打乱，模型就会拿错误的答案训练。

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
)

# batch_size=2	每批最多包含 2 条样本
# shuffle=True	每次遍历数据集时打乱样本顺序

for batch_X, batch_y in loader:
    print(batch_X.shape)
    print(batch_y.shape)

#因为共有 5 条样本，每批最多 2 条，所以会有 3 个 batch。默认 drop_last=False，最后一个 batch 只剩 1 条。




#把 DataLoader 接入训练循环
for epoch in range(10):
    for batch_X, batch_y in loader:
        prediction = model(batch_X)
        loss = criterion(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# 这里有两层循环：
# - 外层 for epoch：控制训练集被完整遍历多少次。
# - 内层 for batch_X, batch_y：逐批读取数据，并完成一次参数更新。
#真实的分类任务还需要根据模型输出形状和损失函数的要求，适当调整标签的 dtype 与形状。例如，BCEWithLogitsLoss 通常要求预测和目标具有匹配的形状，目标使用浮点类型。