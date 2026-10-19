import torch

a = torch.tensor(5)
b = torch.tensor([1, 2, 3])
c = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print(a)
print(b)
print(c)

#这里的 torch.tensor() 可以把 Python 中的数字、列表等数据转换成 Tensor。
#注意：Tensor 不只是一个列表。它还支持高效的数值运算、GPU 计算，以及后续我们要学习的自动求导。


print(a.shape)
print(b.shape)
print(c.shape)

#shape 中有几个数字，Tensor 就有几维。
#每个数字表示对应维度的大小。

x = torch.tensor([1, 2, 3])
y = torch.tensor([1.0, 2.0, 3.0])

print(x.dtype)
print(y.dtype)

#- torch.int64：整数类型。
#- torch.float32：32 位浮点数，可以表示小数。

#为什么要关注它？因为模型训练时，输入特征通常使用浮点数，标签的数据类型则取决于具体任务和损失函数。
#例如，我们之后使用 nn.BCEWithLogitsLoss() 进行二分类时，标签通常也要转换成浮点数。