import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])


#逐元素计算
print(a + b)
print(a - b)
print(a * b)
print(a / b)

#这里的 +、-、*、/ 都是逐元素运算：将两个 Tensor 对应位置的元素分别进行计算。
#例如：
#\[
#[1,2,3]\times[4,5,6]=[1\times4,2\times5,3\times6]
#\]
#注意：Tensor 的 * 通常不是矩阵乘法，而是逐元素相乘。


#矩阵乘法
#矩阵乘法使用 @ 或 torch.matmul()。

a = torch.tensor([
    [1, 2],
    [3, 4]
])

b = torch.tensor([
    [5, 6],
    [7, 8]
])

print(a @ b)


#为什么这对神经网络很重要？
#神经网络中的全连接层 nn.Linear，核心计算可以理解为：
#\[ Y=XW^T+b \]
#- \(X\)：输入特征矩阵。
#- \(W\)：模型需要学习的权重。
#- \(b\)：偏置。
#- \(Y\)：模型输出。
#这也是为什么我们需要理解 Tensor 的形状和矩阵乘法：输入特征的维度必须与模型权重的维度匹配。


layer = torch.nn.Linear(2, 16)
x = torch.randn(32, 2)

y = layer(x)
print(x.shape)
print(y.shape)
# torch.Size([32, 16])