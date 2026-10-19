# 前面你已经完成了一个二分类训练脚本。现在我们解决一个实际问题：
# 模型训练结束、程序退出之后，如何重新加载模型，并用它预测一条新数据？
# 这一课只聚焦于模型推理（Inference），暂时不引入新的训练算法。



#这里保存的是模型的参数状态，而不是完整的模型结构。因此，重新加载时必须先创建相同结构的模型：
import torch
from torch import nn

# 1. 创建与训练时相同结构的模型
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

# 2. 加载训练好的参数
state = torch.load(
    "checkpoints/binary_classifier.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state)

# 3. 切换到评估模式
model.eval()


#这里最重要的是：模型结构必须与保存参数时一致。 如果层数、神经元数量等结构不匹配，通常就无法正确加载参数。
#另外，"checkpoints/binary_classifier.pth" 是相对于当前工作目录的路径，需要确保文件确实存在。
x_new = torch.tensor([[0.8, 0.5]])

with torch.no_grad():
    logits = model(x_new)
    probability = torch.sigmoid(logits)
    prediction = (probability >= 0.5).long()

print("Logit:", logits.item())
print("Probability:", probability.item())
print("Predicted class:", prediction.item())