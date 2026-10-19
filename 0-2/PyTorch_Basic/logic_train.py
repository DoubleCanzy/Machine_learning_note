
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# 1. 固定随机种子，方便复现实验
torch.manual_seed(42)

# 2. 创建模拟数据
# 1000 个样本，每个样本有 2 个特征
X = torch.randn(1000, 2)

# 根据特征构造二分类标签：0 或 1
y = ((X[:, 0] + X[:, 1]) > 0).float().unsqueeze(1)

# 3. 划分训练集和验证集
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size]
)

# 4. 使用 DataLoader 按 batch 加载数据
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False
)

# 5. 定义模型
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

model = BinaryClassifier()

# 6. 定义损失函数和优化器
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 7. 训练模型
epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        # 清除上一步累积的梯度
        optimizer.zero_grad()

        # 前向传播
        logits = model(X_batch)

        # 计算损失
        loss = loss_fn(logits, y_batch)

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    train_loss = total_loss / len(train_dataset)

    # 8. 在验证集上评估
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            predictions = (logits >= 0).float()

            correct += (predictions == y_batch).sum().item()
            total += y_batch.numel()

    accuracy = correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {train_loss:.4f} "
        f"Val Accuracy: {accuracy:.4f}"
    )

# 9. 保存模型参数
import os

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/model.pth")

print("模型已保存到 checkpoints/model.pth")
