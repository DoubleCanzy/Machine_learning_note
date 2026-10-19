# 我们让模型判断一组二维数据属于 类别 0 还是类别 1。
# 一、先明确整个项目要做什么
# 假设每个样本有两个特征 \(x_1,x_2\)，当两者之和大于 0 时，标签为 1，否则为 0：
# 我们会生成 1,000 条模拟数据，让神经网络学习这个分类规律。

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split


# ==========================================
# 1. 基础配置
# ==========================================

torch.manual_seed(42)

batch_size = 32
epochs = 30
learning_rate = 0.01


# ==========================================
# 2. 生成模拟数据
# ==========================================

# 1000 条样本，每条样本有 2 个特征
X = torch.randn(1000, 2)

# 根据两个特征之和生成二分类标签
# 标签形状：[1000, 1]
y = ((X[:, 0] + X[:, 1]) > 0).float().unsqueeze(1)

# X.shape = [1000, 2]
# y.shape = [1000, 1]


# ==========================================
# 3. 划分训练集和验证集
# ==========================================

dataset = TensorDataset(X, y)

train_size = 800
val_size = 200

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)


# ==========================================
# 4. 定义神经网络
# ==========================================

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

# 输入：[batch_size, 2]
#隐藏层：16 个神经元  Linear(2, 16) + ReLU()
# 输出：[batch_size, 1]
#二分类只需要一个 logit。BCEWithLogitsLoss 内部会以数值稳定的方式结合 sigmoid 和二元交叉熵计算损失，因此训练时不需要自己先调用 torch.sigmoid()。——这啥，现在还不清楚？

# ==========================================
# 5. 定义损失函数和优化器
# ==========================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)


# ==========================================
# 6. 训练模型
# ==========================================

for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_X, batch_y in train_loader:
        # 前向传播
        logits = model(batch_X)

        # 计算损失
        loss = criterion(logits, batch_y)

        # 清空梯度
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累计损失，按样本数加权
        batch_count = batch_X.size(0)
        total_loss += loss.item() * batch_count
        total_samples += batch_count

    avg_train_loss = total_loss / total_samples


    # ======================================
    # 7. 在验证集上评估
    # ======================================

    model.eval()

    correct = 0
    val_samples = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            logits = model(batch_X)

            # logits >= 0 等价于 sigmoid(logits) >= 0.5
            predictions = (logits >= 0).float()

            correct += (predictions == batch_y).sum().item()
            val_samples += batch_y.numel()

    val_accuracy = correct / val_samples

    print(
        f"Epoch [{epoch + 1:02d}/{epochs}] "
        f"Train Loss: {avg_train_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.2%}"
    )


# ==========================================
# 8. 保存训练后的模型参数
# ==========================================

save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / "binary_classifier.pth"

torch.save(model.state_dict(), save_path)
#state_dict() 保存模型中可训练参数等状态。它不等于完整的模型代码。
#如果需要重新加载模型，需要先定义相同结构的网络，再加载参数：
# model = nn.Sequential(
#     nn.Linear(2, 16),
#     nn.ReLU(),
#     nn.Linear(16, 1),
# )
# state = torch.load(
#     "checkpoints/binary_classifier.pth",
#     map_location="cpu",
#     weights_only=True,
# )
# model.load_state_dict(state)
# model.eval()


print(f"\n模型参数已保存到：{save_path}")


