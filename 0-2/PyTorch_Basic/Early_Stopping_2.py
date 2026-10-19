import torch

epochs = 100
patience = 5

best_val_loss = float("inf")
epochs_without_improvement = 0   #作为计时器

for epoch in range(epochs):
    # 1. 训练阶段
    model.train()

    for batch_X, batch_y in train_loader:
        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 2. 验证阶段
    model.eval()

    total_val_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            n = batch_X.size(0)
            total_val_loss += loss.item() * n
            total_samples += n

    avg_val_loss = total_val_loss / total_samples

    # 3. 判断验证损失是否改善
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0

        torch.save(model.state_dict(), "best_model.pth")
    else:
        epochs_without_improvement += 1

#每次验证损失刷新最低值，就保存当时的模型参数。这样即使后续训练发生过拟合，也可以找回之前表现更好的版本。


    print(
        f"Epoch {epoch + 1}: "
        f"val_loss={avg_val_loss:.4f}, "
        f"best={best_val_loss:.4f}, "
        f"patience_count={epochs_without_improvement}"
    )

    # 4. 判断是否早停
    if epochs_without_improvement >= patience:
        print("Early stopping!")
        break

# 5. 恢复验证损失最低时的模型参数
model.load_state_dict(
    torch.load("best_model.pth", map_location="cpu", weights_only=True)
)
#这里的 map_location="cpu" 假设你使用 CPU 训练或希望把模型加载到 CPU。如果模型在 GPU 上训练并且你希望继续在 GPU 上使用，需要相应调整设备。