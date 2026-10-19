best_val_loss = float("inf")

for epoch in range(epochs):
    # 训练阶段
    model.train()
    for batch_X, batch_y in train_loader:
        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证阶段
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

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        torch.save(
            model.state_dict(),
            "best_model.pth",
        )

    print(
        f"Epoch {epoch + 1}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"Best: {best_val_loss:.4f}"
    )



#     这里有两个重要细节：
# - float("inf") 表示正无穷，保证第一次有限的验证损失会小于初始值。
# - best_model.pth 只会在验证损失改善时被覆盖，因此保存的是训练过程中验证损失最低时的模型参数。
# 这段代码实现了保存最佳模型，但还没有实现自动停止。严格意义上的 Early Stopping 通常还会设置一个 patience，即允许验证指标连续多少轮不改善。