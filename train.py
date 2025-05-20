import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.ptv3 import PointTransformerV3
from dataset import EarDataset
from utils import seed_everything, calculate_iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)

# 模型定义（纯硬编码）
model = PointTransformerV3(
    in_channels=6,
    num_classes=2,
    enc_channels=[32, 64, 128, 256],
    enc_depths=[2, 2, 2, 2],
    enc_heads=[2, 4, 8, 8],
    patch_size=48,
    mlp_ratio=4.0,
    drop_path=0.1
).to(device)

# 数据加载器
train_set = EarDataset(data_root="data", split="train")
val_set = EarDataset(data_root="data", split="val")
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练单个 epoch
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        label = batch.pop("label").to(device)        # (N,)
        for k in batch:
            batch[k] = batch[k].to(device)           # 不要 unsqueeze！

        out = model(batch)
        pred = out.feat                              # (N, C)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 验证函数
def validate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            label = batch.pop("label").to(device)
            for k in batch:
                batch[k] = batch[k].to(device)
            out = model(batch)
            pred = torch.argmax(out.feat, dim=1)
            preds.append(pred.cpu())
            labels.append(label.cpu())
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    iou = calculate_iou(preds, labels, num_classes=2)
    return iou

# 开始训练
best_iou = 0.0
for epoch in range(1, 81):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_iou = validate(model, val_loader, device)
    print(f"[Epoch {epoch}/80] Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "best_model.pt")
        print(f"✅ Saved new best model (IoU = {val_iou:.4f})")

