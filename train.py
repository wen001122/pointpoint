# train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.ptv3 import PointTransformerV3
from dataset import EarDataset
from model.point import Point
from utils import compute_iou, seed_everything


# import yaml
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        label = batch.pop("label").to(device)
        for k in batch:
            batch[k] = batch[k].to(device)

        out = model(batch)  # Point对象
        pred = out.feat  # (N, 2)

        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    all_pred = []
    all_label = []
    with torch.no_grad():
        for batch in dataloader:
            label = batch.pop("label").to(device)
            for k in batch:
                batch[k] = batch[k].to(device)

            out = model(batch)
            pred = out.feat.argmax(dim=1)

            all_pred.append(pred.cpu())
            all_label.append(label.cpu())

    all_pred = torch.cat(all_pred)
    all_label = torch.cat(all_label)
    iou = compute_iou(all_pred, all_label, num_classes=2)
    return iou

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    train_set = EarDataset(data_root="data", split="train")
    val_set = EarDataset(data_root="data", split="val")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # 模型
    # 模型
    model = PointTransformerV3(in_channels=6, num_classes=2).to(device)  # 坐标 + 法向量共 6 维


    # 优化器 & 损失
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_iou = 0.0
    for epoch in range(1, 81):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_iou = validate(model, val_loader, device)

        print(f"[Epoch {epoch}/80] Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

        # 保存模型
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_model.pt")
            print(f"✅ New best model saved at epoch {epoch} with IoU {val_iou:.4f}")

if __name__ == "__main__":
    main()
