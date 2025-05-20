import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import EarDataset
from model.ptv3 import PointTransformerV3
from model.point import Point

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

train_set = EarDataset(data_root="data", split="train")
val_set = EarDataset(data_root="data", split="val")
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        label = batch["label"].to(device)
        for k in batch:
            if k != "label":
                batch[k] = batch[k].to(device)

        point = Point(batch).sparsify()
        out = model(point)
        pred = out.feat

        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_seen = 0

    for batch in dataloader:
        label = batch["label"].to(device)
        for k in batch:
            if k != "label":
                batch[k] = batch[k].to(device)

        point = Point(batch).sparsify()
        out = model(point)
        pred = out.feat.argmax(dim=1)

        total_correct += (pred == label).sum().item()
        total_seen += label.numel()

    return total_correct / total_seen

for epoch in range(1, 101):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_acc = validate(model, val_loader, device)
    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
