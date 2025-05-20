import torch
import numpy as np
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_iou(pred, label, num_classes=2):
    """
    计算每类 IoU 的平均值（用于二分类语义分割）
    pred, label: Tensor, shape=(N,)
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        label_cls = (label == cls)
        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()
        if union == 0:
            ious.append(1.0)  # 如果该类在预测和标签中都不存在，认为 IoU=1
        else:
            ious.append(intersection / union)
    return sum(ious) / len(ious)
