# utils.py

import torch
import numpy as np
import random
import os

def compute_iou(pred, label, num_classes=2):
    """
    计算每类IoU和mean IoU

    Args:
        pred: 预测标签 (N,)
        label: 真实标签 (N,)
        num_classes: 类别数

    Returns:
        mean_iou: 所有类别IoU的平均值
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        label_cls = label == cls

        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()

        if union == 0:
            ious.append(float("nan"))  # 忽略这个类
        else:
            ious.append(intersection / union)

    mean_iou = np.nanmean(ious)
    return mean_iou


def seed_everything(seed=42):
    """
    设置随机种子，确保结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
