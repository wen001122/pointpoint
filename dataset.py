import torch
import numpy as np
from torch.utils.data import Dataset
import os

class EarDataset(Dataset):
    def __init__(self, data_root="data", split="train"):
        self.data_root = data_root
        self.split = split
        self.file_list = [
            f for f in os.listdir(data_root) if f.endswith(".npy") and split in f
        ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_root, self.file_list[idx])
        data = np.load(file_path)  # shape: (24000, 7) → [x, y, z, nx, ny, nz, label]

        coord = data[:, :3]      # x, y, z
        normal = data[:, 3:6]    # nx, ny, nz
        label = data[:, 6]       # 0 or 1

        feat = np.concatenate([coord, normal], axis=1)  # shape: (24000, 6)

        return {
            "coord": torch.tensor(coord, dtype=torch.float32),       # ✅ 用于 grid
            "feat": torch.tensor(feat, dtype=torch.float32),         # ✅ 模型输入特征
            "label": torch.tensor(label, dtype=torch.long),          # ✅ 分割标签
            "batch": torch.zeros(coord.shape[0], dtype=torch.long),  # ✅ 单个 batch 全 0
            "grid_size": 0.02
        }
