# dataset.py

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class EarDataset(Dataset):
    def __init__(self, data_root, split="train"):
        self.data_dir = Path(data_root) / split
        self.files = sorted(list(self.data_dir.glob("*.npy")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])  # shape: (24000, 7)
        coord = data[:, :3]              # x, y, z
        normal = data[:, 3:6]            # nx, ny, nz
        label = data[:, 6]               # semantic label (0 or 1)

        # 合并坐标和法向量作为输入特征
        feat = np.concatenate([coord, normal], axis=1)  # shape: (24000, 6)

        return {
            "coord": torch.tensor(coord, dtype=torch.float32),
            "feat": torch.tensor(feat, dtype=torch.float32),     # 6维输入
            "label": torch.tensor(label, dtype=torch.long),
            "batch": torch.zeros(coord.shape[0], dtype=torch.long),  # 单个 batch
            "grid_size": 0.02
        }
