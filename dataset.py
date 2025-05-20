import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EarDataset(Dataset):
    def __init__(self, data_root="data", split="train"):
        self.split = split
        self.data_root = os.path.join(data_root, split)  # e.g. data/train
        self.file_list = [
            os.path.join(self.data_root, f)
            for f in os.listdir(self.data_root)
            if f.endswith(".npy")
        ]
        print(f"[EarDataset] Loaded {len(self.file_list)} files from {self.data_root}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])  # (24000, 7)
        coord = data[:, :3]
        normal = data[:, 3:6]
        label = data[:, 6]
        feat = np.concatenate([coord, normal], axis=1)  # shape: (24000, 6)

        return {
            "coord": torch.tensor(coord, dtype=torch.float32),
            "feat": torch.tensor(feat, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "batch": torch.zeros(coord.shape[0], dtype=torch.long),  # 单个 batch
            "grid_size": 0.02,
        }
