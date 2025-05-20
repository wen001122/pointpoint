import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EarDataset(Dataset):
    def __init__(self, data_root, split="train"):
        self.data_root = os.path.join(data_root, split)
        self.file_list = [f for f in os.listdir(self.data_root) if f.endswith(".npy")]
        self.file_list.sort()
        print(f"[EarDataset] Loaded {len(self.file_list)} files from {self.data_root}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_root, self.file_list[idx])
        data = np.load(data_path)  # (24000, 7)

        coord = data[:, :3]       # x, y, z
        normal = data[:, 3:6]     # nx, ny, nz
        label = data[:, 6]        # semantic label (0 or 1)
        feat = np.concatenate([coord, normal], axis=1)  # shape: (24000, 6)

        return {
            "coord": torch.tensor(coord, dtype=torch.float32),
            "feat": torch.tensor(feat, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "batch": torch.zeros(coord.shape[0], dtype=torch.long),  # 单个 batch
            "grid_size": 0.02
        }
