import os
import torch
from torch.utils.data import Dataset
import numpy as np

class EarDataset(Dataset):
    def __init__(self, data_root="data", split="train"):
        split_dir = os.path.join(data_root, split)
        self.file_list = sorted([
            os.path.join(split_dir, fname)
            for fname in os.listdir(split_dir)
            if fname.endswith(".npy")
        ])
        print(f"[EarDataset] Loaded {len(self.file_list)} files from {split_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])  # (24000, 7)

        coord = data[:, :3]       # x, y, z
        normal = data[:, 3:6]     # nx, ny, nz
        label = data[:, 6]        # class label

        feat = np.concatenate([coord, normal], axis=1)  # (24000, 6)

        return {
            "coord": torch.tensor(coord, dtype=torch.float32),      # (24000, 3)
            "feat": torch.tensor(feat, dtype=torch.float32),        # (24000, 6)
            "label": torch.tensor(label, dtype=torch.long),         # (24000,)
            "batch": torch.zeros(coord.shape[0], dtype=torch.long), # 全0，单文件batch
            "grid_size": 0.02
        }
