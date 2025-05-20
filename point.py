import torch
import torch.nn as nn
import spconv.pytorch as spconv

class Point:
    def __init__(self, data_dict):
        self.coord = data_dict["coord"]
        self.feat = data_dict["feat"]
        self.label = data_dict["label"]
        self.batch = data_dict["batch"]
        self.grid_size = data_dict["grid_size"]
        self.grid_coord = self.voxelize(self.coord, self.grid_size)
        self.offset = self.batch2offset(self.batch)

    def voxelize(self, coord, grid_size):
        return torch.floor(coord / grid_size).int()

    def batch2offset(self, batch):
        bincount = torch.bincount(batch)
        return torch.cumsum(bincount, dim=0).long()

    def sparsify(self):
        # 注意：只使用前三维 grid_coord（排除 normal）
        grid_coord = self.grid_coord[:, :3].int()

        # 拼接 batch 维度作为稀疏张量索引
        indices = torch.cat([
            self.batch.unsqueeze(-1).int(),
            grid_coord
        ], dim=1)

        # 构造 spconv 稀疏张量
        sparse_shape = torch.max(grid_coord, dim=0).values + 1
        sparse_shape = [self.batch.max().item() + 1] + sparse_shape.tolist()

        sparse_tensor = spconv.SparseConvTensor(
            features=self.feat,
            indices=indices,
            spatial_shape=sparse_shape,
            batch_size=self.batch.max().item() + 1
        )

        self.sparse_shape = sparse_shape
        self.sparse_conv_feat = sparse_tensor

        return self
