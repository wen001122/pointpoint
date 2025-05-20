import torch
import torch_scatter

class Point:
    def __init__(self, data_dict):
        self.coord = data_dict["coord"]
        self.feat = data_dict["feat"]
        self.label = data_dict.get("label", None)  # 可选字段，兼容测试
        self.batch = data_dict["batch"]
        self.grid_size = data_dict["grid_size"]

        self.grid_coord = (self.coord / self.grid_size).floor().int()
        self.offset = self.batch2offset(self.batch)

    def batch2offset(self, batch):
        bincount = torch.bincount(batch)
        return torch.cumsum(bincount, dim=0).long()

    def sparsify(self):
        grid_coord = self.grid_coord[:, :3].int()
        indices = torch.cat([
            self.batch.unsqueeze(-1).int(),
            grid_coord
        ], dim=1)

        spatial_shape = torch.max(grid_coord, dim=0).values + 1
        sparse_shape = [self.batch.max().item() + 1] + spatial_shape.tolist()

        import spconv.pytorch as spconv
        self.sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=indices,
            spatial_shape=spatial_shape.tolist(),
            batch_size=self.batch.max().item() + 1
        )
        return self
