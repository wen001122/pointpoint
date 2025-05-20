import torch

class Point:
    def __init__(self, data_dict):
        self.coord = data_dict["coord"]
        self.feat = data_dict["feat"]
        self.label = data_dict.get("label", None)
        self.batch = data_dict["batch"]
        self.grid_size = data_dict["grid_size"]

        self.grid_coord = (self.coord / self.grid_size).floor().int()
        self.offset = self.batch2offset(self.batch)

    def batch2offset(self, batch):
        batch_cpu = batch.view(-1).to("cpu")  # 转成CPU 1D
        bincount = torch.bincount(batch_cpu)
        return torch.cumsum(bincount, dim=0).long()

    def sparsify(self):
        grid_coord = self.grid_coord[:, :3].int()
        batch_column = self.batch.view(-1, 1).int()
        indices = torch.cat([batch_column, grid_coord], dim=1)

        spatial_shape = torch.max(grid_coord, dim=0).values + 1

        import spconv.pytorch as spconv
        self.sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=indices,
            spatial_shape=spatial_shape.tolist(),
            batch_size=self.batch.max().item() + 1
        )
        return self
