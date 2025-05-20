# model/point.py

import torch
import spconv.pytorch as spconv
from addict import Dict

class Point(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        code = encode(self.grid_coord, self.batch, depth, order=order)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order)
        inverse[order] = torch.arange(0, code.shape[0], device=order.device)

        self["serialized_code"] = code.unsqueeze(0)
        self["serialized_order"] = order.unsqueeze(0)
        self["serialized_inverse"] = inverse.unsqueeze(0)

    def sparsify(self, pad=96):
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(torch.max(self.grid_coord, dim=0).values, pad).tolist()

        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat([self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1),
            spatial_shape=sparse_shape,
            batch_size=self.batch.max().item() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat


class PointModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
        for name, module in kwargs.items():
            self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            if isinstance(module, PointModule):
                input = module(input)
            elif isinstance(input, Point):
                input.feat = module(input.feat)
                if "sparse_conv_feat" in input.keys():
                    input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(input.feat)
            else:
                input = module(input)
        return input


def offset2batch(offset):
    bincount = torch.diff(torch.cat([offset.new_zeros(1), offset]))
    return torch.arange(len(bincount), device=offset.device).repeat_interleave(bincount)


def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def encode(grid_coord, batch, depth, order="z"):
    """
    Encode grid coordinates into Z-order (Morton) code
    """
    x, y, z = grid_coord.unbind(1)
    code = morton3D(x, y, z, depth)
    return code + batch * (2 ** (depth * 3))


def morton3D(x, y, z, depth):
    x = x & ((1 << depth) - 1)
    y = y & ((1 << depth) - 1)
    z = z & ((1 << depth) - 1)
    answer = 0
    for i in range(depth):
        answer |= ((x >> i) & 1) << (3 * i + 0)
        answer |= ((y >> i) & 1) << (3 * i + 1)
        answer |= ((z >> i) & 1) << (3 * i + 2)
    return answer
