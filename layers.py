# model/layers.py

import torch
import torch.nn as nn
import torch_scatter
import math
import spconv.pytorch as spconv
from timm.layers import DropPath
from model.point import PointModule, PointSequential


class RPE(nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)
            + self.pos_bnd
            + torch.arange(3, device=coord.device) * self.rpe_num
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        return out.permute(0, 3, 1, 2)


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        enable_rpe=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.patch_size = patch_size
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.enable_rpe = enable_rpe
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if enable_rpe else None

    def forward(self, point):
        B, C = point.feat.shape
        H = self.num_heads
        K = self.patch_size

        order = point.serialized_order[0]
        inverse = point.serialized_inverse[0]
        qkv = self.qkv(point.feat)[order].reshape(-1, K, 3, H, C // H)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.enable_rpe:
            rel = point.grid_coord[order].reshape(-1, K, 3)
            rel = rel.unsqueeze(2) - rel.unsqueeze(1)
            attn = attn + self.rpe(rel)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)[inverse]
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, drop=0.0):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        enable_rpe=False,
    ):
        super().__init__()
        self.norm1 = PointSequential(nn.LayerNorm(channels))
        self.attn = SerializedAttention(
            channels=channels,
            num_heads=num_heads,
            patch_size=patch_size,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            enable_rpe=enable_rpe,
        )
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())
        self.norm2 = PointSequential(nn.LayerNorm(channels))
        self.mlp = PointSequential(
            MLP(in_channels=channels, hidden_channels=int(channels * mlp_ratio), drop=proj_drop)
        )

    def forward(self, point):
        shortcut = point.feat
        point = self.norm1(point)
        point = self.attn(point)
        point.feat = shortcut + self.drop_path(point).feat

        shortcut = point.feat
        point = self.norm2(point)
        point = self.mlp(point)
        point.feat = shortcut + self.drop_path(point).feat
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(self, in_channels, out_channels, stride=2, reduce="max"):
        super().__init__()
        self.stride = stride
        self.reduce = reduce
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, point):
        code = point.serialized_code[0]
        _, cluster, counts = torch.unique(code, sorted=True, return_inverse=True, return_counts=True)
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        new_feat = torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce)

        point_dict = {
            "feat": new_feat,
            "coord": point.coord[head_indices],
            "grid_coord": point.grid_coord[head_indices] >> 1,
            "batch": point.batch[head_indices],
            "serialized_code": point.serialized_code[:, head_indices] >> 3,
            "serialized_order": torch.argsort(point.serialized_code[:, head_indices]),
            "serialized_inverse": torch.argsort(torch.argsort(point.serialized_code[:, head_indices])),
            "serialized_depth": point.serialized_depth - 1,
        }
        return point.__class__(point_dict)


class SerializedUnpooling(PointModule):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.skip_proj = nn.Linear(skip_channels, out_channels)

    def forward(self, point):
        parent = point["pooling_parent"]
        inverse = point["pooling_inverse"]
        point.feat = self.proj(point.feat)
        parent.feat = self.skip_proj(parent.feat)
        parent.feat = parent.feat + point.feat[inverse]
        return parent


class Embedding(PointModule):
    def __init__(self, in_channels, embed_channels):
        super().__init__()
        self.stem = PointSequential(
            spconv.SubMConv3d(
                in_channels, embed_channels, kernel_size=5, padding=1, bias=False, indice_key="stem"
            ),
            nn.BatchNorm1d(embed_channels),
            nn.GELU(),
        )

    def forward(self, point):
        point = self.stem(point)
        return point
