# model/ptv3.py

import torch
import torch.nn as nn
from model.point import PointSequential, Point
from model.layers import Embedding, Block, SerializedPooling, SerializedUnpooling

class PointTransformerV3(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=2,
        enc_channels=(32, 64, 128, 256),
        enc_depths=(2, 2, 2, 2),
        enc_heads=(2, 4, 8, 8),
        patch_size=48,
        mlp_ratio=4.0,
        drop_path=0.1,
    ):
        super().__init__()
        self.embedding = Embedding(in_channels, enc_channels[0])
        self.encoder = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.unpooling = nn.ModuleList()

        # encoder
        for i in range(len(enc_channels)):
            blocks = PointSequential(*[
                Block(
                    channels=enc_channels[i],
                    num_heads=enc_heads[i],
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path
                )
                for _ in range(enc_depths[i])
            ])
            self.encoder.append(blocks)
            if i < len(enc_channels) - 1:
                self.pooling.append(
                    SerializedPooling(
                        in_channels=enc_channels[i],
                        out_channels=enc_channels[i+1],
                        stride=2
                    )
                )

        # decoder
        for i in reversed(range(len(enc_channels) - 1)):
            self.unpooling.append(
                SerializedUnpooling(
                    in_channels=enc_channels[i+1],
                    skip_channels=enc_channels[i],
                    out_channels=enc_channels[i]
                )
            )
            blocks = PointSequential(*[
                Block(
                    channels=enc_channels[i],
                    num_heads=enc_heads[i],
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path
                )
                for _ in range(enc_depths[i])
            ])
            self.decoder.append(blocks)

        # 分类头（point-wise）
        self.cls_head = nn.Sequential(
            nn.Linear(enc_channels[0], enc_channels[0]),
            nn.BatchNorm1d(enc_channels[0]),
            nn.GELU(),
            nn.Linear(enc_channels[0], num_classes)
        )

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order="z", shuffle_orders=False)
        point.sparsify()

        skips = []
        point = self.embedding(point)

        for i in range(len(self.encoder)):
            point = self.encoder[i](point)
            if i < len(self.pooling):
                skips.append(point)
                point = self.pooling[i](point)

        for i in range(len(self.decoder)):
            point["pooling_parent"] = skips[-(i + 1)]
            point["pooling_inverse"] = skips[-(i + 1)].serialized_inverse[0]
            point = self.unpooling[i](point)
            point = self.decoder[i](point)

        out_feat = self.cls_head(point.feat)
        point.feat = out_feat
        return point
