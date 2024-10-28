#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : trans_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/4/3 20:21

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.segformer import SegFormer


class TransEncoder(nn.Module):
    def __init__(self, in_channels: int = 6,
                 out_channels: int = 3,
                 strength_factor: float = 1.):
        super().__init__()
        self.model = SegFormer(
            in_channels=in_channels,
            widths=[64, 128, 256, 512],
            depths=[3, 4, 6, 3],
            all_num_heads=[1, 2, 4, 8],
            patch_sizes=[7, 3, 3, 3],
            overlap_sizes=[4, 2, 2, 2],
            reduction_ratios=[8, 4, 2, 1],
            mlp_expansions=[4, 4, 4, 4],
            decoder_channels=128,
            scale_factors=[32, 16, 8, 4],
            num_classes=out_channels
        )
        self.strength_factor = strength_factor

    def forward(self, host: torch.Tensor, secret: torch.Tensor, normalize: bool = False):
        if normalize:
            host = (host - 0.5) * 2.
            secret = (secret - 0.5) * 2.
        inputs = torch.cat([host, secret], dim=1)
        residual = self.model(inputs)
        container = residual * self.strength_factor + host
        return container.clamp(0, 1)


def run():
    from torchinfo import summary
    trans_encoder = TransEncoder()
    host = torch.rand(4, 3, 256, 256)
    secret = torch.rand(4, 3, 256, 256)
    # labels = torch.randint(0, 1, (4, 256, 256))
    outputs = trans_encoder(host, secret)
    print(outputs.shape)
    # summary(trans_encoder, input_size=(4, 3, 256, 256))
    pass


if __name__ == '__main__':
    run()
