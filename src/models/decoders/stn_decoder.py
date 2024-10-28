#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : stn_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/29 21:15

# Import lib here
import torch
import torch.nn as nn

from src.models.components.spatial_transformer_network import SpatialTransformerNetwork
from segmentation_models_pytorch import Unet
from typing import Tuple, List


class STNDecoder(nn.Module):
    def __init__(self, input_size: Tuple[int, int, int],
                 out_channels: int = 1,
                 num_stn: int = 1):
        super(STNDecoder, self).__init__()
        in_channels, height, width = input_size

        self.stns = nn.ModuleList([SpatialTransformerNetwork(input_size) for _ in range(num_stn)])

        self.unet = Unet(
            encoder_name='timm-efficientnet-b1',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=out_channels,
            activation='sigmoid'
        )

    def forward(self, x: torch.Tensor, normalize: bool = False):
        if normalize:
            x = (x - 0.5) * 2

        stn_outs = []
        for stn in self.stns:
            x = stn(x)
            stn_outs.append(x)
        logit = self.unet(x)
        return logit


def run():
    pass


if __name__ == '__main__':
    run()
