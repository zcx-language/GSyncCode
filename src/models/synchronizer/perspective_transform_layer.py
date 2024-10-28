#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : perspective_transform_layer.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/6/14 10:51

# Import lib here
import torch
import torch.nn as nn
from kornia.geometry.transform import warp_perspective
from fastai.layers import ResBlock, ConvLayer
from typing import Tuple


class PerspectiveTransformLayer(nn.Module):
    def __init__(self, in_channels: int, output_size: Tuple[int, int]) -> None:
        super().__init__()

        self.output_size = output_size

        # Regressor of theta
        self.regressor = nn.Sequential(
            ResBlock(1, ni=in_channels, nf=32, stride=2, ks=5),
            ResBlock(1, ni=32, nf=64, stride=2, ks=5),
            ResBlock(1, ni=64, nf=64, stride=2, ks=5),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 9),
        )

        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32))

    def forward(self, image: torch.Tensor,
                normalize: bool = False):
        if normalize:
            image = (image - 0.5) * 2.
        theta = self.regressor(image).reshape(-1, 3, 3)
        warped = warp_perspective(image, theta, dsize=tuple(self.output_size))
        return warped


def run():
    pass


if __name__ == '__main__':
    run()
