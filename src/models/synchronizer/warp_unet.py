#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : warp_unet.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/7/22 10:28

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import Unet
from src.models.components.binarized_layer import BinarizedLayer

from typing import Tuple


class WarpUNet(nn.Module):
    def __init__(self, output_size: Tuple[int, int], binarized: bool = False):
        super().__init__()
        self.output_size = output_size
        self.unet = Unet(encoder_name="resnet18",
                         encoder_weights="imagenet",
                         in_channels=1,
                         classes=1,
                         activation=None)
        self.binarized = binarized
        self.binarized_layer = BinarizedLayer()

    def forward(self, image: torch.Tensor,
                normalize: bool = False):
        if normalize:
            image = (image - 0.5) * 2.
        if self.binarized:
            image = self.binarized_layer(image)
        warped = self.unet(image)
        return F.interpolate(warped, size=tuple(self.output_size), mode='bilinear')


def run():
    pass


if __name__ == '__main__':
    run()
