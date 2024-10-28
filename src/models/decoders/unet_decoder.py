#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : unet_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/29 10:40

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from torchinfo import summary
from mmseg.models.backbones import UNet
import pdb


class UNetDecoder(nn.Module):
    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1):
        super().__init__()

        self.unet = UNet(in_channels=in_channels, base_channels=32, num_stages=4, strides=(1, 1, 1, 1),
                         enc_num_convs=(2, 2, 2, 2), dec_num_convs=(2, 2, 2), downsamples=(True, True, True),
                         enc_dilations=(1, 1, 1, 1), dec_dilations=(1, 1, 1))
        self.conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, container: torch.Tensor, normalize: bool = False):
        if normalize:
            container = tvf.normalize(container, mean=[0.4413, 0.4107, 0.3727], std=[0.2435, 0.2338, 0.2316])
        return self.conv(self.unet(container)[-1])


def run():
    pass


if __name__ == '__main__':
    run()
