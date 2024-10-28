#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ScreenShootResilient
# @File         : stegastamp_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/11/27 21:05
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.basic_blocks import ConvBNReLU
from src.models.freq_transform.haar import Haar, InverseHaar
from fastai.layers import SEBlock


class HaarDownsampling(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.haar = Haar(in_channels)
        self.conv = nn.Conv2d(in_channels*4, in_channels, 1)

    def forward(self, x: torch.tensor):
        x = self.haar(x)
        x = self.conv(x)
        return x


class HaarUpsampling(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, in_channels*4, 1)
        self.inverse_haar = InverseHaar(in_channels*4)

    def forward(self, x: torch.tensor):
        x = self.conv(x)
        x = self.inverse_haar(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        self.se_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels, in_channels // reduction, 1),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        return x * self.se_layers(x)


class StegaStampEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, haar_sampling: bool = False):
        super().__init__()

        self.conv1 = ConvBNReLU(in_channels, 32, 3)

        # Down sampling
        if not haar_sampling:
            self.conv2 = ConvBNReLU(32, 32, 3, stride=2)
            self.conv3 = ConvBNReLU(32, 64, 3, stride=2)
            self.conv4 = ConvBNReLU(64, 128, 3, stride=2)
            self.conv5 = ConvBNReLU(128, 256, 3, stride=2)
        else:
            self.conv2 = nn.Sequential(
                ConvBNReLU(32, 32, 3),
                HaarDownsampling(32)
            )
            self.conv3 = nn.Sequential(
                ConvBNReLU(32, 64, 3),
                HaarDownsampling(64)
            )
            self.conv4 = nn.Sequential(
                ConvBNReLU(64, 128, 3),
                HaarDownsampling(128)
            )
            self.conv5 = nn.Sequential(
                ConvBNReLU(128, 256, 3),
                HaarDownsampling(256)
            )

        # Up sampling
        self.up6 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2) if not haar_sampling else HaarUpsampling(256),
            ConvBNReLU(256, 128, 3)
        )
        self.conv6 = ConvBNReLU(256, 128, 3)
        self.up7 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2) if not haar_sampling else HaarUpsampling(128),
            ConvBNReLU(128, 64, 3)
        )
        self.conv7 = ConvBNReLU(128, 64, 3)
        self.up8 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2) if not haar_sampling else HaarUpsampling(64),
            ConvBNReLU(64, 32, 3)
        )
        self.conv8 = ConvBNReLU(64, 32, 3)
        self.up9 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2) if not haar_sampling else HaarUpsampling(32),
            ConvBNReLU(32, 32, 3)
        )
        self.conv9 = ConvBNReLU(64, 32, 3)
        self.residual = nn.Conv2d(32, out_channels, 1)

    def forward(self, image, secret, normalize: bool = False):
        if normalize:
            image = (image - 0.5) * 2.
            secret = (secret - 0.5) * 2.

        inputs = torch.cat([image, secret], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(conv5)
        conv6 = self.conv6(torch.cat([conv4, up6], dim=1))
        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv3, up7], dim=1))
        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv2, up8], dim=1))
        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv1, up9], dim=1))
        residual = self.residual(conv9)
        return residual


def run():
    pass


if __name__ == '__main__':
    run()
