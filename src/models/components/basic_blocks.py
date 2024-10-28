#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ScreenShootResilient
# @File         : basic_blocks.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/11/27 21:06
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size: int = 3,
                 stride: int = 1):
        super(ConvBNReLU, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


def run():
    pass


if __name__ == '__main__':
    run()
