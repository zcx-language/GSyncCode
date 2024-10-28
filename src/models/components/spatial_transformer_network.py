#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : spatial_transformer_network.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/29 22:29

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, input_size: Tuple[int, int, int]):
        super(SpatialTransformerNetwork, self).__init__()
        n_channels, height, width = input_size

        self.localization = nn.Sequential(
            nn.Conv2d(n_channels, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 1, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.sqz_height = height // 4
        self.sqz_width = width // 4

        self.fc_loc = nn.Sequential(
            nn.Linear(self.sqz_height*self.sqz_width, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, inputs: torch.Tensor):
        feat = self.localization(inputs).reshape(-1, self.sqz_height*self.sqz_width)
        theta = self.fc_loc(feat).reshape(-1, 2, 3)

        grid = F.affine_grid(theta, inputs.shape, align_corners=False)
        outputs = F.grid_sample(inputs, grid)
        return outputs


def run():
    pass


if __name__ == '__main__':
    run()
