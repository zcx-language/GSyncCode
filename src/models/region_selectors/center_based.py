#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : center_based.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/14 16:13

# Import lib here
import numpy as np
import torch
import torch.nn as nn
from kornia.filters import Sobel
from kornia.color import rgb_to_y

import pdb


class CenterBasedRegionSelector(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, image: torch.Tensor):
        n_batch, n_channel, height, width = image.shape
        h_pad = (height - self.size) // 2
        w_pad = (width - self.size) // 2
        patch = image[:, :, h_pad:-h_pad, w_pad:-w_pad]
        top_left_pt = torch.tensor([h_pad, w_pad], dtype=torch.int, device=image.device)
        return patch, torch.stack([top_left_pt for _ in range(n_batch)], dim=0)


def run():
    pass


if __name__ == '__main__':
    run()
