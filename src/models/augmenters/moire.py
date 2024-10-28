#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : moire.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/10 10:10

# Import lib here
import random
import math
import numpy as np
import torch
import torch.nn as nn

from typing import Union, Tuple, Optional


def moire_gen(p_size, theta, center_x, center_y):
    z = np.zeros((p_size, p_size))
    for i in range(p_size):
        for j in range(p_size):
            z1 = 0.5 + 0.5 * math.cos(2 * math.pi * np.sqrt((i + 1 - center_x) ** 2 + (j + 1 - center_y) ** 2))
            z2 = 0.5 + 0.5 * math.cos(
                math.cos(theta / 180 * math.pi) * (j + 1) + math.sin(theta / 180 * math.pi) * (i + 1))
            z[i, j] = np.min([z1, z2])
    M = (z + 1) / 2
    return M


def moire_distortion(embed_image_shape):
    batch_size, channels, height, width = embed_image_shape
    Z = np.zeros(embed_image_shape)
    for i in range(channels):
        theta = np.random.randint(0, 180)
        center_x = np.random.rand(1) * height
        center_y = np.random.rand(1) * width
        M = moire_gen(height, theta, center_x, center_y)
        Z[:, i, :, :] = M
    return np.ascontiguousarray(Z, dtype=np.float32)


class Moire(nn.Module):
    def __init__(self, weight_bound: float, p: float = 0.5):
        super().__init__()
        self.weight_bound = weight_bound
        self.p = p

    def forward(self, image: torch.Tensor, pattern: Optional[torch.Tensor] = None):
        if random.random() <= self.p:
            assert image.min() >= 0 and image.max() <= 1, \
                f'Need inputs in range[0, 1], but got [{image.min()}, {image.max()}]'
            if pattern is None:
                pattern = torch.from_numpy(moire_distortion(image.shape)*2-1).to(image.device)
            weight = random.random() * self.weight_bound
            moire_image = pattern * weight + image * (1-weight)
        else:
            moire_image = image
        return moire_image.clamp(0, 1)


def run():
    import qrcode
    import pdb
    from src.utils.image_tools import image_show, to_tensor, img_norm, img_denorm

    atk = Moire()

    qrcode_img = np.array(qrcode.make('Accept'), dtype=np.float32)
    qrcode_tsr = to_tensor(qrcode_img).unsqueeze(dim=0)
    warped_qrcode_tsr = atk(qrcode_tsr)
    image_show(warped_qrcode_tsr)
    pass


if __name__ == '__main__':
    run()
