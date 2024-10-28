#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : illumination.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/10 10:53

# Import lib here
import random
import numpy as np
import torch
import torch.nn as nn

from typing import Optional


def illumination_distortion(c, embed_image_shape):
    mask = np.zeros(embed_image_shape)
    mask_2d = np.zeros((embed_image_shape[2], embed_image_shape[3]))
    a = 0.7 + np.random.rand(1) * 0.2
    b = 1.1 + np.random.rand(1) * 0.2
    if c == 0:
        direction = np.random.randint(1, 5)
        for i in range(embed_image_shape[2]):
            mask_2d[i, :] = -((b - a) / (mask.shape[2] - 1)) * (i - mask.shape[3]) + a
        if direction == 1:
            O = mask_2d
        elif direction == 2:
            O = np.rot90(mask_2d, 1)
        elif direction == 3:
            O = np.rot90(mask_2d, 2)
        elif direction == 4:
            O = np.rot90(mask_2d, 3)
#         for batch in range(embed_image_shape[0]):
#             for channel in range(embed_image_shape[1]):
#                 mask[batch, channel, :, :] = O
    else:
        x = np.random.randint(0, mask.shape[2])
        y = np.random.randint(0, mask.shape[3])
        max_len = np.max([np.sqrt(x ** 2 + y ** 2),
                          np.sqrt((x - 255) ** 2 + y ** 2),
                          np.sqrt(x ** 2 + (y - 255) ** 2),
                          np.sqrt((x - 255) ** 2 + (y - 255) ** 2)])
        for i in range(mask.shape[2]):
            for j in range(mask.shape[3]):
                mask[:, :, i, j] = np.sqrt((i - x) ** 2 + (j - y) ** 2) / max_len * (a - b) + b
        O = mask
    return np.ascontiguousarray(O, dtype=np.float32)


class Illumination(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, image: torch.Tensor, pattern: Optional[torch.Tensor] = None):
        if random.random() <= self.p:
            assert image.min() >= 0 and image.max() <= 1, \
                f'Need inputs in range[0, 1], but got [{image.min()}, {image.max()}]'
            if pattern is None:
                c = random.randint(0, 1)
                pattern = torch.from_numpy(illumination_distortion(c, image.shape)).to(image.device)
            distorted_image = image * pattern
        else:
            distorted_image = image
        return distorted_image.clamp(0, 1)


def run():
    import qrcode
    import pdb
    from src.utils.image_tools import image_show, to_tensor, img_norm, img_denorm

    atk = Illumination()

    qrcode_img = np.array(qrcode.make('Accept'), dtype=np.float32)
    qrcode_tsr = to_tensor(qrcode_img).unsqueeze(dim=0)
    warped_qrcode_tsr = atk(qrcode_tsr)
    image_show(warped_qrcode_tsr)
    pass


if __name__ == '__main__':
    run()
