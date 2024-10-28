#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : weighted_rgb_loss.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/4/11 17:14

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.color import rgb_to_yuv

from typing import Tuple, List, Optional


class WeightedRGBLoss(nn.Module):
    def __init__(self, weights: Optional[Tuple[float, float, float]] = None,
                 dist_type: str = 'l1'):
        super().__init__()
        self.weights = weights

        if dist_type == 'l1':
            self.loss_func = F.l1_loss
        elif dist_type == 'l2':
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, container: torch.Tensor, host: torch.Tensor):
        if self.weights:
            r_loss = self.weights[0] * self.loss_func(container[:, 0], host[:, 0])
            g_loss = self.weights[1] * self.loss_func(container[:, 1], host[:, 1])
            b_loss = self.weights[2] * self.loss_func(container[:, 2], host[:, 2])
            loss = (r_loss + g_loss + b_loss) / 3.
        else:
            loss = self.loss_func(container, host)
        return loss


def run():
    import numpy as np
    from PIL import Image
    img = Image.open('/home/chengxin/Desktop/Accept_qrcode.png').convert('RGB')
    img = np.array(img)
    y = Image.fromarray(img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114)
    y.show()
    u = Image.fromarray(img[:, :, 0] * -0.14713 + img[:, :, 1] * -0.28886 + img[:, :, 2] * 0.436)
    u.show()
    v = Image.fromarray(img[:, :, 0] * 0.615 + img[:, :, 1] * -0.51499 + img[:, :, 2] * -0.10001)
    v.show()
    pass


if __name__ == '__main__':
    run()
