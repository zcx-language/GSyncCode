#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : inv_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/27 21:37

# Import lib here
import torch
import torch.nn as nn

from src.models.freq_transform.haar import Haar
from src.models.components import InvArch


class InvEncoder(nn.Module):
    """Inv Encoder
    Args:
        in_channels (int): number of input channels
        strength_factor (float): strength factor for secret image
    """
    def __init__(self, in_channels: int, strength_factor: float, n_inv_blocks: int = 6):
        super().__init__()
        self.haar = Haar(in_channels)
        self.inv_blocks = nn.ModuleList([InvArch(in_channels*4, in_channels*4) for _ in range(n_inv_blocks)])
        self.in_channels = in_channels
        self.strength_factor = strength_factor

    def forward(self, host: torch.Tensor, secret: torch.Tensor, normalize: bool = False):
        if normalize:
            host = (host - 0.5) * 2.
            secret = (secret - 0.5) * 2.

        down_host = self.haar(host)
        down_secret = self.haar(secret)
        fusion = torch.cat([down_host, down_secret], dim=1)

        for blk in self.inv_blocks:
            fusion = blk(fusion, rev=False)

        en_secret = fusion[:, self.in_channels*4:]
        container = down_host + self.strength_factor * en_secret
        container = self.haar(container, reverse=True)
        return container.clamp(0, 1)


def run():
    pass


if __name__ == '__main__':
    run()
