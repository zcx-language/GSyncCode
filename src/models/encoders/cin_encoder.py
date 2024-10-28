#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : cin_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : https://github.com/rmpku/CIN/blob/main/codes/models/CIN.py
# @CreateTime   : 2023/3/21 10:31

# Import lib here
import torch
import torch.nn as nn

from src.models.freq_transform.haar import Haar
from src.models.components import InvArch
from src.models.region_selectors.gradient_based import SobelBasedRegionSelector


class CINEncoder(nn.Module):
    """CIN Encoder
    Args:
        in_channels (int): number of input channels
        encode_size (int): size of region for encoding
        strength_factor (float): strength factor for secret image
    """
    def __init__(self, in_channels: int, encode_size: int, strength_factor: float, n_inv_blocks: int = 16):
        super().__init__()
        self.haar = Haar(in_channels)
        self.inv_blocks = nn.ModuleList([InvArch(in_channels*4, in_channels*4) for _ in range(n_inv_blocks)])
        self.in_channels = in_channels
        self.encode_size = encode_size
        self.strength_factor = strength_factor
        self.region_selector = SobelBasedRegionSelector(encode_size, 16)

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

    def encode(self, host: torch.Tensor, secret: torch.Tensor, normalize: bool = False):
        # If host's shape is not equal to secret's shape, region selection in host
        if host.shape[2:] != secret.shape[2:]:
            region, top_left_pt = self.region_selector(host)
            encoded_region = self.forward(region, secret, normalize=normalize)

            container = torch.clone(host)
            host_gt = torch.ones_like(host)
            secret_gt = torch.ones_like(host)
            region_height, region_width = region.shape[2:]
            for batch_idx in range(host.shape[0]):
                h_idx = top_left_pt[batch_idx, 0]
                w_idx = top_left_pt[batch_idx, 1]
                container[batch_idx, :, h_idx:h_idx+region_height, w_idx:w_idx+region_width] = encoded_region[batch_idx]
                host_gt[batch_idx, :, h_idx:h_idx+region_height, w_idx:w_idx+region_width] = region[batch_idx]
                secret_gt[batch_idx, :, h_idx:h_idx+region_height, w_idx:w_idx+region_width] = secret[batch_idx]
        else:
            raise NotImplementedError
        return container, host_gt, secret_gt


def run():
    pass


if __name__ == '__main__':
    run()
