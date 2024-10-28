#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : cin_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/21 11:24

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders.cin_encoder import CINEncoder

import pdb


class CINDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, encoder, container: torch.Tensor, normalize: bool = False):
        if normalize:
            container = (container - 0.5) * 2.
        down_container = encoder.haar(container)
        fusion = torch.cat([down_container, down_container], dim=1)

        for blk in reversed(encoder.inv_blocks):
            fusion = blk(fusion, rev=True)

        rev_host = encoder.haar(fusion[:, :12], reverse=True)
        rev_secret = encoder.haar(fusion[:, 12:], reverse=True)
        return rev_host.clamp(0, 1), rev_secret.clamp(0, 1)

    def decode(self, encoder, container: torch.Tensor, normalize: bool = False):
        encode_size = encoder.encode_size
        container = F.interpolate(container, size=encode_size, mode='nearest')
        return self.forward(encoder, container, normalize)


def run():
    encoder = CINEncoder(3, strength_factor=1.)
    decoder = CINDecoder(encoder)
    host = torch.rand(2, 3, 128, 128)
    secret = torch.rand(2, 3, 128, 128)
    pdb.set_trace()
    container = encoder(host, secret, normalize=True)
    noised_container = container + torch.rand_like(container) * 0.01
    rev_host, rev_secret = decoder(noised_container)
    print('pause')
    pass


if __name__ == '__main__':
    run()
