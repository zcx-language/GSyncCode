#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : invisible_markers_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/8 09:58

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torchinfo import summary
from segmentation_models_pytorch import Unet
import pdb

from typing import Optional, Tuple


class InvisibleMarkersEncoder(nn.Module):
    def __init__(self, in_channels: int = 6,
                 out_channels: int = 3,
                 depth: int = 4,
                 decoder_channels: Tuple = (128, 64, 32, 16),
                 ckpt_path: Optional[str] = None):
        super().__init__()

        self.unet = Unet(
            encoder_name='timm-efficientnet-b1',
            encoder_depth=depth,
            encoder_weights='imagenet',
            decoder_channels=decoder_channels,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

        self.img_norm = Normalize(mean=[0.4413, 0.4107, 0.3727], std=[0.2435, 0.2338, 0.2316])
        self.code_norm = Normalize(0.5, 0.5)

        if ckpt_path is not None:
            self.unet.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=True)

    def forward(self, host: torch.Tensor, secret: torch.Tensor, normalize: bool = False):
        if normalize:
            host = self.img_norm(host)
            secret = self.code_norm(secret)
        residual = self.unet(torch.cat([host, secret], dim=1))
        return residual


def run():
    from torchinfo import summary
    import pdb
    encoder = InvisibleMarkersEncoder()
    input_data = (torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128))
    summary(encoder, input_data=input_data, device='cpu')
    pdb.set_trace()
    print('pause')
    pass


if __name__ == '__main__':
    run()
