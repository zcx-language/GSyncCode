#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : inv_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : https://github.com/ilsang/PyTorch-SE-Segmentation/blob/master/model.py
# @CreateTime   : 2023/3/27 21:38

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import Unet

from src.models.freq_transform.haar import Haar


class InvDecoder(nn.Module):
    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1):
        super().__init__()
        self.in_haar = Haar(in_channels)
        self.out_haar = Haar(out_channels)

        self.unet = Unet(
            encoder_name='timm-efficientnet-b1',
            encoder_weights='imagenet',
            in_channels=in_channels * 4,
            classes=out_channels * 4,
        )

    def forward(self, container: torch.Tensor, normalize: bool = False):
        if normalize:
            container = (container - 0.5) * 2.
        container = self.in_haar(container)
        logit = self.unet(container)
        logit = self.out_haar(logit, reverse=True)
        return logit


def run():
    from torchinfo import summary
    se_unet = InvDecoder(3, 1, init_features=32, network_depth=4, bottleneck_layers=2, reduction_ratio=16)
    summary(se_unet, input_size=(16, 3, 256, 256))
    pass


if __name__ == '__main__':
    run()
