#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : invisible_markers_decoder.py
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

from typing import Optional


class InvisibleMarkersDecoder(nn.Module):
    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1,
                 ckpt_path: Optional[str] = None):
        super().__init__()

        self.unet = Unet(
            encoder_name='timm-efficientnet-b1',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=out_channels,
        )

        self.img_norm = Normalize(mean=[0.4413, 0.4107, 0.3727], std=[0.2435, 0.2338, 0.2316])

        if ckpt_path is not None:
            self.unet.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=True)

    def forward(self, container: torch.Tensor, normalize: bool = False):
        if normalize:
            container = self.img_norm(container)
        rev_secret_logit = self.unet(container)
        return rev_secret_logit


def run():
    from torchinfo import summary
    import time
    import pdb

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decoder = InvisibleMarkersDecoder().to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    beg_time = time.time()
    decoder(inputs)
    summary(decoder, input_size=(1, 3, 256, 256), depth=10)
    # print(time.time() - beg_time)
    pass


if __name__ == '__main__':
    run()
