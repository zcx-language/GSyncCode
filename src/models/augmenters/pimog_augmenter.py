#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : pimog_augmenter.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/8 14:11

# Import lib here
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import RandomGaussianNoise
from omegaconf import DictConfig

from .random_perspective import RandomPerspective
from .illumination import Illumination
from .moire import Moire

from typing import Optional, Dict


class PIMoGAugmenter(nn.Module):
    def __init__(self, perspective_scale: float = 0.5,
                 perspective_p: float = 1.,
                 illumination_p: float = 1.,
                 moire_weight_bound: float = 0.15,
                 moire_p: float = 1.,
                 noise_std: float = 0.01,
                 noise_p: float = 1.):
        super().__init__()
        self.perspective = RandomPerspective(perspective_scale, p=perspective_p, resample='nearest',
                                             sampling_method='area_preserving', padding_mode='reflection')
        self.illumination = Illumination(p=illumination_p)
        self.moire = Moire(moire_weight_bound, p=moire_p)
        self.noise = RandomGaussianNoise(std=noise_std, p=noise_p)

    def forward(self, image: torch.Tensor, distortion_pattern: Optional[Dict] = None):
        if distortion_pattern is None:
            distortion_pattern = {}
        distorted_image = self.perspective(image)
        illumination_pattern = distortion_pattern.get('illumination', None)
        distorted_image = self.illumination(distorted_image, illumination_pattern)
        moire_pattern = distortion_pattern.get('moire', None)
        distorted_image = self.moire(distorted_image, moire_pattern)
        distorted_image = self.noise(distorted_image)
        return distorted_image.clamp(0, 1)


def run():
    import qrcode
    import pdb
    from torchvision.transforms import ToTensor, Normalize
    from src.utils.image_tools import image_show

    to_tensor = ToTensor()
    img_norm = Normalize(0.5, 0.5)
    img_denorm = Normalize(-1, 2)

    qrcode_img = np.array(qrcode.make('Accept'), dtype=np.float32)
    qrcode_tsr = img_norm(to_tensor(qrcode_img)).squeeze(dim=0)
    atk = PIMoGAugmenter()
    warped_qrcode_tsr = atk(qrcode_tsr)
    pdb.set_trace()
    image_show(img_denorm(warped_qrcode_tsr))
    pass


if __name__ == '__main__':
    run()
