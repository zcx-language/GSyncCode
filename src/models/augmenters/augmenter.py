#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : augmenter.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/15 23:11

# Import lib here
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from typing import List, Dict, Optional


class Augmenter(nn.Module):
    def __init__(self, aug_dict: DictConfig):
        super().__init__()
        self.aug_dict = aug_dict
        self.jpeg = aug_dict.get('jpeg')
        self._perspective = None

    @property
    def distortion_types(self):
        return self.aug_dict.keys()

    @property
    def perspective(self):
        return self._perspective

    def forward(self, image: torch.Tensor,
                batch_idx: int,
                distortion_pattern: Optional[Dict] = None,
                return_individual: bool = False):
        image = image.clamp(0, 1)
        if distortion_pattern is None:
            distortion_pattern = {}
        distorted_image = image
        orig_image = torch.clone(image)
        distortion_dict = dict()
        for aug_name, aug in self.aug_dict.items():
            if aug_name == 'perspective':
                # increase the distortion scale from 0 to 0.5 in 5000 batches
                scale = aug.keywords['distortion_scale_bound'] * min(1., (batch_idx+2)/5000.)
                self._perspective = aug(distortion_scale=scale)
                # o_distorted_image = distorted_image
                distorted_image = self._perspective(distorted_image)
                if return_individual:
                    distortion_dict[aug_name] = self._perspective(orig_image,
                                                                  params=self._perspective._params).clamp(0., 1.)

                # Padding the border with original image, but the `_perspective` has no such choice.
                # We achieve this manually by setting the padding_mode of `_perspective` to `fill` and filled with 1.
                # Filled with 1. is required by getting 2d barcode gt.
                # mask = torch.zeros_like(distorted_image)
                # mask = self._perspective(mask, params=self._perspective._params).ge(0.5).float()
                # distorted_image = mask * o_distorted_image + (1-mask) * distorted_image

            elif aug_name == 'illumination':
                pattern = distortion_pattern.get('illumination', None)
                distorted_image = aug(distorted_image, pattern)
                if return_individual:
                    distortion_dict[aug_name] = aug(orig_image, pattern).clamp(0., 1.)
            elif aug_name == 'moire':
                pattern = distortion_pattern.get('moire', None)
                distorted_image = aug(distorted_image, pattern)
                if return_individual:
                    distortion_dict[aug_name] = aug(orig_image, pattern).clamp(0., 1.)
            elif aug_name == 'jpeg':
                distorted_image = self.jpeg(distorted_image)
                if return_individual:
                    distortion_dict[aug_name] = self.jpeg(orig_image).clamp(0., 1.)
            else:
                # aug = aug.to(distorted_image.device)
                distorted_image = aug(distorted_image)
                if return_individual:
                    distortion_dict[aug_name] = aug(orig_image).clamp(0., 1.)
            distorted_image = distorted_image.clamp(0., 1.)
        distortion_dict['combine'] = distorted_image
        return distortion_dict


def run():
    from PIL import Image
    from src.utils.image_tools import image_show

    from src.models.augmenters.random_perspective import RandomPerspective
    from src.models.augmenters.illumination import Illumination
    from src.models.augmenters.moire import Moire
    from kornia.augmentation import RandomGaussianBlur, RandomGaussianNoise
    from src.models.augmenters.DiffJPEG.DiffJPEG import DiffJPEG

    perspective = RandomPerspective(distortion_scale=0.5, resample='nearest',
                                    p=1., sampling_method='basic', padding_mode='border')
    illumination = Illumination(p=1.)
    moire = Moire(weight_bound=0.1, p=1.)
    blur = RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.8, 0.8), p=1.)
    noise = RandomGaussianNoise(mean=0.0, std=0.02)
    jpeg = DiffJPEG(height=256, width=256, differentiable=True, quality=80, p=1.)

    aug = Augmenter(dict(
        perspective2=perspective,
        illumination=illumination,
        moire=moire,
        blur=blur,
        noise=noise,
        jpeg=jpeg,
    ))

    image_path = '/home/chengxin/Pictures/04161909.png'
    image = np.array(Image.open(image_path).convert('RGB').resize((256, 256)), dtype=np.float32) / 255.
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    aug_image = aug(image, 999999999)
    image_show(aug_image)
    pass


if __name__ == '__main__':
    run()
