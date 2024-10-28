#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : haar_loss.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/4/5 17:46

# Import lib here
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.color import rgb_to_yuv
from src.models.freq_transform.haar import Haar
from typing import List, Dict, Optional, Tuple


class YUVHaarLoss(nn.Module):
    def __init__(self, freq_weight: List,
                 yuv_weight: Optional[List] = None,
                 image_size: Tuple[int, int] = None,
                 edge_gain: float = 0.0,
                 norm_type: str = 'l2'):
        super().__init__()
        self.freq_weight = [weight for weight in freq_weight for i in range(3)]
        yuv_weight = [1., 1., 1.] if yuv_weight is None else yuv_weight
        self.yuv_weight = list(yuv_weight) * 4
        self.weight = [freq_weight * yuv_weight for freq_weight, yuv_weight in zip(self.freq_weight, self.yuv_weight)]
        self.haar = Haar(in_channels=3)
        self.norm_type = norm_type

        if edge_gain == 0:
            falloff_im = np.ones(image_size, dtype=np.float32)
        else:
            falloff_speed = 4
            falloff_im = np.ones(image_size, dtype=np.float32)
            for i in range(int(falloff_im.shape[0] / falloff_speed)):  # for i in range 100
                falloff_im[-i, :] *= (np.cos(4 * np.pi * i / image_size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
                falloff_im[i, :] *= (np.cos(4 * np.pi * i / image_size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
            for j in range(int(falloff_im.shape[1] / falloff_speed)):
                falloff_im[:, -j] *= (np.cos(4 * np.pi * j / image_size[0] + np.pi) + 1) / 2
                falloff_im[:, j] *= (np.cos(4 * np.pi * j / image_size[0] + np.pi) + 1) / 2
            falloff_im = 1 - falloff_im
            falloff_im = np.ones_like(falloff_im) + falloff_im * edge_gain
        self.register_buffer('falloff_im', torch.from_numpy(falloff_im), persistent=False)

    def forward(self, source: torch.Tensor,
                target: torch.Tensor):
        return yuv_haar_loss(source, target, self.freq_weight, norm_type=self.norm_type)


def yuv_haar_loss(source: torch.Tensor, target: torch.Tensor, freq_channel_weights: List[float], norm_type: str = 'l2'):
    if norm_type == 'l1':
        loss_func = F.l1_loss
    elif norm_type == 'l2':
        loss_func = F.mse_loss
    else:
        raise NotImplementedError

    source = rgb_to_yuv(source)
    target = rgb_to_yuv(target)

    haar = Haar(in_channels=3).to(source.device)
    source_haar = haar(source)
    target_haar = haar(target)

    loss = torch.mean(loss_func(source_haar[:, 0], target_haar[:, 0])) * freq_channel_weights[0] + \
           torch.mean(loss_func(source_haar[:, 1], target_haar[:, 1])) * freq_channel_weights[1] + \
           torch.mean(loss_func(source_haar[:, 2], target_haar[:, 2])) * freq_channel_weights[2] + \
           torch.mean(loss_func(source_haar[:, 3], target_haar[:, 3])) * freq_channel_weights[3] + \
           torch.mean(loss_func(source_haar[:, 4], target_haar[:, 4])) * freq_channel_weights[4] + \
           torch.mean(loss_func(source_haar[:, 5], target_haar[:, 5])) * freq_channel_weights[5] + \
           torch.mean(loss_func(source_haar[:, 6], target_haar[:, 6])) * freq_channel_weights[6] + \
           torch.mean(loss_func(source_haar[:, 7], target_haar[:, 7])) * freq_channel_weights[7] + \
           torch.mean(loss_func(source_haar[:, 8], target_haar[:, 8])) * freq_channel_weights[8] + \
           torch.mean(loss_func(source_haar[:, 9], target_haar[:, 9])) * freq_channel_weights[9] + \
           torch.mean(loss_func(source_haar[:, 10], target_haar[:, 10])) * freq_channel_weights[10] + \
           torch.mean(loss_func(source_haar[:, 11], target_haar[:, 11])) * freq_channel_weights[11]
    return loss


class RGBHaarLoss(nn.Module):
    def __init__(self, image_size: Tuple[int, int],
                 freq_weight: List,
                 rgb_weight: Optional[List] = None,
                 dist_type: str = 'l1',
                 edge_gain: float = 1.0):
        super().__init__()
        self.freq_weight = [weight for weight in freq_weight for i in range(3)]
        rgb_weight = [1., 1., 1.] if rgb_weight is None else rgb_weight
        self.rgb_weight = list(rgb_weight) * 4
        self.weight = [freq_weight * rgb_weight for freq_weight, rgb_weight in zip(self.freq_weight, self.rgb_weight)]
        self.haar = Haar(in_channels=3)

        if dist_type == 'l1':
            self.loss_func = F.l1_loss
        elif dist_type == 'l2':
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

        falloff_speed = 4
        falloff_im = np.ones(image_size, dtype=np.float32)
        for i in range(int(falloff_im.shape[0] / falloff_speed)):  # for i in range 100
            falloff_im[-i, :] *= (np.cos(4 * np.pi * i / image_size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
            falloff_im[i, :] *= (np.cos(4 * np.pi * i / image_size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
        for j in range(int(falloff_im.shape[1] / falloff_speed)):
            falloff_im[:, -j] *= (np.cos(4 * np.pi * j / image_size[0] + np.pi) + 1) / 2
            falloff_im[:, j] *= (np.cos(4 * np.pi * j / image_size[0] + np.pi) + 1) / 2
        falloff_im = 1 - falloff_im
        falloff_im = np.ones_like(falloff_im) + falloff_im * edge_gain
        self.register_buffer('falloff_im', torch.from_numpy(falloff_im), persistent=False)

    def forward(self, container: torch.Tensor,
                host: torch.Tensor):
        # haar transform: B, 3, H, W -> B, 12, H//2, W//2
        # R_{ll}G_{ll}B_{ll}R_{hl}G_{hl}B_{hl}R_{lh}G_{lh}B_{lh}R_{hh}G_{hh}B_{hh}
        container_haar = self.haar(container)
        host_haar = self.haar(host)
        loss = torch.mean(self.loss_func(container_haar[:, 0], host_haar[:, 0], reduction='none') * self.falloff_im) * self.weight[0] + \
               torch.mean(self.loss_func(container_haar[:, 1], host_haar[:, 1], reduction='none') * self.falloff_im) * self.weight[1] + \
               torch.mean(self.loss_func(container_haar[:, 2], host_haar[:, 2], reduction='none') * self.falloff_im) * self.weight[2] + \
               torch.mean(self.loss_func(container_haar[:, 3], host_haar[:, 3], reduction='none') * self.falloff_im) * self.weight[3] + \
               torch.mean(self.loss_func(container_haar[:, 4], host_haar[:, 4], reduction='none') * self.falloff_im) * self.weight[4] + \
               torch.mean(self.loss_func(container_haar[:, 5], host_haar[:, 5], reduction='none') * self.falloff_im) * self.weight[5] + \
               torch.mean(self.loss_func(container_haar[:, 6], host_haar[:, 6], reduction='none') * self.falloff_im) * self.weight[6] + \
               torch.mean(self.loss_func(container_haar[:, 7], host_haar[:, 7], reduction='none') * self.falloff_im) * self.weight[7] + \
               torch.mean(self.loss_func(container_haar[:, 8], host_haar[:, 8], reduction='none') * self.falloff_im) * self.weight[8] + \
               torch.mean(self.loss_func(container_haar[:, 9], host_haar[:, 9], reduction='none') * self.falloff_im) * self.weight[9] + \
               torch.mean(self.loss_func(container_haar[:, 10], host_haar[:, 10], reduction='none') * self.falloff_im) * self.weight[10] + \
               torch.mean(self.loss_func(container_haar[:, 11], host_haar[:, 11], reduction='none') * self.falloff_im) * self.weight[11]
        return loss


def run():
    pass


if __name__ == '__main__':
    run()
