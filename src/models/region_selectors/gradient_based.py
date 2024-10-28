#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : gradient_based.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/14 10:27

# Import lib here
import numpy as np
import torch
import torch.nn as nn
from kornia.filters import Sobel
from kornia.color import rgb_to_y

import pdb


class SobelBasedRegionSelector(nn.Module):
    def __init__(self, size: int,
                 stride: int):
        super().__init__()
        self.sobel = Sobel(normalized=True)
        self.avg_pool = nn.AvgPool2d((size, size), stride=(stride, stride))
        self.size = size
        self.stride = stride

    def forward(self, image: torch.Tensor):
        assert image.min() >= 0. and image.max() <= 1.
        batch_size, channels, height, width = image.shape
        if channels == 3:
            gray = rgb_to_y(image)
        else:
            gray = image
        gradient = self.sobel(gray)
        pool_gradient = self.avg_pool(gradient)
        n_rows = pool_gradient.shape[-2]
        max_val, max_idx = torch.max(pool_gradient.reshape(batch_size, -1), dim=-1)
        top_left_h = torch.div(max_idx, n_rows, rounding_mode='trunc') * self.stride
        top_left_w = max_idx % n_rows * self.stride
        regions = []
        for b_idx in range(batch_size):
            top_left_h_ = top_left_h[b_idx]
            top_left_w_ = top_left_w[b_idx]

            # Avoid choose the boundary region as it causes chaos after perspective transform with border padding mode
            if top_left_h_ == 0:
                top_left_h_ = self.stride
                top_left_h[b_idx] = top_left_h_
            if top_left_h_ == height - self.size:
                top_left_h_ = height - self.size - self.stride
                top_left_h[b_idx] = top_left_h_
            if top_left_w_ == 0:
                top_left_w_ = self.stride
                top_left_w[b_idx] = top_left_w_
            if top_left_w_ == width - self.size:
                top_left_w_ = width - self.size - self.stride
                top_left_w[b_idx] = top_left_w_

            regions.append(image[b_idx, :, top_left_h_:top_left_h_+self.size, top_left_w_:top_left_w_+self.size])
        return torch.stack(regions, dim=0), torch.stack([top_left_h, top_left_w], dim=-1)


def run():
    import torchshow as ts
    from PIL import Image
    from torchvision import transforms

    sobel_based_region_selection = SobelBasedRegionSelector(128, 16)

    img_path = '/home/chengxin/Desktop/cherry.png'
    img = Image.open(img_path).convert('RGB').resize((256, 256))
    img = transforms.ToTensor()(img)
    regions = sobel_based_region_selection(torch.stack([img, img], dim=0))
    print(regions.shape)
    ts.show(regions)


if __name__ == '__main__':
    run()
