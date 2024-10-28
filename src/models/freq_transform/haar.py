#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : haar.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : https://github.com/rmpku/CIN/blob/main/codes/models/modules/InvDownscaling.py
# @CreateTime   : 2023/3/16 16:47

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

from typing import List, Tuple


class Haar(nn.Module):
    """Attention, in_channels the channel number of forward haar transform, which is 0.25 times of reverse transform"""
    def __init__(self, in_channels: int):
        super().__init__()

        # Initialize weights
        weights = torch.ones(4, 1, 2, 2, dtype=torch.float32)
        weights[1, 0, 0, 1] = -1
        weights[1, 0, 1, 1] = -1
        weights[2, 0, 1, 0] = -1
        weights[2, 0, 1, 1] = -1
        weights[3, 0, 1, 0] = -1
        weights[3, 0, 0, 1] = -1
        weights = weights * 0.25    # Normalize
        weights = torch.cat([weights]*in_channels, dim=0)
        # persistent=False: do not save the weights in the state_dict
        self.register_buffer('weights', weights, persistent=False)
        self.in_channels = in_channels

    def forward(self, inputs: torch.Tensor, reverse: bool = False):
        n_batches, n_channels, height, width = inputs.shape
        if not reverse:
            outputs = F.conv2d(inputs, self.weights, bias=None, stride=2, groups=self.in_channels)
            outputs = outputs.reshape(n_batches, self.in_channels, 4, height//2, width//2)
            outputs = outputs.transpose(1, 2).reshape(n_batches, self.in_channels*4, height//2, width//2)
        else:
            outputs = inputs.reshape(n_batches, 4, self.in_channels, height, width)
            outputs = outputs.transpose(1, 2).reshape(n_batches, self.in_channels*4, height, width)
            outputs = F.conv_transpose2d(outputs, self.weights, bias=None, stride=2, groups=self.in_channels)
        return outputs


class InverseHaar(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # Initialize weights
        weights = torch.ones(4, 1, 2, 2, dtype=torch.float32)
        weights[1, 0, 0, 1] = -1
        weights[1, 0, 1, 1] = -1
        weights[2, 0, 1, 0] = -1
        weights[2, 0, 1, 1] = -1
        weights[3, 0, 1, 0] = -1
        weights[3, 0, 0, 1] = -1
        weights = weights * 0.25    # Normalize
        weights = torch.cat([weights]*(in_channels//4), dim=0)
        self.register_buffer('weights', weights, persistent=False)

    def forward(self, inputs: torch.Tensor):
        n_batches, n_channels, height, width = inputs.shape
        outputs = inputs.reshape(n_batches, 4, n_channels//4, height, width)
        outputs = outputs.transpose(1, 2).reshape(n_batches, n_channels, height, width)
        outputs = F.conv_transpose2d(outputs, self.weights, bias=None, stride=2, groups=n_channels//4)
        return outputs


def run():
    import numpy as np
    import torchshow as ts
    from PIL import Image
    from torchvision import transforms
    import pywt
    import cv2
    # from src.utils.image_tools import image_show
    import matplotlib.pyplot as plt


    haar = Haar(in_channels=1)
    img = Image.open('/home/chengxin/Desktop/Accept_dmtx.png').convert('L').resize((128, 128))
    img = np.array(img)
    img = np.pad(img, ((9, 9), (9, 9)), mode='constant', constant_values=255)
    coeffs = pywt.dwt2(img, 'haar', mode='symmetric')
    LL, (LH, HL, HH) = coeffs
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    # print the unique values in the image and the numbers of each unique value
    # print(Counter(img.flatten()), len(img.flatten()))

    # 显示近似系数
    plt.subplot(3, 3, 2)
    plt.imshow(LL, cmap='gray')
    plt.title('Approximation')
    # print(Counter(LL.flatten()), len(LL.flatten()))

    # 显示水平细节系数
    plt.subplot(3, 3, 3)
    plt.imshow(LH, cmap='gray')
    plt.title('Horizontal detail')
    # print(Counter(LH.flatten()), len(LH.flatten()))

    # 显示垂直细节系数
    plt.subplot(3, 3, 4)
    plt.imshow(HL, cmap='gray')
    plt.title('Vertical detail')
    # print(Counter(HL.flatten()), len(HL.flatten()))

    # 显示对角细节系数
    plt.subplot(3, 3, 5)
    plt.imshow(HH, cmap='gray')
    plt.title('Diagonal detail')
    # print(Counter(HH.flatten()), len(HH.flatten()))

    # inverse transform
    zeros_mat = np.zeros_like(HH)

    img = pywt.idwt2((LL, (zeros_mat, zeros_mat, zeros_mat)), 'haar', mode='symmetric')
    plt.subplot(3, 3, 6)
    plt.imshow(img, cmap='gray')
    plt.title('Inverse LL')

    img = pywt.idwt2((zeros_mat, (LH, zeros_mat, zeros_mat)), 'haar', mode='symmetric')
    plt.subplot(3, 3, 7)
    plt.imshow(img, cmap='gray')
    plt.title('Inverse LH')

    img = pywt.idwt2((zeros_mat, (zeros_mat, HL, zeros_mat)), 'haar', mode='symmetric')
    plt.subplot(3, 3, 8)
    plt.imshow(img, cmap='gray')
    plt.title('Inverse HL')

    img = pywt.idwt2((zeros_mat, (zeros_mat, zeros_mat, HH)), 'haar', mode='symmetric')
    plt.subplot(3, 3, 9)
    plt.imshow(img, cmap='gray')
    plt.title('Inverse HH')

    # plt.tight_layout()
    plt.show()


    img = Image.open('/home/chengxin/Desktop/dress.png').convert('L').resize((128, 128))
    img = np.array(img)
    img = np.pad(img, ((9, 9), (9, 9)), mode='constant', constant_values=255)
    coeffs = pywt.dwt2(img, 'haar', mode='symmetric')
    LL2, (LH2, HL2, HH2) = coeffs
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    # 显示近似系数
    plt.subplot(3, 3, 2)
    plt.imshow(LL2, cmap='gray')
    plt.title('Approximation')

    # 显示水平细节系数
    plt.subplot(3, 3, 3)
    plt.imshow(LH2, cmap='gray')
    plt.title('Horizontal detail')

    # 显示垂直细节系数
    plt.subplot(3, 3, 4)
    plt.imshow(HL2, cmap='gray')
    plt.title('Vertical detail')

    # 显示对角细节系数
    plt.subplot(3, 3, 5)
    plt.imshow(HH2, cmap='gray')
    plt.title('Diagonal detail')

    # inverse transform
    img = pywt.idwt2((LL2, (LH2, HL2, HH)), 'haar', mode='symmetric')
    plt.subplot(3, 3, 6)
    plt.imshow(img, cmap='gray')
    plt.title('Inverse merge')

    plt.show()

    # print(np.array(img).min(), np.array(img).max())
    # img = transforms.ToTensor()(img).unsqueeze(dim=0)
    # ts.show(img.squeeze(dim=0))
    # haar_img = haar(img)
    # ts.show(haar_img[0])
    # ts.show(haar_img[0, 1:2])
    # import pdb; pdb.set_trace()
    # ts.show(haar_img[0, 2:3])
    # ts.show(haar_img[0, 3:4])

    # torch.save(haar.state_dict(), '/home/chengxin/Desktop/haar.pth')
    # image = np.array(Image.open('/home/chengxin/Desktop/Accept_qrcode.png').convert('L').resize((256, 256)))
    # # image = image[64:-64, 64:-64, 2]
    # img = transforms.ToTensor()(image).unsqueeze(dim=0)
    # haar_img = haar(img)
    # ts.show(haar_img.squeeze().unsqueeze(dim=1))
    pass

"""
Counter({255: 12228, 0: 9088}) 21316
Counter({510.00000000000006: 2491, 0.0: 1713, 255.00000000000003: 981, 382.50000000000006: 79, 127.50000000000001: 65}) 5329
Counter({0.0: 4756, -255.00000000000003: 223, 255.00000000000003: 206, 127.50000000000001: 89, -127.50000000000001: 55}) 5329
Counter({0.0: 4660, 255.00000000000003: 268, -255.00000000000003: 257, -127.50000000000001: 83, 127.50000000000001: 61}) 5329
Counter({0.0: 5158, -127.50000000000001: 73, 127.50000000000001: 71, 255.00000000000003: 14, -255.00000000000003: 13}) 5329
"""


if __name__ == '__main__':
    run()
