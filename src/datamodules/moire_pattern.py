#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : moire_pattern.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/12 17:58

# Import lib here
import numpy as np
import math
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from typing import Tuple


def moire_gen(p_size, theta, center_x, center_y):
    z = np.zeros((p_size, p_size))
    for i in range(p_size):
        for j in range(p_size):
            z1 = 0.5 + 0.5 * math.cos(2 * math.pi * np.sqrt((i + 1 - center_x) ** 2 + (j + 1 - center_y) ** 2))
            z2 = 0.5 + 0.5 * math.cos(
                math.cos(theta / 180 * math.pi) * (j + 1) + math.sin(theta / 180 * math.pi) * (i + 1))
            z[i, j] = np.min([z1, z2])
    M = (z + 1) / 2
    return M


def get_moire_pattern(pattern_shape: Tuple[int, int, int]):
    channels, height, width = pattern_shape
    pattern = np.zeros(pattern_shape)
    for i in range(channels):
        theta = np.random.randint(0, 180)
        center_x = np.random.rand(1) * height
        center_y = np.random.rand(1) * width
        M = moire_gen(height, theta, center_x, center_y)
        pattern[i, :, :] = M
    return np.ascontiguousarray(pattern, dtype=np.float32)


class MoirePattern(Dataset):
    """
    Args:
        data_dir:
        pattern_size: (channels, height, width)
    """
    def __init__(self, data_dir: str,
                 pattern_size: Tuple[int, int],
                 stage: str = 'train'):
        super().__init__()
        assert stage.lower() in ['train', 'valid', 'test']

        paths = sorted(str(path) for path in Path(data_dir).glob('*.jpeg'))
        nums = len(paths)
        if stage.lower() == 'valid':
            self.paths = paths[int(nums*0.95):]
        elif stage.lower() == 'test':
            self.paths = paths[int(nums*0.9):int(nums*0.95)]
        else:
            self.paths = paths[:int(nums*0.9)]

        self.pattern_size = pattern_size

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        pattern = Image.open(path).resize(self.pattern_size[::-1])
        return self.to_tensor(pattern)


def run():
    import pdb

    # Generate patterns
    # save_path = '/sda1/Datasets/MoirePattern/images'
    # nums = 25000
    # for i in range(nums):
    #     moire_ary = np.clip(get_moire_pattern((3, 256, 256))*2-1, 0, 1).transpose(1, 2, 0)
    #     moire_pattern = (moire_ary * 255).astype(np.uint8)
    #     moire_pattern = Image.fromarray(moire_pattern)
    #     moire_pattern.save(f'{save_path}/{i:06d}.jpeg')

    # Iter dataset
    dataset = MoirePattern((3, 128, 128))
    for batch in dataset:
        pdb.set_trace()
        print('pause')

    pass


if __name__ == '__main__':
    run()
