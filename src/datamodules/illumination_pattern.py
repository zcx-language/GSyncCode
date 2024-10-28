#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : illumination_pattern.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/12 16:02

# Import lib here
import numpy as np
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from typing import List, Tuple, Optional


def get_illumination_pattern(type_idx: int, pattern_shape: Tuple[int, int, int]):
    channels, height, width = pattern_shape
    mask = np.zeros(pattern_shape[-2:])

    a = 0.7 + np.random.rand(1) * 0.2
    b = 1.1 + np.random.rand(1) * 0.2
    if type_idx == 0:
        direction = np.random.randint(1, 5)
        for i in range(height):
            mask[i, :] = -((b - a) / (height - 1)) * (i - width) + a
        if direction == 1:
            pass
        elif direction == 2:
            mask = np.rot90(mask, 1)
        elif direction == 3:
            mask = np.rot90(mask, 2)
        elif direction == 4:
            mask = np.rot90(mask, 3)
    else:
        x = np.random.randint(0, height)
        y = np.random.randint(0, width)
        max_len = np.max([np.sqrt(x ** 2 + y ** 2),
                          np.sqrt((x - 255) ** 2 + y ** 2),
                          np.sqrt(x ** 2 + (y - 255) ** 2),
                          np.sqrt((x - 255) ** 2 + (y - 255) ** 2)])
        for i in range(height):
            for j in range(width):
                mask[i, j] = np.sqrt((i - x) ** 2 + (j - y) ** 2) / max_len * (a - b) + b

    pattern = np.stack([mask for i in range(channels)], axis=0)
    return np.ascontiguousarray(pattern, dtype=np.float32)


class IlluminationPattern(Dataset):
    """
    Args:
        pattern_shape: (channels, height, width)
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
        # convert pixel value range from [0,1] to [0.7, 1.3]
        self.normalize = transforms.Normalize(-7/6, 10/6)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        pattern = Image.open(path).convert('L').resize(self.pattern_size[::-1])
        return self.normalize(self.to_tensor(pattern))


def run():
    import pdb
    from PIL import Image

    # dataset = IlluminationPattern((3, 128, 128))
    # for batch in dataset:
    #     pdb.set_trace()
    #     print('pause')
    # pass
    save_dir = '/sda1/Datasets/IlluminationPattern/images/'
    nums = 25000
    for i in range(nums):
        c = random.randint(0, 1)
        illu_pattern = get_illumination_pattern(c, (3, 256, 256))[0]
        illu_pattern = (illu_pattern - illu_pattern.max()) / (illu_pattern.max() - illu_pattern.min())
        illu_pattern = (illu_pattern * 255).astype(np.uint8)
        illu_pattern = Image.fromarray(illu_pattern)
        illu_pattern.save(f'{save_dir}/{i:06d}.jpeg')


if __name__ == '__main__':
    run()
