#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ScreenShootResilient
# @File         : host_barcode_illum_moire.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/10/26 14:07
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any

import torch
import numpy as np
import random
import string
import torchvision.transforms.functional as tvf
from pathlib import Path
from pylibdmtx import pylibdmtx
from torch.utils.data import Dataset
from PIL import Image


class HostBarcodeIllumMoire(Dataset):
    def __init__(self, host_dir: str,
                 illum_dir: str,
                 moire_dir: str,
                 msg_len: int = 12,
                 barcode_size: Tuple[int, int] = (128, 128),
                 img_size: Tuple[int, int] = (256, 256),
                 stage: str = 'train'):
        super().__init__()
        self.msg_len = msg_len
        self.barcode_size = barcode_size
        self.img_size = img_size

        host_paths = sorted(path for path in Path(host_dir).glob('*.jpg'))
        self.illum_paths = sorted(path for path in Path(illum_dir).glob('*.jpeg'))
        self.moire_paths = sorted(path for path in Path(moire_dir).glob('*.jpeg'))
        num_host = len(host_paths)
        if stage == 'train':
            self.host_paths = host_paths[:int(num_host*0.9)]
        elif stage == 'val':
            self.host_paths = host_paths[int(num_host*0.9):int(num_host*0.95)]
        elif stage == 'test':
            self.host_paths = host_paths[int(num_host*0.95):]
        else:
            raise ValueError
        self.ascii_list = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)

    def __getitem__(self, item):
        # Get host image
        host_path = self.host_paths[item]
        host_img = Image.open(host_path).convert('RGB').resize(self.img_size)
        host = tvf.to_tensor(host_img)

        # Get barcode image
        msg = random.choices(self.ascii_list, k=self.msg_len)
        msg_byte = ''.join(msg).encode('utf-8')
        msg = torch.tensor(list(msg_byte))
        code_img = pylibdmtx.encode(msg_byte, size='16x16')
        code_img = Image.frombytes('RGB', (code_img.width, code_img.height), code_img.pixels).convert('L')
        code_img = code_img.crop((10, 10, 90, 90)).resize(self.barcode_size, resample=0)
        code = tvf.to_tensor(code_img)

        # Get illumination pattern
        illum_path = self.illum_paths[item % len(self.illum_paths)]
        illum_img = Image.open(illum_path).convert('L').resize(self.img_size)
        illum = tvf.normalize(tvf.to_tensor(illum_img), [-7./6], [10./6], inplace=True)

        # Get moire pattern
        moire_path = self.moire_paths[item % len(self.moire_paths)]
        moire_img = Image.open(moire_path).resize(self.img_size)
        moire = tvf.to_tensor(moire_img)

        return host, msg, code, illum, moire

    def __len__(self):
        return len(self.host_paths)


def run():
    pass


if __name__ == '__main__':
    run()
