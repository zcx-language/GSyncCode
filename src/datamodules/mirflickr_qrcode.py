#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
# @Project      : ScreenShootResilient
# @File         : mirflickr_qrcode.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/8 16:49

# Import lib here
import numpy as np
import string
import random
import torch
import qrcode
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from pathlib import Path
from pylibdmtx import pylibdmtx
from omegaconf import DictConfig

from typing import List, Tuple, Optional
import pdb


class MirflickrQRCode(Dataset):
    """
    Args:
        data_dir:
        stage:
        img_size:
        msg_len:
        qrcode_size:
    """

    def __init__(self, data_dir: str,
                 img_size: Tuple[int, int] = (256, 256),
                 msg_len: int = 12,
                 qrcode_size: Tuple[int, int] = (128, 128),
                 stage: str = 'train',
                 code_type: str = 'qrcode'):
        super().__init__()
        assert stage.lower() in ['train', 'valid', 'test']
        img_paths = sorted(str(path) for path in Path(data_dir).glob('*.jpg'))
        num_img = len(img_paths)
        if stage.lower() == 'valid':
            self.img_paths = img_paths[int(num_img*0.95):]
        elif stage.lower() == 'test':
            self.img_paths = img_paths[int(num_img*0.9):int(num_img*0.95)]
        else:
            self.img_paths = img_paths[:int(num_img*0.9)]

        self.img_size = img_size
        self.msg_len = msg_len
        self.qrcode_size = qrcode_size
        self.code_type = code_type

        self.qrcode_generator = qrcode.QRCode(version=1,
                                              error_correction=qrcode.constants.ERROR_CORRECT_H,
                                              box_size=self.qrcode_size[0]//21,
                                              border=0)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(0.5, 0.5)
        ])

    def __getitem__(self, item):
        # Generate image tensors
        img_path = self.img_paths[item]
        host = Image.open(img_path).resize(self.img_size[::-1])
        host = self.img_transform(host)

        # Generate code images randomly
        ascii_list = random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=self.msg_len)
        secret_byte = ''.join(ascii_list).encode('utf-8')
        if self.code_type == 'qrcode':
            # msg = np.random.binomial(1, .5, self.msg_len)
            self.qrcode_generator.clear()
            self.qrcode_generator.add_data(secret_byte)
            self.qrcode_generator.make(fit=True)
            code_img = self.qrcode_generator.make_image().resize(self.qrcode_size[::-1], resample=0)   # Nearest
            code_img = np.array(code_img, dtype=np.uint8) * 255
        elif self.code_type == 'dmtx':
            # 16x16: 100x100; 18x18: 110x110; 20x20: 120x120
            # one bit: 5x5 pixels
            encoded = pylibdmtx.encode(secret_byte, size='16x16')
            code_img = Image.frombytes('RGB', (encoded.width, encoded.height), encoded.pixels).convert('L')
            code_img = code_img.crop((10, 10, 90, 90))
            code_img = code_img.resize(self.qrcode_size[::-1], resample=0)  # Nearest
            code_img = np.array(code_img, dtype=np.uint8)
        else:
            raise ValueError

        # Generate Edge
        code_edge = cv2.Canny(code_img, 100, 200)
        # code_edge = cv2.GaussianBlur(code_edge, (3, 3), 0.5)
        kernel = np.ones((3, 3), dtype=np.uint8)
        code_edge = cv2.dilate(code_edge, kernel, 1)
        # code_edge = cv2.erode(code_edge, kernel, 1)
        # code_edge = cv2.
        # code_edge = 255 - code_edge

        # Generate vertex
        # dst = cv2.cornerHarris(code_edge, 2, 3, 0.04)
        # code_corner = np.zeros_like(code_edge)
        # code_corner[dst > 0.01 * dst.max()] = 255
        # corners = cv2.goodFeaturesToTrack(code_edge, maxCorners=100, qualityLevel=0.01, minDistance=10)

        code_img = self.img_transform(code_img)
        code_edge = self.img_transform(code_edge)
        # code_corner = self.img_transform(code_corner)
        secret_byte = torch.tensor(list(secret_byte))
        return host, code_img, code_edge, secret_byte

    def __len__(self):
        return len(self.img_paths)


def run():
    from src.utils.image_tools import image_show
    # denormalize = transforms.Normalize(-1, 2)
    dataset = MirflickrQRCode('../../datasets/mirflickr/', msg_len=12, code_type='dmtx')
    for batch in dataset:
        # image_show(batch[0])
        # image_show(batch[1])
        image_show(batch[2])
        # image_show(batch[3])
        input()
    pass


if __name__ == '__main__':
    run()
