#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : dct.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/13 10:29

# Import lib here
import numpy as np
import torch
import torch.nn as nn


def get_dct_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == 0:
                weight = np.sqrt(1. / size)
            else:
                weight = np.sqrt(2. / size)
            matrix[i, j] = weight * np.cos(np.pi*(j+0.5)*i/size)
    return matrix


class DiscreteCosineTransform(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        dct_matrix = torch.tensor(get_dct_matrix(size), dtype=torch.float32)
        self.dct_matrix = nn.Parameter(dct_matrix, requires_grad=False)
        self.dct_matrix_t = nn.Parameter(dct_matrix.T, requires_grad=False)

    def forward(self, inputs: torch.Tensor, inverse: bool = False):
        if not inverse:
            outputs = self.dct_matrix @ inputs @ self.dct_matrix_t
        else:
            outputs = self.dct_matrix_t @ inputs @ self.dct_matrix
        return outputs


class InverseDiscreteCosineTransform(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        dct_matrix = torch.tensor(get_dct_matrix(size), dtype=torch.float32)
        self.dct_matrix = nn.Parameter(dct_matrix, requires_grad=False)
        self.dct_matrix_t = nn.Parameter(dct_matrix.T, requires_grad=False)

    def forward(self, inputs: torch.Tensor):
        outputs = self.dct_matrix_t @ inputs @ self.dct_matrix
        return outputs


def run():
    import pdb
    from PIL import Image
    dct = DiscreteCosineTransform(128)
    idct = InverseDiscreteCosineTransform(128)
    qrcode_img_path = '/home/chengxin/Desktop/Accept_qrcode.png'
    qrcode_img = np.array(Image.open(qrcode_img_path).convert('L').resize((128, 128)))

    qrcode_coef = dct(qrcode_img)
    pdb.set_trace()
    print('yes')
    pass


if __name__ == '__main__':
    run()
