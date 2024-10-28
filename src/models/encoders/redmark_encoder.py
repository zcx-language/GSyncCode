#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : redmark_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/13 23:23

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.freq_transform.dct import DiscreteCosineTransform, InverseDiscreteCosineTransform


class ReDMarkEncoder(nn.Module):
    def __init__(self, input_size: tuple[int, int]):
        super().__init__()
        self.dct = DiscreteCosineTransform(input_size[-1])
        pass

    def forward(self, inputs):
        pass


def run():
    pass


if __name__ == '__main__':
    run()
