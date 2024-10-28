#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : no_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/4/7 23:04

# Import lib here
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, host: torch.Tensor, secret: torch.Tensor, normalize: bool = False):
        return secret


def run():
    pass


if __name__ == '__main__':
    run()
