#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : invertible_arch.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/21 10:15

# Import lib here
import torch
import torch.nn as nn

from .dense_block import DenseBlock


class InvArch(nn.Module):
    def __init__(self, split_len1, split_len2, clamp=1.0):
        super(InvArch, self).__init__()
        self.split_len1 = split_len1
        self.split_len2 = split_len2
        self.clamp = clamp

        self.F = DenseBlock(self.split_len2, self.split_len1)
        self.G = DenseBlock(self.split_len1, self.split_len2)
        self.H = DenseBlock(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        """
        param {x1} : image
        param {x2} : msg
        """
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)


def run():
    pass


if __name__ == '__main__':
    run()
