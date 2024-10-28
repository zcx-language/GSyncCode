#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : l1_loss.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/28 09:58

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, host: torch.Tensor, secret: torch.Tensor):
        return F.l1_loss(host, secret)


def run():
    pass


if __name__ == '__main__':
    run()
