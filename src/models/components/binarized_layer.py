#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : binary_layer.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/7/25 13:07

# Import lib here
import torch
import torch.nn as nn

from torch.autograd import Function


class BinarizedF(Function):
    def forward(self, input):
        self.save_for_backward(input)
        a = torch.ones_like(input)
        b = -torch.ones_like(input)
        output = torch.where(input >= 0, a, b)
        return output

    def backward(self, output_grad):
        input, = self.saved_tensors
        exp_i = torch.exp(input)
        exp_i_ = torch.exp(-input)
        tanh_i = torch.divide((exp_i - exp_i_), (exp_i + exp_i_))
        input_grad = output_grad * (1 - tanh_i.pow(2))
        return input_grad


class BinarizedLayer(nn.Module):
    def __init__(self):
        super(BinarizedLayer, self).__init__()

    def forward(self, x):
        return BinarizedF.apply(x)

    pass


def run():
    pass


if __name__ == '__main__':
    run()
