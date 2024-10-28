#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : mask_par.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/4/5 23:56

# Import lib here
import torch
from torchmetrics import Metric
from typing import List, Optional


class MaskPAR(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('corr_bits', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total_bits', default=torch.tensor(0), dist_reduce_fx='sum')

        is_differentiable: Optional[bool] = False
        higher_is_better: Optional[bool] = False
        full_state_update: bool = False

    def update(self, inputs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        # Att: target must be filled with 1. after perspective transform
        m_inputs = inputs * mask
        eq = m_inputs.eq(target)
        self.corr_bits += torch.sum(eq)
        self.total_bits += torch.sum(mask)

    def compute(self) -> float:
        if self.total_bits == 0:
            return 0.
        return self.corr_bits / self.total_bits


def run():
    pass


if __name__ == '__main__':
    run()
