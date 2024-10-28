#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : ScreenShootResilient
# @File         : str_bit_accuracy.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/10 21:14

# Import lib here
import torch
from torchmetrics import Metric
from typing import List, Optional


class StrBitAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('corr_bits', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total_bits', default=torch.tensor(0), dist_reduce_fx='sum')

        is_differentiable: Optional[bool] = False
        higher_is_better: Optional[bool] = False
        full_state_update: bool = False

    def update(self, batch_msg_hat: List[List[bytes]], batch_msg_gt: List[List[bytes]]) -> None:
        for i_msg_hat, i_msg_gt in zip(batch_msg_hat, batch_msg_gt):
            # Iter batch
            if i_msg_hat:
                # Get the first data if there exist
                msg_hat = i_msg_hat[0].decode('utf-8', errors='ignore')
            else:
                msg_hat = ''
            if i_msg_gt:
                msg_gt = i_msg_gt[0].decode('utf-8', errors='ignore')
            else:
                msg_gt = ''

            if msg_gt and not msg_hat:
                msg_hat = '\x00' * len(msg_gt)

            if len(msg_gt) != len(msg_hat):
                continue

            # for bit_hat, bit_gt in zip(msg_hat, msg_gt):
            #     if bit_hat == bit_gt:
            #         self.corr_bits += 1
            msg_hat_bits = ''.join(format(ord(char), '08b') for char in msg_hat)
            msg_gt_bits = ''.join(format(ord(char), '08b') for char in msg_gt)
            # print(msg_hat_bits, len(msg_hat_bits))
            # print(msg_gt_bits)

            self.corr_bits += sum(bit_hat == bit_gt for bit_hat, bit_gt in zip(msg_hat_bits, msg_gt_bits))
            self.total_bits += len(msg_gt) * 8

    def compute(self) -> float:
        if self.total_bits == 0:
            return 0.
        return self.corr_bits / self.total_bits


def run():
    str_bit_acc = StrBitAccuracy()
    str_bit_acc.update([[b'abcdefghijkl'], []], [[b'abcdefghijkl'], [b'abcdefghijkl']])
    print(str_bit_acc.total_bits, str_bit_acc.corr_bits)
    print(str_bit_acc.compute())
    pass


if __name__ == '__main__':
    run()
