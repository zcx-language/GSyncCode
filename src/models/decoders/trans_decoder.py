#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : trans_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/4/3 23:07

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation
from src.models.components.segformer import SegFormerDecoder, SegFormerSegmentationHead


class TransDecoder(nn.Module):
    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1):
        super().__init__()
        self.encoder = SegformerForSemanticSegmentation.from_pretrained(
            'nvidia/mit-b1',
            num_labels=out_channels
        ).segformer.encoder

        self.decoder = SegFormerDecoder(
            out_channels=128,
            widths=[64, 128, 320, 512],
            scale_factors=[4, 8, 16, 32],
        )
        self.head = SegFormerSegmentationHead(
            channels=128,
            num_classes=out_channels,
            num_features=4
        )

    def forward(self, container: torch.Tensor, normalize: bool = False):
        if normalize:
            container = (container - 0.5) * 2.
        features = self.encoder(container, output_hidden_states=True, return_dict=True)['hidden_states']
        features = self.decoder(features)
        logits = self.head(features)
        return logits


def run():
    trans_decoder = TransDecoder()
    container = torch.randn(2, 3, 256, 256)
    print(trans_decoder(container).shape)
    pass


if __name__ == '__main__':
    run()
