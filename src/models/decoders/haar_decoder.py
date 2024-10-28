#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : haar_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : https://github.com/ilsang/PyTorch-SE-Segmentation/blob/master/model.py
# @CreateTime   : 2023/3/16 22:17

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

from segmentation_models_pytorch import Unet
from src.models.freq_transform.haar import Haar, InverseHaar
from src.models.encoders.haar_encoder import UNetDec, UNetEnc, Bottleneck_block
import pdb


class HaarDecoder(nn.Module):
    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1):
        super().__init__()
        self.in_haar = Haar(in_channels)
        self.out_haar = Haar(out_channels)

        self.unet = Unet(
            encoder_name='timm-efficientnet-b1',
            encoder_weights='imagenet',
            in_channels=in_channels * 4,
            classes=out_channels * 4,
        )

    def forward(self, container: torch.Tensor, normalize: bool = False):
        if normalize:
            container = (container - 0.5) * 2.
        container = self.in_haar(container)
        logit = self.unet(container)
        logit = self.out_haar(logit, reverse=True)
        return logit


class HaarDecoder2(nn.Module):
    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1,
                 init_features: int = 32,
                 network_depth: int = 4,
                 bottleneck_layers: int = 1,
                 reduction_ratio: int = 16,
                 haar_transform: bool = True,
                 se_block: bool = True,
                 adaptor: bool = False,
                 ckpt_path: str = None):
        super(HaarDecoder2, self).__init__()

        self.reduction_ratio = reduction_ratio
        self.network_depth = network_depth
        self.bottleneck_layers = bottleneck_layers

        skip_connection_channel_counts = []

        self.first_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=init_features, kernel_size=3,
                                    stride=1, padding=1, bias=True)

        self.encodingBlocks = nn.ModuleList([])
        features = init_features
        for i in range(self.network_depth):
            self.encodingBlocks.append(UNetEnc(features, 2 * features,
                                               reduction_ratio=self.reduction_ratio,
                                               haar_transform=haar_transform,
                                               se_block=se_block))
            skip_connection_channel_counts.insert(0, 2 * features)
            features *= 2
        final_encoding_channels = skip_connection_channel_counts[0]

        self.bottleNecks = nn.ModuleList([])
        for i in range(self.bottleneck_layers):
            dilation_factor = 1
            self.bottleNecks.append(Bottleneck_block(final_encoding_channels,
                                                     final_encoding_channels, dilation_rate=dilation_factor,
                                                     reduction_ratio=self.reduction_ratio))

        self.decodingBlocks = nn.ModuleList([])
        for i in range(self.network_depth):
            if i == 0:
                prev_deconv_channels = final_encoding_channels
            self.decodingBlocks.append(UNetDec(prev_deconv_channels + skip_connection_channel_counts[i],
                                               skip_connection_channel_counts[i],
                                               reduction_ratio=self.reduction_ratio,
                                               haar_transform=haar_transform,
                                               se_block=se_block))
            prev_deconv_channels = skip_connection_channel_counts[i]

        self.final = nn.Conv2d(2 * init_features, out_channels, 1)

        if adaptor:
            self.adaptor = nn.Sequential(
                Bottleneck_block(final_encoding_channels, final_encoding_channels,
                                 dilation_rate=1, reduction_ratio=self.reduction_ratio),
                nn.Conv2d(final_encoding_channels, out_channels, 1),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        else:
            self.adaptor = None

        self.img_norm = Normalize(mean=[0.4413, 0.4107, 0.3727], std=[0.2435, 0.2338, 0.2316])

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path)['state_dict'])

    def forward(self, container: torch.Tensor, normalize: bool = False):
        if normalize:
            container = self.img_norm(container)

        x = container
        out = self.first_conv(x)
        skip_connections = []
        for i in range(self.network_depth):
            out = self.encodingBlocks[i](out)
            skip_connections.append(out)

        for i in range(self.bottleneck_layers):
            out = self.bottleNecks[i](out)

        for i in range(self.network_depth):
            skip = skip_connections.pop()
            out = self.decodingBlocks[i](torch.cat([out, skip], 1))

        out = self.final(out)
        return out


class HaarDecoder3(nn.Module):
    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1,
                 init_features: int = 32,
                 network_depth: int = 4,
                 bottleneck_layers: int = 1,
                 reduction_ratio: int = 16,
                 haar_transform: bool = True,
                 se_block: bool = True,
                 num_half_down: int = 0,
                 ckpt_path: str = None):
        super(HaarDecoder3, self).__init__()

        self.reduction_ratio = reduction_ratio
        self.network_depth = network_depth
        self.bottleneck_layers = bottleneck_layers
        self.num_half_down = num_half_down

        skip_connection_channel_counts = []

        self.first_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=init_features, kernel_size=3,
                                    stride=1, padding=1, bias=True)

        self.encodingBlocks = nn.ModuleList([])
        features = init_features
        for i in range(self.network_depth):
            self.encodingBlocks.append(UNetEnc(features, 2 * features,
                                               reduction_ratio=self.reduction_ratio,
                                               haar_transform=haar_transform,
                                               se_block=se_block))
            skip_connection_channel_counts.insert(0, 2 * features)
            features *= 2
        final_encoding_channels = skip_connection_channel_counts[0]

        self.bottleNecks = nn.ModuleList([])
        for i in range(self.bottleneck_layers):
            dilation_factor = 1
            self.bottleNecks.append(Bottleneck_block(final_encoding_channels,
                                                     final_encoding_channels, dilation_rate=dilation_factor,
                                                     reduction_ratio=self.reduction_ratio))

        self.decodingBlocks = nn.ModuleList([])
        for i in range(self.network_depth):
            if i == 0:
                prev_deconv_channels = final_encoding_channels
            self.decodingBlocks.append(UNetDec(prev_deconv_channels + skip_connection_channel_counts[i],
                                               skip_connection_channel_counts[i],
                                               reduction_ratio=self.reduction_ratio,
                                               haar_transform=haar_transform,
                                               se_block=se_block))
            prev_deconv_channels = skip_connection_channel_counts[i]

        # self.final = nn.Conv2d(2 * init_features, out_channels, 1)
        self.final = nn.Sequential(
            nn.Conv2d(init_features * 2**(self.num_half_down+1), 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

        # self.img_norm = Normalize(mean=[0.4413, 0.4107, 0.3727], std=[0.2435, 0.2338, 0.2316])
        self.img_norm = Normalize(0.5, 0.5)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path)['state_dict'])

    def forward(self, container: torch.Tensor, normalize: bool = False):
        if normalize:
            container = self.img_norm(container)

        x = container
        out = self.first_conv(x)
        skip_connections = []
        for i in range(self.network_depth):
            out = self.encodingBlocks[i](out)
            skip_connections.append(out)

        for i in range(self.bottleneck_layers):
            out = self.bottleNecks[i](out)

        for i in range(self.network_depth - self.num_half_down):
            skip = skip_connections.pop()
            out = self.decodingBlocks[i](torch.cat([out, skip], 1))

        out = self.final(out)
        return F.interpolate(out, scale_factor=2**self.num_half_down)


def run():
    from torchinfo import summary
    haar_decoder = HaarDecoder3(out_channels=1, num_half_down=3)
    summary(haar_decoder, input_size=(4, 3, 256, 256))
    pass


if __name__ == '__main__':
    run()
