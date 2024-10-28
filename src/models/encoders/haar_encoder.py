#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : haar_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : https://github.com/ilsang/PyTorch-SE-Segmentation/blob/master/model.py
# @CreateTime   : 2023/3/16 21:06

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

from src.models.freq_transform.haar import Haar, InverseHaar
from segmentation_models_pytorch import Unet
import pdb


class HaarEncoder(nn.Module):
    def __init__(self, strength_factor: float = 1.):
        super().__init__()
        self.strength_factor = strength_factor
        self.haar = Haar(3)
        self.channel_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(12, 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 12, kernel_size=1),
            nn.Sigmoid())
        self.channel_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(12, 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 12, kernel_size=1),
            nn.Sigmoid())

        self.unet = Unet(
            encoder_name='timm-efficientnet-b1',
            encoder_weights='imagenet',
            in_channels=3 * 4,
            classes=3 * 4,
        )

        self.img_norm = Normalize(mean=[0.4413, 0.4107, 0.3727], std=[0.2435, 0.2338, 0.2316])
        self.code_norm = Normalize(0.5, 0.5)

    def forward(self, host: torch.Tensor, secret: torch.Tensor, normalize: bool = False):
        if normalize:
            host = self.img_norm(host)
            secret = self.code_norm(secret)

        # haar transform: B, 3, H, W -> B, 12, H//2, W//2
        # R_{ll}G_{ll}B_{ll}R_{hl}G_{hl}B_{hl}R_{lh}G_{lh}B_{lh}R_{hh}G_{hh}B_{hh}
        host_haar = self.haar(host)
        secret_haar = self.haar(secret)
        att_host_haar = host_haar + host_haar * self.channel_att1(host_haar)
        att_secret_haar = secret_haar + secret_haar * self.channel_att2(secret_haar)
        haar = self.unet(att_host_haar + att_secret_haar)
        container = self.haar(host_haar + haar * self.strength_factor, reverse=True)
        return container.clamp(0, 1)


class HaarEncoder2(nn.Module):
    def __init__(self, in_channels: int = 6,
                 out_channels: int = 3,
                 init_features: int = 32,
                 network_depth: int = 4,
                 bottleneck_layers: int = 1,
                 reduction_ratio: int = 16,
                 haar_transform: bool = True,
                 se_block: bool = True,
                 ckpt_path: str = None):
        super(HaarEncoder2, self).__init__()

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

        # self.img_norm = Normalize(mean=[0.4413, 0.4107, 0.3727], std=[0.2435, 0.2338, 0.2316])
        self.img_norm = Normalize(0.5, 0.5)
        self.code_norm = Normalize(0.5, 0.5)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path)['state_dict'])

    def forward(self, host: torch.Tensor, secret: torch.Tensor, normalize: bool = False):
        if normalize:
            host = self.img_norm(host)
            secret = self.code_norm(secret)

        x = torch.cat([host, secret], dim=1)

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


class UNetDec(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 reduction_ratio,
                 haar_transform: bool = False,
                 se_block: bool = False):
        super(UNetDec, self).__init__()

        up = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]

        if haar_transform:
            up += [
                nn.Conv2d(out_channels, out_channels * 4, 1),
                nn.ReLU(inplace=True),
                InverseHaar(out_channels * 4)
            ]
        else:
            up += [
                nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2),
                nn.ReLU(inplace=True)
            ]

        self.up = nn.Sequential(*up)

        self.se_block = se_block
        if se_block:
            self.excitation1 = nn.Conv2d(out_channels, out_channels // reduction_ratio, kernel_size=1)
            self.excitation2 = nn.Conv2d(out_channels // reduction_ratio, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.up(x)
        if self.se_block:
            fm_size = out.size()[2]
            scale_weight = F.avg_pool2d(out, fm_size)
            scale_weight = torch.relu(self.excitation1(scale_weight))
            scale_weight = torch.sigmoid(self.excitation2(scale_weight))
            out = out * scale_weight.expand_as(out)
        return out


class UNetEnc(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 reduction_ratio: int,
                 haar_transform: bool = False,
                 se_block: bool = False,
                 dropout: bool = False):
        super(UNetEnc, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if haar_transform:
            layers += [
                Haar(out_channels),
                nn.Conv2d(out_channels * 4, out_channels, 1),
                nn.ReLU(inplace=True)
            ]
        else:
            layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        if dropout:
            layers += [nn.Dropout(.5)]

        self.down = nn.Sequential(*layers)

        self.se_block = se_block
        if se_block:
            self.excitation1 = nn.Conv2d(out_channels, out_channels // reduction_ratio, kernel_size=1)
            self.excitation2 = nn.Conv2d(out_channels // reduction_ratio, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.down(x)
        if self.se_block:
            fm_size = out.size()[2]
            scale_weight = F.avg_pool2d(out, fm_size)
            scale_weight = torch.relu(self.excitation1(scale_weight))
            scale_weight = torch.sigmoid(self.excitation2(scale_weight))
            out = out * scale_weight.expand_as(out)
        return out


class Bottleneck_block(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 dilation_rate: int,
                 reduction_ratio: int,
                 dropout: bool = False,
                 se_block: bool = False):
        super(Bottleneck_block, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]

        if dropout:
            layers += [nn.Dropout(.5)]

        self.center = nn.Sequential(*layers)
        self.se_block = se_block

        if self.se_block:
            self.excitation1 = nn.Conv2d(out_channels, out_channels // reduction_ratio, kernel_size=1)
            self.excitation2 = nn.Conv2d(out_channels // reduction_ratio, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.center(x)
        if self.se_block:
            fm_size = out.size()[2]
            scale_weight = F.avg_pool2d(out, fm_size)
            scale_weight = torch.relu(self.excitation1(scale_weight))
            scale_weight = torch.sigmoid(self.excitation2(scale_weight))
            out = out * scale_weight.expand_as(out)
        return out


class HaarEncoder3(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, network_depth: int = 4, bottleneck_layers: int = 2,
                 reduction_ratio: int = 4, dropout: bool = False):
        super(HaarEncoder3, self).__init__()
        pass

    def forward(self, host: torch.Tensor, secret: torch.Tensor):
        pass


def run():
    from torchinfo import summary
    img = torch.rand(4, 3, 128, 128)
    qrcode = torch.rand(4, 3, 128, 128)
    # haar_encoder = HaarEncoder()
    # summary(haar_encoder, input_data=[img, qrcode])
    haar_encoder2 = HaarEncoder2(in_channels=6, out_channels=3)
    summary(haar_encoder2, input_data=[img, qrcode])
    pass


if __name__ == '__main__':
    run()
