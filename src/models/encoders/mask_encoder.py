#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : mask_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/4/11 00:02

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from segmentation_models_pytorch import Unet

from src.models.freq_transform.haar import Haar, InverseHaar
from src.models.components import InvArch

from typing import Tuple, List, Optional


class MaskEncoder1(nn.Module):
    def __init__(self, in_channels: int = 6,
                 out_channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1),
        )

        self.img_norm = Normalize(mean=[0.4413, 0.4107, 0.3727], std=[0.2435, 0.2338, 0.2316])
        self.code_norm = Normalize(0.5, 0.5)

    def forward(self, host: torch.Tensor, secret: torch.Tensor, normalize: bool = False):
        if normalize:
            host = self.img_norm(host)
            secret = self.code_norm(secret)
        residual = self.model(torch.cat([host, secret], dim=1))
        return residual


class MaskEncoderUNet(nn.Module):
    def __init__(self, in_channels: int = 6,
                 out_channels: int = 3,
                 depth: int = 4,
                 decoder_channels: Tuple = (128, 64, 32, 16),
                 ckpt_path: Optional[str] = None):
        super().__init__()

        self.unet = Unet(
            encoder_name='timm-efficientnet-b1',
            encoder_depth=depth,
            encoder_weights='imagenet',
            decoder_channels=decoder_channels,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

        self.img_norm = Normalize(mean=[0.4413, 0.4107, 0.3727], std=[0.2435, 0.2338, 0.2316])
        self.code_norm = Normalize(0.5, 0.5)

        if ckpt_path is not None:
            self.unet.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=True)

    def forward(self, host: torch.Tensor, secret: torch.Tensor, normalize: bool = False):
        if normalize:
            host = self.img_norm(host)
            secret = self.code_norm(secret)

        residual = self.unet(torch.cat([host, secret], dim=1))
        return residual


class MaskEncoderInvNet(nn.Module):
    def __init__(self, in_channels: int = 3,
                 n_inv_blocks: int = 6):
        super().__init__()
        self.haar = Haar(in_channels)
        self.inv_blocks = nn.ModuleList([InvArch(in_channels*4, in_channels*4) for _ in range(n_inv_blocks)])
        self.in_channels = in_channels

        self.img_norm = Normalize(mean=[0.4413, 0.4107, 0.3727], std=[0.2435, 0.2338, 0.2316])
        self.code_norm = Normalize(0.5, 0.5)

    def forward(self, host: torch.Tensor, secret: torch.Tensor, normalize: bool = False):
        if normalize:
            host = self.img_norm(host)
            secret = self.code_norm(secret)

        down_host = self.haar(host)
        down_secret = self.haar(secret)
        fusion = torch.cat([down_host, down_secret], dim=1)

        for blk in self.inv_blocks:
            fusion = blk(fusion, rev=False)

        en_secret = fusion[:, self.in_channels*4:]
        residual = self.haar(en_secret, reverse=True)
        return residual


class MaskEncoderHaarUNet(nn.Module):
    def __init__(self, in_channels: int = 6,
                 out_channels: int = 3,
                 init_features: int = 32,
                 network_depth: int = 4,
                 bottleneck_layers: int = 1,
                 reduction_ratio: int = 16,
                 ckpt_path: str = None):
        super().__init__()

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
            self.encodingBlocks.append(UNetEnc(features, 2 * features, reduction_ratio=self.reduction_ratio))
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
                                               reduction_ratio=self.reduction_ratio))
            prev_deconv_channels = skip_connection_channel_counts[i]

        self.final = nn.Conv2d(2 * init_features, out_channels, 1)

        self.img_norm = Normalize(mean=[0.4413, 0.4107, 0.3727], std=[0.2435, 0.2338, 0.2316])
        self.code_norm = Normalize(0.5, 0.5)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)

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

        residual = self.final(out)
        return residual


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, reduction_ratio):
        super(UNetDec, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels*4, 1),
            nn.ReLU(inplace=True),
            InverseHaar(out_channels*4),
        )

        self.excitation1 = nn.Conv2d(out_channels, out_channels // reduction_ratio, kernel_size=1)
        self.excitation2 = nn.Conv2d(out_channels // reduction_ratio, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.up(x)
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
                 dropout: bool = False):
        super(UNetEnc, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Haar(out_channels),
            nn.Conv2d(out_channels*4, out_channels, 1),
            nn.ReLU(inplace=True),

        ]
        if dropout:
            layers += [nn.Dropout(.5)]

        self.down = nn.Sequential(*layers)
        self.excitation1 = nn.Conv2d(out_channels, out_channels // reduction_ratio, kernel_size=1)
        self.excitation2 = nn.Conv2d(out_channels // reduction_ratio, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.down(x)
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
                 dropout=False):
        super(Bottleneck_block, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]

        if dropout:
            layers += [nn.Dropout(.5)]

        self.center = nn.Sequential(*layers)
        self.excitation1 = nn.Conv2d(out_channels, out_channels // reduction_ratio, kernel_size=1)
        self.excitation2 = nn.Conv2d(out_channels // reduction_ratio, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.center(x)
        fm_size = out.size()[2]
        scale_weight = F.avg_pool2d(out, fm_size)
        scale_weight = torch.relu(self.excitation1(scale_weight))
        scale_weight = torch.sigmoid(self.excitation2(scale_weight))
        out = out * scale_weight.expand_as(out)
        return out


def run():
    from torchinfo import summary
    mask_encoder_unet = MaskEncoderUNet(in_channels=6, out_channels=3, depth=4, decoder_channels=(128, 64, 32, 16))
    summary(mask_encoder_unet, input_data=(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)),
            depth=5, device='cuda')
    mask_encoder_invnet = MaskEncoderInvNet(in_channels=3, n_inv_blocks=6)
    summary(mask_encoder_invnet, input_data=(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)),
            depth=5, device='cuda')
    pass


if __name__ == '__main__':
    run()
