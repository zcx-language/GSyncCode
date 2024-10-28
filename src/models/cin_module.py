#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : cin_module.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/21 15:06

# Import lib here
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanAbsoluteError

from omegaconf import DictConfig
from typing import Any, List, Dict, Optional
import pdb

from src.models.metrics import StrBitAccuracy
from src.utils.qrcode_tools import batch_qrcode_decode
from src.utils.gan_tools import ImagePool, cal_gradient_penalty


class CINModule(LightningModule):
    """
    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, region_selector: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 discriminator: nn.Module,
                 augmenter: nn.Module,
                 loss_cfg: DictConfig,
                 model_cfg: DictConfig):
        super().__init__()
        torch.set_float32_matmul_precision('medium')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(model_cfg, logger=False)
        # print(self.hparams)
        # pdb.set_trace()

        self.region_selector = region_selector
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.augmenter = augmenter
        self.loss_cfg = loss_cfg
        self.model_cfg = model_cfg

        # Utils
        self.encoded_region_pool = ImagePool()

        # Metric objects for calculating and averaging accuracy across batches
        self.train_host2container_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.train_secret2rev_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.train_host2rev_psnr = PeakSignalNoiseRatio(data_range=1.)

        self.valid_host2container_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.valid_secret2rev_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.valid_secret2rev_par = BinaryAccuracy()
        self.valid_host2rev_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.valid_str_acc = StrBitAccuracy()

        self.test_host2container_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.test_secret2rev_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.test_secret2rev_par = BinaryAccuracy()
        self.test_host2rev_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.test_str_acc = StrBitAccuracy()

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def encode(self, image: torch.Tensor, qrcode: torch.Tensor):
        assert image.min() >= 0. and image.max() <= 1., \
            f'Error, Expect input image ranged [0, 1], but got range({image.min()}, {image.max()}).'
        assert qrcode.min() >= 0. and qrcode.max() <= 1., \
            f'Error, Expect input qrcode ranged [0, 1], but got range({qrcode.min()}, {qrcode.max()}).'
        orig_device = image.device
        image, qrcode = image.to(self.device), qrcode.to(self.device)
        if image.ndim == 3:
            image = image.unsqueeze(dim=0)
        if qrcode.ndim == 3:
            qrcode = qrcode.unsqueeze(dim=0)

        region, top_left_pt = self.region_selector(image)
        region_height, region_width = region.shape[-2:]
        encoded_region = self.encoder(region, qrcode, normalize=True).clamp(0, 1)
        container = torch.clone(image)
        for batch_idx in range(image.shape[0]):
            h_idx = top_left_pt[batch_idx, 0]
            w_idx = top_left_pt[batch_idx, 1]
            container[batch_idx, :, h_idx:h_idx+region_height, w_idx:w_idx+region_width] = encoded_region[batch_idx]

        container = container.squeeze(dim=0).to(orig_device)
        top_left_pt = top_left_pt.to(orig_device)
        return container, top_left_pt

    def decode(self, image: torch.Tensor):
        assert image.min() >= 0. and image.max() <= 1., \
            f'Error, Expect input image ranged [0, 1], but got range({image.min()}, {image.max()}).'
        orig_device = image.device
        image = image.to(self.device)
        if image.ndim == 3:
            image = image.unsqueeze(dim=0)
        results = self.decoder(image, normalize=True).squeeze(dim=0).clamp(0, 1)
        return results.to(orig_device)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.valid_host2container_psnr.reset()
        self.valid_host2rev_psnr.reset()
        self.valid_secret2rev_psnr.reset()
        self.valid_str_acc.reset()

    def shared_step(self, batch: Any):
        if isinstance(batch, Dict):
            host, secret = batch['mirflickr_qrcode']
            illumination_pattern = batch['illumination_pattern']
            moire_pattern = batch['moire_pattern']
        else:
            host, secret = batch
            illumination_pattern = None
            moire_pattern = None

        # Convert gray to rgb
        if secret.shape[1] == 1:
            secret = torch.cat([secret, secret, secret], dim=1)

        # Encode
        region, top_left_pt = self.region_selector(host)
        encoded_region = self.encoder(region, secret, normalize=True)

        container = torch.clone(host)
        host_gt = torch.ones_like(host)
        secret_gt = torch.ones_like(host)
        region_height, region_width = region.shape[2:]
        for batch_idx in range(host.shape[0]):
            h_idx = top_left_pt[batch_idx, 0]
            w_idx = top_left_pt[batch_idx, 1]
            container[batch_idx, :, h_idx:h_idx + region_height, w_idx:w_idx + region_width] = encoded_region[batch_idx]
            host_gt[batch_idx, :, h_idx:h_idx + region_height, w_idx:w_idx + region_width] = region[batch_idx]
            secret_gt[batch_idx, :, h_idx:h_idx + region_height, w_idx:w_idx + region_width] = secret[batch_idx]

        # Augment
        aug_container = self.augmenter(container, {'illumination': illumination_pattern,
                                                   'moire': moire_pattern})
        host_gt = self.augmenter.perspective(host_gt, params=self.augmenter.perspective._params)
        secret_gt = self.augmenter.perspective(secret_gt, params=self.augmenter.perspective._params)

        # Resize for decoding
        aug_container = F.interpolate(aug_container, size=secret.shape[-2:], mode='nearest')
        host_gt = F.interpolate(host_gt, size=secret.shape[-2:], mode='nearest')
        secret_gt = F.interpolate(secret_gt, size=secret.shape[-2:], mode='nearest')

        # Decode
        rev_host, rev_secret = self.decoder(self.encoder, aug_container, normalize=True)
        return host, secret, container, aug_container, host_gt, secret_gt, rev_host, rev_secret

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        current_batch_idx = self.trainer.fit_loop.total_batch_idx
        host, secret, container, aug_container, host_gt, secret_gt, rev_host, rev_secret = self.shared_step(batch)

        # Calculate loss
        if optimizer_idx == 0:
            loss = self.cal_disc_loss(host, container)
        elif optimizer_idx == 1:
            loss = self.cal_enc_dec_loss(host, secret, container, aug_container,
                                         host_gt, secret_gt, rev_host, rev_secret)
        else:
            raise ValueError

        # Update and log metrics
        with torch.no_grad():
            host2container_psnr = self.train_host2container_psnr(container, host)
            host2rev_psnr = self.train_host2rev_psnr(rev_host, host_gt)
            secret2rev_psnr = self.train_secret2rev_psnr(rev_secret, secret_gt)
        self.log('train/host2container_psnr', host2container_psnr)
        self.log('train/host2rev_psnr', host2rev_psnr)
        self.log('train/secret2rev_psnr', secret2rev_psnr)

        # Visualize image
        if current_batch_idx % 3000 == 0:
            show_image = dict(
                host=host[0],
                secret=secret[0],
                container=container[0],
                aug_container=aug_container[0],
                host_gt=host_gt[0],
                secret_gt=secret_gt[0],
                rev_host=rev_host[0],
                rev_secret=rev_secret[0]
            )
            self.visualize2logger('train', current_batch_idx, show_image)
        return loss

    def cal_disc_loss(self, host, container):
        real_pred = self.discriminator(host)
        container = self.encoded_region_pool.push_and_pop(container.detach())
        fake_pred = self.discriminator(container)
        gradient_penalty = cal_gradient_penalty(self.discriminator, host, container)[0]
        loss = -torch.mean(real_pred) + torch.mean(fake_pred) + gradient_penalty
        self.log('train/disc_loss', loss)
        return loss

    def cal_enc_dec_loss(self, host, secret, container, aug_container, host_gt, secret_gt, rev_host, rev_secret):
        # Generator loss
        gen_loss = -torch.mean(self.discriminator(container))

        host2container_loss = F.l1_loss(container, host)
        host2rev_loss = F.l1_loss(rev_host, host_gt)
        secret2rev_loss = F.l1_loss(rev_secret, secret_gt)

        loss = (host2container_loss * self.loss_cfg.host2container_weight +
                host2rev_loss * self.loss_cfg.host2rev_weight +
                secret2rev_loss * self.loss_cfg.secret2rev_weight +
                gen_loss * self.loss_cfg.gen_weight)
        self.log('train/enc_dec_loss', loss)
        self.log('train/host2container_loss', host2container_loss)
        self.log('train/host2rev_loss', host2rev_loss)
        self.log('train/secret2rev_loss', secret2rev_loss)
        return loss

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        host, secret, container, aug_container, host_gt, secret_gt, rev_host, rev_secret = self.shared_step(batch)

        # Update and log metrics
        with torch.no_grad():
            self.valid_host2container_psnr.update(container, host)
            self.valid_host2rev_psnr.update(rev_host, host_gt)
            self.valid_secret2rev_psnr.update(rev_secret, secret_gt)

        # Visualize image
        if batch_idx == 0:
            show_image = dict(
                host=host[0],
                secret=secret[0],
                container=container[0],
                aug_container=aug_container[0],
                host_gt=host_gt[0],
                secret_gt=secret_gt[0],
                rev_host=rev_host[0],
                rev_secret=rev_secret[0]
            )
            self.visualize2logger('valid', self.current_epoch, show_image)

        # evaluate decode accuracy
        msg_hats, _ = batch_qrcode_decode(torch.mean(rev_secret, dim=1, keepdim=True).gt(0.5).to(torch.float32))
        msg_gts, _ = batch_qrcode_decode(torch.mean(secret_gt, dim=1, keepdim=True).gt(0.5).to(torch.float32))
        # assert len(msg_hats) == len(msg_true_gts) and len(msg_gts) == len(msg_true_gts), \
        #     f'Error, {len(msg_hats)} vs. {len(msg_true_gts)} vs. {len(msg_gts)}'
        self.valid_str_acc.update(msg_hats, msg_gts)
        # self.valid_str_acc_bound.update(msg_gts, msg_true_gts)

    def validation_epoch_end(self, outputs: List[Any]):
        host2container_psnr = self.valid_host2container_psnr.compute()
        host2rev_psnr = self.valid_host2rev_psnr.compute()
        secret2rev_psnr = self.valid_secret2rev_psnr.compute()
        str_acc = self.valid_str_acc.compute()

        self.logger_instance.add_scalar('valid/host2container_psnr', host2container_psnr, self.current_epoch)
        # self.logger_instance.add_scalar('valid/ssim', ssim, self.current_epoch)
        self.logger_instance.add_scalar('valid/host2rev_psnr', host2rev_psnr, self.current_epoch)
        self.logger_instance.add_scalar('valid/secret2rev_psnr', secret2rev_psnr, self.current_epoch)
        self.logger_instance.add_scalar('valid/str_acc', str_acc, self.current_epoch)

        # Log for checkpoint
        avg_psnr = (host2container_psnr + secret2rev_psnr) / 2.
        self.log('valid/avg_psnr', avg_psnr)

    def test_step(self, batch: Any, batch_idx: int):
        host, secret, container, aug_container, host_gt, secret_gt, rev_host, rev_secret = self.shared_step(batch)

        # Update and log metrics
        with torch.no_grad():
            self.test_host2container_psnr.update(container, host)
            self.test_host2rev_psnr.update(rev_host, host_gt)
            self.test_secret2rev_psnr.update(rev_secret, secret_gt)

        # Visualize image
        if batch_idx % 10 == 0:
            show_image = dict(
                host=host[0],
                secret=secret[0],
                container=container[0],
                aug_container=aug_container[0],
                host_gt=host_gt[0],
                secret_gt=secret_gt[0],
                rev_host=rev_host[0],
                rev_secret=rev_secret[0]
            )
            self.visualize2logger('test', batch_idx, show_image)

        # Test decode accuracy
        msg_hats, _ = batch_qrcode_decode(torch.mean(rev_secret, dim=1, keepdim=True).ge(0.5).to(torch.float32))
        msg_gts, _ = batch_qrcode_decode(torch.mean(secret, dim=1, keepdim=True).ge(0.5).to(torch.float32))
        # assert len(msg_hats) == len(msg_true_gts) and len(msg_gts) == len(msg_true_gts), \
        #     f'Error, {len(msg_hats)} vs. {len(msg_true_gts)} vs. {len(msg_gts)}'
        self.test_str_acc.update(msg_hats, msg_gts)
        # self.valid_str_acc_bound.update(msg_gts, msg_true_gts)

    def test_epoch_end(self, outputs: List[Any]):
        host2container_psnr = self.test_host2container_psnr.compute()
        host2rev_psnr = self.test_host2rev_psnr.compute()
        secret2rev_psnr = self.test_secret2rev_psnr.compute()
        str_acc = self.test_str_acc.compute()

        self.logger_instance.add_scalar('test/host2container_psnr', host2container_psnr, self.current_epoch)
        # self.logger_instance.add_scalar('valid/ssim', ssim, self.current_epoch)
        self.logger_instance.add_scalar('test/host2rev_psnr', host2rev_psnr, self.current_epoch)
        self.logger_instance.add_scalar('test/secret2rev_psnr', secret2rev_psnr, self.current_epoch)
        self.logger_instance.add_scalar('test/str_acc', str_acc, self.current_epoch)

    def configure_optimizers(self):
        """
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        enc_dec_optim = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ], lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)

        disc_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
        return {'optimizer': disc_optim}, {'optimizer': enc_dec_optim}

    @property
    def logger_instance(self):
        return self.logger.experiment

    def visualize2logger(self, stage: str, step: int, image_dict: dict):
        for label, image in image_dict.items():
            self.logger_instance.add_image(f'{stage}/{label}', image, step)


def run():
    pass


if __name__ == '__main__':
    run()
