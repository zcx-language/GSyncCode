#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : auto_warp.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/29 22:46

# Import lib here
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanAbsoluteError

from omegaconf import DictConfig
from typing import Any, List, Dict, Optional
import pdb

from src.models.metrics import StrBitAccuracy
from src.utils.qrcode_tools import batch_qrcode_decode
from src.utils.gan_tools import ImagePool, cal_gradient_penalty


class AutoWarp(LightningModule):
    """
    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, region_selector: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 corrector: Optional[nn.Module] = None,
                 discriminator: Optional[nn.Module] = None,
                 augmenter: Optional[nn.Module] = None,
                 loss_func: Optional[DictConfig] = None,
                 loss_cfg: Optional[DictConfig] = None,
                 model_cfg: Optional[DictConfig] = None):
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
        self.corrector = corrector
        self.discriminator = discriminator
        self.augmenter = augmenter
        self.loss_func = loss_func
        self.loss_cfg = loss_cfg
        self.model_cfg = model_cfg
        self.encode_weight = 0.

        # Utils
        self.encoded_region_pool = ImagePool()

        # Metric objects for calculating and averaging accuracy across batches
        self.train_region_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.train_secret_par = BinaryAccuracy(threshold=0.5)

        self.valid_region_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.valid_secret_par = BinaryAccuracy(threshold=0.5)
        # No need for validation, just test is ok.
        # self.valid_str_acc = StrBitAccuracy()

        # Used for choosing the best ckpt that can balance the metric between container and secret.
        self.valid_region_mae = MeanAbsoluteError()
        self.valid_secret_mae = MeanAbsoluteError()

        self.test_region_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.test_secret_par = BinaryAccuracy(threshold=0.5)
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

    def shared_step(self, batch: Any):
        if isinstance(batch, Dict):
            host, secret = batch['mirflickr_qrcode']
            illumination_pattern = batch['illumination_pattern']
            moire_pattern = batch['moire_pattern']
        else:
            host, secret = batch
            illumination_pattern = None
            moire_pattern = None

        # Region selection
        region, top_left_pt = self.region_selector(host)

        # Encode
        encoded_region = self.encoder(region, secret.repeat(1, 3, 1, 1), normalize=True).clamp(0, 1)

        # Assemble container, secret_gt
        container = torch.clone(host)
        secret_gt = torch.ones((host.shape[0], 1, host.shape[2], host.shape[3]),
                               dtype=torch.float32, device=container.device)
        region_height, region_width = region.shape[-2:]
        for batch_idx in range(host.shape[0]):
            h_idx = top_left_pt[batch_idx, 0]
            w_idx = top_left_pt[batch_idx, 1]
            container[batch_idx, :, h_idx:h_idx+region_height, w_idx:w_idx+region_width] = encoded_region[batch_idx]
            secret_gt[batch_idx, :, h_idx:h_idx+region_height, w_idx:w_idx+region_width] = secret[batch_idx]

        # Augment container
        aug_container = self.augmenter(container, {'illumination': illumination_pattern,
                                                   'moire': moire_pattern})

        # Assemble correct container ground truth
        cor_container_gt = self.augmenter.perspective.inverse(aug_container)

        # Correct
        cor_container = self.corrector(aug_container, normalize=True)

        # Decode
        secret_hat_logit = self.decoder(cor_container, normalize=True)
        return host, region, encoded_region, container, secret_gt, aug_container, cor_container, secret_hat_logit

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.valid_region_psnr.reset()
        self.valid_secret_par.reset()
        self.valid_region_mae.reset()
        self.valid_secret_mae.reset()

    def cal_disc_loss(self, encoded_region, region):
        real_pred = self.discriminator(region)
        encoded_region = self.encoded_region_pool.push_and_pop(encoded_region.detach())
        fake_pred = self.discriminator(encoded_region)
        gradient_penalty = cal_gradient_penalty(self.discriminator, region, encoded_region)[0]
        loss = -torch.mean(real_pred) + torch.mean(fake_pred) + gradient_penalty
        self.log('train/disc_loss', loss)
        return loss

    def cal_enc_dec_loss(self, encoded_region, region, secret_hat_logit, secret_gt):
        # Generator loss
        gen_loss = -torch.mean(self.discriminator(encoded_region)) * self.loss_cfg.gen_weight
        # Vis loss
        vis_loss = self.loss_func.vis_loss(encoded_region, region) * self.loss_cfg.vis_weight
        # Encode loss
        encode_loss = (gen_loss + vis_loss) * self.encode_weight

        # Decode loss
        decode_loss = self.loss_func.decode_loss(secret_hat_logit, secret_gt) * self.loss_cfg.decode_weight
        # qrcode_loss = F.l1_loss(qrcode_hat, qrcode_gt, reduction='none')
        # qrcode_loss = qrcode_loss * (torch.abs(qrcode_gt) + torch.ones_like(qrcode_gt))
        # qrcode_loss = qrcode_loss.sum()

        loss = encode_loss + decode_loss
        self.log('train/enc_dec_loss', loss)
        self.log('train/encode_weight', self.encode_weight)
        self.log('train/encode_loss', encode_loss)
        self.log('train/decode_loss', decode_loss)
        return loss

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        current_batch_idx = self.trainer.fit_loop.total_batch_idx
        host, secret, region, encoded_region, container, \
            aug_container, secret_gt, secret_hat_logit = self.shared_step(batch)

        # Calculate loss
        if optimizer_idx == 0:
            loss = self.cal_disc_loss(encoded_region, region)
        elif optimizer_idx == 1:
            loss = self.cal_enc_dec_loss(encoded_region, region, secret_hat_logit, secret_gt)
            # Update and log metrics
            with torch.no_grad():
                region_psnr = self.train_region_psnr(encoded_region, region)
                secret_par = self.train_secret_par(secret_hat_logit.sigmoid(), secret_gt.int())
            self.log('train/region_psnr', region_psnr)
            self.log('train/secret_par', secret_par)

            # Visualize image
            if current_batch_idx % 3000 == 0:
                show_image = dict(
                    region=region[0],
                    encoded_region=encoded_region[0],
                    container=container[0],
                    aug_container=aug_container[0],
                    secret_gt=secret_gt[0],
                    secret_hat=secret_hat_logit[0].sigmoid().ge(0.5).float()
                )
                self.visualize2logger('train', current_batch_idx, show_image)
        else:
            raise ValueError
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
        host, secret, region, encoded_region, container, \
            aug_container, secret_gt, secret_hat_logit = self.shared_step(batch)

        # Update and log metrics
        with torch.no_grad():
            self.valid_region_psnr.update(encoded_region, region)
            self.valid_secret_par.update(secret_hat_logit.sigmoid(), secret_gt.int())
            self.valid_region_mae.update(encoded_region, region)
            self.valid_secret_mae.update(secret_hat_logit.sigmoid().ge(0.5).float(), secret_gt)

        # Visualize image
        if batch_idx == 0:
            show_image = dict(
                region=region[0],
                encoded_region=encoded_region[0],
                container=container[0],
                aug_container=aug_container[0],
                secret_gt=secret_gt[0],
                secret_hat=secret_hat_logit[0].sigmoid().ge(0.5).float()
            )
            self.visualize2logger('valid', self.current_epoch, show_image)

        # Test decode accuracy

    def validation_epoch_end(self, outputs: List[Any]):
        region_psnr = self.valid_region_psnr.compute()
        secret_par = self.valid_secret_par.compute()
        container_mae = self.valid_region_mae.compute()
        secret_mae = self.valid_secret_mae.compute()

        self.logger_instance.add_scalar('valid/region_psnr', region_psnr, self.current_epoch)
        self.logger_instance.add_scalar('valid/secret_par', secret_par, self.current_epoch)

        # Update encode loss weight
        if secret_par > self.model_cfg.min_par:
            self.encode_weight += self.loss_cfg.encode_increment

        # Log for checkpoint
        avg_mae = (container_mae + secret_mae) / 2.
        self.log('valid/avg_mae', avg_mae)

    def test_step(self, batch: Any, batch_idx: int):
        host, secret, region, encoded_region, container, \
            aug_container, secret_gt, secret_hat_logit = self.shared_step(batch)

        # Update and log metrics
        with torch.no_grad():
            self.test_region_psnr.update(encoded_region, region)
            self.test_secret_par.update(secret_hat_logit.sigmoid(), secret_gt.int())

        msg_hats, _ = batch_qrcode_decode(secret_hat_logit.sigmoid().ge(0.5).float())
        msg_gts, _ = batch_qrcode_decode(secret_gt)
        self.test_str_acc.update(msg_hats, msg_gts)

        if batch_idx % 200 == 0:
            show_image = dict(
                region=region[0],
                encoded_region=encoded_region[0],
                container=container[0],
                aug_container=aug_container[0],
                secret_gt=secret_gt[0],
                secret_hat=secret_hat_logit[0].sigmoid().ge(0.5).float()
            )
            self.visualize2logger('test', batch_idx, show_image)

    def test_epoch_end(self, outputs: List[Any]):
        region_psnr = self.test_region_psnr.compute()
        secret_par = self.test_secret_par.compute()
        str_acc = self.test_str_acc.compute()

        self.logger_instance.add_scalar('test/region_psnr', region_psnr, self.current_epoch)
        self.logger_instance.add_scalar('test/secret_par', secret_par, self.current_epoch)
        self.logger_instance.add_scalar('test/str_acc', str_acc, self.current_epoch)

    def configure_optimizers(self):
        """
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        enc_dec_optim = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)

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
