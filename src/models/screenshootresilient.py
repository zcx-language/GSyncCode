#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ScreenShootResilient
# @File         : screenshootresilient.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/8 16:10
#
# Import lib here
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
import pytorch_lightning as pl
from torchvision.transforms import Normalize
from torchmetrics import Accuracy, PeakSignalNoiseRatio, MeanAbsoluteError, StructuralSimilarityIndexMeasure
from torchmetrics.functional import accuracy, peak_signal_noise_ratio
from src.models.metrics import StrBitAccuracy, MaskPAR
from src.models.loss_funcs.weighted_yuv_loss import weighted_yuv_loss
from src.models.loss_funcs.haar_loss import yuv_haar_loss

from omegaconf import DictConfig
from typing import Any, List, Dict, Optional

from src.utils.td_barcode_tools import batch_qrcode_decode, batch_dmtx_decode
from src.utils.gan_tools import ImagePool, cal_gradient_penalty
from lpips.lpips import LPIPS
from src.utils import get_pylogger

LPIPS_LOSS = None
log = get_pylogger(__name__)


class ScreenShootResilient(pl.LightningModule):
    def __init__(self, region_selector: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 discriminator: nn.Module,
                 augmenter: nn.Module,
                 loss_cfg: DictConfig,
                 model_cfg: DictConfig):
        super().__init__()
        torch.set_float32_matmul_precision('high')
        self.automatic_optimization = False

        self.region_selector = region_selector
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.augmenter = augmenter
        self.loss_cfg = loss_cfg
        self.model_cfg = model_cfg

        self.valid_region_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.valid_secret_par = Accuracy(task='binary')
        self.valid_patch_par = MaskPAR()
        self.valid_str_acc = StrBitAccuracy()

        # Used for choosing the best ckpt that can balance the metric between container and secret.
        self.valid_region_mae = MeanAbsoluteError()
        self.valid_secret_mae = MeanAbsoluteError()

        self.test_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=1.)
        self.test_region_psnr = PeakSignalNoiseRatio(data_range=1.)
        self.test_region_ssim = StructuralSimilarityIndexMeasure(data_range=1.)
        self.test_secret_par = Accuracy(task='binary')
        self.test_patch_par = MaskPAR()
        self.test_str_acc = StrBitAccuracy()

        # self.augmenter.distortion_types
        # Metric in individual distortions.
        self.test_patch_par_dict = nn.ModuleDict({
            key: MaskPAR() for key in self.augmenter.distortion_types
        })
        self.test_str_acc_dict = nn.ModuleDict({
            key: StrBitAccuracy() for key in self.augmenter.distortion_types
        })

        # Log model arch to file
        from torchinfo import summary
        log.info(f'Encoder Summary:\n'
                 f'{summary(self.encoder, input_size=((1, 3, 128, 128), (1, 3, 128, 128)), depth=5, verbose=0)}')
        log.info(f'Decoder Summary:\n'
                 f'{summary(self.decoder, input_size=(1, 3, 256, 256), depth=5, verbose=0)}')

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
            container[batch_idx, :, h_idx:h_idx + region_height, w_idx:w_idx + region_width] = encoded_region[batch_idx]

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
        results = self.decoder(image).squeeze(dim=0).sigmoid().ge(0.5).float()
        return results.to(orig_device)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.valid_region_psnr.reset()
        self.valid_secret_par.reset()
        self.valid_patch_par.reset()
        self.valid_region_mae.reset()
        self.valid_secret_mae.reset()
        self.valid_str_acc.reset()

        global LPIPS_LOSS
        LPIPS_LOSS = LPIPS(net='vgg').to(self.device)

    def shared_step(self, batch: Any, mode: str):
        assert mode in {'train', 'valid', 'test'}
        host, secret_byte, secret, illumination_pattern, moire_pattern = batch

        region, top_left_pt = self.region_selector(host)

        if self.model_cfg.embed_edge:
            raise NotImplementedError
        else:
            emb = secret.repeat(1, 3, 1, 1)

        residual = self.encoder(region, emb, normalize=True)
        if self.model_cfg.mask_residual:
            residual = residual * emb

        encoded_region = (region * self.model_cfg.alpha + residual * self.model_cfg.beta).clamp(0, 1)
        container = torch.clone(host)
        secret_gt = torch.ones((host.shape[0], 1, host.shape[2], host.shape[3]),
                               dtype=torch.float32, device=container.device)
        patch_mask = torch.zeros_like(secret_gt)

        region_height, region_width = region.shape[-2:]
        for batch_idx in range(host.shape[0]):
            h_idx = top_left_pt[batch_idx, 0]
            w_idx = top_left_pt[batch_idx, 1]
            container[batch_idx, :, h_idx:h_idx + region_height, w_idx:w_idx + region_width] = encoded_region[batch_idx]
            secret_gt[batch_idx, :, h_idx:h_idx + region_height, w_idx:w_idx + region_width] = secret[batch_idx]
            patch_mask[batch_idx, :, h_idx:h_idx + region_height, w_idx:w_idx + region_width] = 1.

        if mode == 'train':
            current_batch_idx = self.trainer.fit_loop.total_batch_idx
        else:
            current_batch_idx = 999999999
        aug_container_dict = self.augmenter(container,
                                            current_batch_idx,
                                            {'illumination': illumination_pattern, 'moire': moire_pattern},
                                            return_individual=mode == 'test')

        o_secret_gt = secret_gt.ge(0.5).float()
        o_patch_mask = patch_mask.ge(0.5)
        if self.model_cfg.geometric_sync:
            secret_gt = self.augmenter.perspective(secret_gt, params=self.augmenter.perspective._params)
            patch_mask = self.augmenter.perspective(patch_mask, params=self.augmenter.perspective._params)
            # Perspective's padding model is fill with 1, it automatically converts gray to rgb,
            # so we need to convert it back to gray.
            # if secret_gt.shape[1] == 3:
            #     secret_gt = secret_gt.mean(dim=1, keepdim=True)
            # if patch_mask.shape[1] == 3:
            #     patch_mask = patch_mask.mean(dim=1, keepdim=True)
        secret_gt = secret_gt.ge(0.5).float()
        patch_mask = patch_mask.ge(0.5)

        secret_hat_logit_dict = {}
        for aug_name, aug_container in aug_container_dict.items():
            beg_time = time.time()
            secret_hat_logit_dict[aug_name] = self.decoder(aug_container, normalize=True)
            print(f'Time: {time.time() - beg_time}s')
        return (host, secret, secret_byte, region, encoded_region, residual, container, aug_container_dict, secret_gt,
                secret_hat_logit_dict, patch_mask, o_secret_gt, o_patch_mask)

    def training_step(self, batch: Any, batch_idx: int):
        (host, secret, secret_byte, region, encoded_region, residual, container, aug_container_dict, secret_gt,
         secret_hat_logit_dict, patch_mask, _, _) = self.shared_step(batch, 'train')
        aug_container = aug_container_dict['combine']
        secret_hat_logit = secret_hat_logit_dict['combine']
        secret_hat = secret_hat_logit.sigmoid()

        # Calculate loss
        enc_dec_optim, disc_optim = self.optimizers()
        # GAN loss
        if self.loss_cfg.gan_weight:
            # Update discriminator
            self.set_requires_grad(self.discriminator, True)
            fake_pred = self.discriminator(encoded_region.detach())
            real_pred = self.discriminator(region)
            dis_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred)) + \
                       F.mse_loss(real_pred, torch.ones_like(real_pred))
            self.log('train/dis_loss', dis_loss.item())
            disc_optim.zero_grad()
            self.manual_backward(dis_loss)
            disc_optim.step()

            self.set_requires_grad(self.discriminator, False)
            fake_pred = self.discriminator(encoded_region)
            gan_loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred)) * self.loss_cfg.gan_weight
            self.log('train/gan_loss', gan_loss.item())
        else:
            gan_loss = torch.tensor(0., device=self.device)

        # LPIPS loss
        if self.loss_cfg.lpips_weight:
            lpips_loss = LPIPS_LOSS(encoded_region, region).mean() * self.loss_cfg.lpips_weight
            self.log('train/lpips_loss', lpips_loss.item())
        else:
            lpips_loss = torch.tensor(0., device=self.device)

        # L2 loss
        if self.loss_cfg.l2_yuv_weight:
            l2_loss = weighted_yuv_loss(encoded_region, region, yuv_weights=self.loss_cfg.l2_yuv_weight)
            self.log('train/l2_loss', l2_loss.item())
        else:
            l2_loss = torch.tensor(0., device=self.device)

        # Haar loss
        if self.loss_cfg.haar_yuv_weight:
            haar_loss = yuv_haar_loss(encoded_region, region, freq_channel_weights=self.loss_cfg.haar_yuv_weight)
            self.log('train/haar_loss', haar_loss.item())
        else:
            haar_loss = torch.tensor(0., device=self.device)

        # Decode loss
        attention = torch.ones_like(patch_mask) + patch_mask
        decode_loss = F.binary_cross_entropy_with_logits(secret_hat_logit, secret_gt,
                                                         attention) * self.loss_cfg.decode_weight
        self.log('train/decode_loss', decode_loss.item())

        loss = gan_loss + lpips_loss + l2_loss + decode_loss + haar_loss
        self.log('train/loss', loss.item())
        enc_dec_optim.zero_grad()
        self.manual_backward(loss)
        enc_dec_optim.step()

        # Log metrics
        with torch.no_grad():
            region_psnr = peak_signal_noise_ratio(encoded_region, region, data_range=1.).item()
            secret_par = accuracy(secret_hat, secret_gt.int(), task='binary').item()
        self.log('train/region_psnr', region_psnr, prog_bar=True)
        self.log('train/secret_par', secret_par, prog_bar=True)

        # Visualize image
        # if self.total_steps % 3000 == 0:
        #     show_image = dict(
        #         region=region[0],
        #         encoded_region=encoded_region[0],
        #         redisual=residual[0].clamp(0, 1),
        #         container=container[0],
        #         aug_container=aug_container[0],
        #         secret_gt=secret_gt[0],
        #         secret_hat=secret_hat[0]
        #     )
        #     self.logger_instance.add_image('train/region', self.image_denorm(region[0]), self.current_epoch)

    def validation_step(self, batch: Any, batch_idx: int):
        (host, secret, secret_byte, region, encoded_region, residual, container, aug_container_dict, secret_gt,
         secret_hat_logit_dict, patch_mask, _, _) = self.shared_step(batch, 'valid')
        aug_container = aug_container_dict['combine']
        secret_hat_logit = secret_hat_logit_dict['combine']
        secret_hat = secret_hat_logit.sigmoid()

        # Update and log metrics
        with torch.no_grad():
            self.valid_region_psnr.update(encoded_region, region)
            self.valid_secret_par.update(secret_hat, secret_gt.int())
            self.valid_patch_par.update(secret_hat.ge(0.5), secret_gt.ge(0.5), patch_mask)
            self.valid_region_mae.update(encoded_region, region)
            self.valid_secret_mae.update(secret_hat, secret_gt)

            if self.model_cfg.code_type == 'qrcode':
                msg_hats, _ = batch_qrcode_decode(secret_hat)
            elif self.model_cfg.code_type == 'dmtx':
                msg_hats, _ = batch_dmtx_decode(secret_hat)
            else:
                raise ValueError
            secret_bytes = secret_byte.cpu()
            msg_gts = [[''.join([chr(byte) for byte in secret_byte]).encode('utf-8')] for secret_byte in secret_bytes]
            self.valid_str_acc.update(msg_hats, msg_gts)

        # Visualize image
        if batch_idx == 0:
            for idx in range(2):
                show_image = torch.cat([
                    F.pad(region[idx], (64, 64, 64, 64), value=0.),
                    F.pad(encoded_region[idx], (64, 64, 64, 64), value=0.),
                    F.pad(residual[idx].clamp(0, 1), (64, 64, 64, 64), value=0.),
                    container[idx],
                    aug_container[idx],
                    secret_gt[idx].repeat(3, 1, 1),
                    secret_hat[idx].repeat(3, 1, 1),
                ], dim=-1)
                self.logger_instance.add_image(f'valid/example{idx}', show_image, self.current_epoch)

    def on_validation_epoch_end(self):
        region_psnr = self.valid_region_psnr.compute().item()
        secret_par = self.valid_secret_par.compute().item()
        patch_par = self.valid_patch_par.compute()
        container_mae = self.valid_region_mae.compute().item()
        secret_mae = self.valid_secret_mae.compute().item()
        str_acc = self.valid_str_acc.compute()

        self.valid_region_psnr.reset()
        self.valid_secret_par.reset()
        self.valid_patch_par.reset()
        self.valid_region_mae.reset()
        self.valid_secret_mae.reset()
        self.valid_str_acc.reset()

        self.logger_instance.add_scalar('valid/region_psnr', region_psnr, self.current_epoch)
        self.logger_instance.add_scalar('valid/secret_par', secret_par, self.current_epoch)
        self.logger_instance.add_scalar('valid/patch_par', patch_par, self.current_epoch)
        self.logger_instance.add_scalar('valid/str_acc', str_acc, self.current_epoch)

        # Log to file
        log.info(f'Epoch {self.current_epoch} '
                 f'valid/region_psnr: {region_psnr:.5f},'
                 f'valid/secret_par: {secret_par:.5f},'
                 f'valid/patch_par: {patch_par:.5f},'
                 f'valid/str_acc: {str_acc:.5f}')

        # Log for checkpoint
        avg_mae = (container_mae + secret_mae) / 2.
        self.log('valid/avg_mae', avg_mae)

    def on_test_start(self):
        # Let the test results be deterministic as the augmentation is random
        pl.seed_everything(42, workers=True)

    def test_step(self, batch: Any, batch_idx: int):
        (host, secret, secret_byte, region, encoded_region, residual, container, aug_container_dict, secret_gt,
         secret_hat_logit_dict, patch_mask, o_secret_gt, o_patch_mask) = self.shared_step(batch, 'test')
        aug_container = aug_container_dict['combine']
        secret_hat_logit = secret_hat_logit_dict['combine']
        secret_hat = secret_hat_logit.sigmoid()

        # Update and log metrics
        with torch.no_grad():
            self.test_psnr.update(container, host)
            self.test_ssim.update(container, host)
            self.test_region_psnr.update(encoded_region, region)
            self.test_region_ssim.update(encoded_region, region)
            self.test_secret_par.update(secret_hat, secret_gt.int())
            self.test_patch_par.update(secret_hat.ge(0.5), secret_gt.ge(0.5), patch_mask)

            if self.model_cfg.code_type == 'qrcode':
                msg_hats, _ = batch_qrcode_decode(secret_hat)
            elif self.model_cfg.code_type == 'dmtx':
                msg_hats, _ = batch_dmtx_decode(secret_hat)
            else:
                raise ValueError
            secret_bytes = secret_byte.cpu()
            msg_gts = [[''.join([chr(byte) for byte in secret_byte]).encode('utf-8')] for secret_byte in secret_bytes]
            self.test_str_acc.update(msg_hats, msg_gts)

        # Visualize image
        show_image = torch.cat([
            F.pad(region[0], (64, 64, 64, 64), value=0.),
            F.pad(encoded_region[0], (64, 64, 64, 64), value=0.),
            F.pad(residual[0].clamp(0, 1), (64, 64, 64, 64), value=0.),
            container[0],
            aug_container[0],
            secret_gt[0].repeat(3, 1, 1),
            secret_hat[0].repeat(3, 1, 1),
        ], dim=-1)
        self.logger_instance.add_image(f'test/example', show_image, batch_idx)

        # Metric of individual type of distortion
        with torch.no_grad():
            for key in self.augmenter.distortion_types:
                secret_hat = secret_hat_logit_dict[key].sigmoid()
                if key == 'perspective':
                    self.test_patch_par_dict[key].update(secret_hat.ge(0.5), secret_gt.ge(0.5), patch_mask)
                else:
                    self.test_patch_par_dict[key].update(secret_hat.ge(0.5), o_secret_gt.ge(0.5), o_patch_mask)

                if self.model_cfg.code_type == 'qrcode':
                    msg_hats, _ = batch_qrcode_decode(secret_hat)
                elif self.model_cfg.code_type == 'dmtx':
                    msg_hats, _ = batch_dmtx_decode(secret_hat)
                else:
                    raise ValueError
                self.test_str_acc_dict[key].update(msg_hats, msg_gts)

    def on_test_epoch_end(self):
        psnr = self.test_psnr.compute().item()
        ssim = self.test_ssim.compute().item()
        region_psnr = self.test_region_psnr.compute().item()
        region_ssim = self.test_region_ssim.compute().item()
        secret_par = self.test_secret_par.compute().item()
        patch_par = self.test_patch_par.compute()
        str_acc = self.test_str_acc.compute()

        self.test_psnr.reset()
        self.test_ssim.reset()
        self.test_region_psnr.reset()
        self.test_region_ssim.reset()
        self.test_secret_par.reset()
        self.test_patch_par.reset()
        self.test_str_acc.reset()

        self.logger_instance.add_scalar('test/psnr', psnr, self.current_epoch)
        self.logger_instance.add_scalar('test/ssim', ssim, self.current_epoch)
        self.logger_instance.add_scalar('test/region_psnr', region_psnr, self.current_epoch)
        self.logger_instance.add_scalar('test/region_ssim', region_ssim, self.current_epoch)
        self.logger_instance.add_scalar('test/secret_par', secret_par, self.current_epoch)
        self.logger_instance.add_scalar('test/patch_par', patch_par, self.current_epoch)
        self.logger_instance.add_scalar('test/str_acc', str_acc, self.current_epoch)

        # Log to file
        log.info(f'test/psnr: {psnr:.5f}')
        log.info(f'test/ssim: {ssim:.5f}')
        log.info(f'test/region_psnr: {region_psnr:.5f}')
        log.info(f'test/region_ssim: {region_ssim:.5f}')
        log.info(f'test/secret_par: {secret_par:.5f}')
        log.info(f'test/patch_par: {patch_par:.5f}')
        log.info(f'test/str_acc: {str_acc:.5f}')

        # Metric of individual type of distortion
        for key in self.augmenter.distortion_types:
            patch_par = self.test_patch_par_dict[key].compute()
            str_acc = self.test_str_acc_dict[key].compute()
            self.logger_instance.add_scalar(f'test/patch_par_{key}', patch_par, self.current_epoch)
            self.logger_instance.add_scalar(f'test/str_acc_{key}', str_acc, self.current_epoch)
            log.info(f'Under {key}: test/patch_par: {patch_par:.5f}, test/str_acc: {str_acc:.5f}')

    def configure_optimizers(self):
        enc_dec_optim = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)

        disc_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
        return enc_dec_optim, disc_optim

    @property
    def total_steps(self):
        return self.trainer.fit_loop.total_batch_idx

    @property
    def logger_instance(self):
        return self.logger.experiment

    @staticmethod
    def image_denorm(image: torch.Tensor, mode: str = 'default'):
        if mode == 'default':
            return (image + 1.) / 2.
        elif mode == 'min_max':
            return (image - image.min()) / (image.max() - image.min())
        else:
            raise ValueError

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, List):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def run():
    pass


if __name__ == '__main__':
    run()
