#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : inference.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/14 23:08

# Import lib here
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml
import hydra
import qrcode
import shutil
import string
import random
import pyrootutils
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegFileWriter
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
from pylibdmtx import pylibdmtx
from datetime import datetime
from kornia.augmentation import RandomGaussianBlur, RandomGaussianNoise
from PIL import Image
from pathlib import Path
from torchvision import transforms
from omegaconf import DictConfig
from albumentations.augmentations.crops.functional import center_crop
from typing import Tuple, List, Optional
import pdb

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.models.augmenters import *
from src.models.augmenters.DiffJPEG.DiffJPEG import DiffJPEG
from src.models.metrics.str_bit_accuracy import StrBitAccuracy
from src.utils.image_tools import image_tensor2numpy
from src.utils.td_barcode_tools import wechat_batch_qrcode_decode, batch_dmtx_decode
from src import utils

log = utils.get_pylogger(__name__)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def iter_scan_image(image: np.ndarray,
                    scan_size: Tuple[int, int] = (512, 512),
                    resize: Tuple[int, int] = (400, 400)):
    height, width, channels = image.shape
    patch_h, patch_w = scan_size
    n_rows, n_cols = height//patch_h, width//patch_w
    re_height, re_width = n_rows * patch_h, n_cols * patch_w
    image = cv2.resize(image, (re_width, re_height))

    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            patch = image[i*patch_h:(i+1)*scan_size[0], j*patch_w:(j+1)*patch_w, :]
            if resize:
                patch = cv2.resize(patch, resize[::-1])
            patches.append(patch)
    return patches


def encode_run(cfg: DictConfig, model,
               image_dir: str,
               save_dir: str,
               msg_len: int = 12):
    log.info("start encoding...")

    tags = f"{cfg.ckpt_path.split('/')[-3]}@{cfg.ckpt_path.split('/')[-1]}"
    image_paths = sorted(path for path in Path(image_dir).glob('*.png'))
    # image_paths = ['/home/chengxin/Project/ScreenShootResilient/logs/example/host2/im95.jpg']

    msgs = []
    for _ in range(len(image_paths)):
        # ascii_list = random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=msg_len)
        # msgs.append(''.join(ascii_list))
        msgs.append('PHcuZ4udWtTE')
        # msgs.append('ChinaHust411')
        # msgs.append('AcceptChexin')

    save_dir = Path(f'{save_dir}/{tags}/{cfg.model.model_cfg.alpha}:{cfg.model.model_cfg.beta}')
    save_dir.mkdir(parents=True, exist_ok=True)
    to_tensor = transforms.ToTensor()

    qrcode_generator = qrcode.QRCode(version=1,
                                     error_correction=qrcode.constants.ERROR_CORRECT_H,
                                     box_size=128 // 21,
                                     border=0)

    for path, msg in tqdm(zip(image_paths, msgs)):
        print(f'Processing {path.name}...')
        image = Image.open(path).convert('RGB').resize((256, 256))
        image = np.array(image)

        # Generate barcode
        # qrcode_generator.clear()
        # qrcode_generator.add_data(''.join(map(str, msg)))
        # qrcode_generator.make(fit=True)
        # qrcode_img = qrcode_generator.make_image().resize((128, 128))
        encoded = pylibdmtx.encode(msg.encode('utf-8'), size='16x16')
        code_img = Image.frombytes('RGB', (encoded.width, encoded.height), encoded.pixels).convert('L')
        code_img = code_img.crop((10, 10, 90, 90))
        code_img = code_img.resize((128, 128), resample=0)  # Nearest
        code_img = np.array(code_img, dtype=np.uint8)

        # Generate Edge
        code_edge = cv2.Canny(code_img, 100, 200)
        # code_edge = cv2.GaussianBlur(code_edge, (3, 3), 0.5)
        kernel = np.ones((3, 3), dtype=np.uint8)
        code_edge = cv2.dilate(code_edge, kernel, 1)
        # code_edge = cv2.erode(code_edge, kernel, 1)

        # Encode
        batch_image = to_tensor(image).unsqueeze(dim=0)
        batch_code_img = to_tensor(code_img).unsqueeze(dim=0)
        batch_code_edge = to_tensor(code_edge).unsqueeze(dim=0)
        beg_time = time.time()
        # batch_region, top_left_pts = model.region_selector(batch_image)
        batch_region = batch_image[:, :, 126:126+128, 100:100+128]
        top_left_pts = [(126, 100)]
        # print(f'{path.name}- region selection time: {time.time()-beg_time:.5}s')
        if model.model_cfg.embed_edge:
            batch_secret = batch_code_edge
        else:
            batch_secret = batch_code_img

        beg_time = time.time()
        residual_tsr = model.encoder(batch_region.to(model.device),
                                     batch_secret.repeat(1, 3, 1, 1).to(model.device), normalize=True)
        # print(f'{path.name}- stage-1 encoding time: {time.time()-beg_time:.5}s')

        residual_tsr = residual_tsr.cpu()
        if model.model_cfg.mask_residual:
            batch_encoded_region = (batch_region * cfg.model.model_cfg.alpha +
                                    residual_tsr * batch_secret.repeat(1, 3, 1, 1) * cfg.model.model_cfg.beta).clamp(0, 1)
        else:
            batch_encoded_region = (batch_region * cfg.model.model_cfg.alpha + residual_tsr * cfg.model.model_cfg.beta).clamp(0, 1)

        # Assemble
        encoded_region = batch_encoded_region.squeeze(dim=0).cpu().numpy()
        encoded_region = (encoded_region.transpose(1, 2, 0) * 255).astype(np.uint8)
        region_height, region_width = batch_region.shape[-2:]
        container = np.copy(image)
        top_left_pt = top_left_pts[0]
        container[top_left_pt[0]:top_left_pt[0]+region_height, top_left_pt[1]:top_left_pt[1]+region_width, :] = encoded_region
        # print(f'{path.name}- all encoding time: {time.time()-beg_time:.5}s')

        # Save
        image_name = path.stem
        Image.fromarray(container).save(save_dir/f'{image_name}_{msg}_{cfg.model.model_cfg.alpha}:{cfg.model.model_cfg.beta}.png')
        res = np.abs(container - image.astype(np.float32))
        # res = (res - res.min()) / (res.max() - res.min())
        Image.fromarray(res.astype(np.uint8)).save(save_dir/f'{image_name}_{msg}_{cfg.model.model_cfg.alpha}:{cfg.model.model_cfg.beta}_res.png')

        fig = plt.figure()
        grid = ImageGrid(fig, 111, nrows_ncols=(1, 7), axes_pad=0.1)
        grid[0].imshow(image)
        grid[1].imshow(image_tensor2numpy(batch_region.squeeze(dim=0)))
        grid[2].imshow(image_tensor2numpy(batch_secret.squeeze(dim=0)), cmap='gray')
        grid[3].imshow(image_tensor2numpy(residual_tsr.squeeze(dim=0).clamp(0, 1)))
        grid[4].imshow(image_tensor2numpy((residual_tsr.squeeze(dim=0).clamp(0, 1)*batch_secret.squeeze(dim=0))))
        grid[5].imshow(encoded_region)
        grid[6].imshow(container)
        plt.savefig(save_dir/f'{image_name}_{msg}_{cfg.model.model_cfg.alpha}:{cfg.model.model_cfg.beta}_all.jpg', dpi=300)
        plt.close(fig)


def decode_run(cfg, model, image_dir: str, output_dir: str, msg_gt: str = 'ChinaHust411'):
    log.info(f'image_dir=\n{image_dir}')
    to_tensor = transforms.ToTensor()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    # screenshootpool = '/home/chengxin/Project/ScreenShootResilient/logs/example/screenshootpool'
    # screenshoot = f'/home/chengxin/Project/ScreenShootResilient/logs/example/screenshoot/{timestamp}/'
    output_dir = f'{output_dir}/{timestamp}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log.info(f'output_dir=\n{output_dir}')

    str_acc = StrBitAccuracy()
    for path in (path for path in Path(image_dir).iterdir() if path.suffix in ['.jpeg', '.jpg', '.png']):
        shutil.copy(path, Path(output_dir)/path.name)

        image = Image.open(path).convert('RGB')
        patches = [np.array(image)] * 4
        # image = np.array(Image.open(path).convert('RGB').resize((438, 438)))
        # height, width = image.shape[:2]

        # Generate batch
        # patches = iter_scan_image(np.array(image), scan_size=(1500, 1500), resize=(256, 256))
        # patches = [np.array(image.resize((256, 256))) for _ in range(4)]
        # Center crop and resize
        # patches = []
        # patches.append(cv2.resize(
        #     cv2.copyMakeBorder(
        #         image, int(width*0.05), int(width*0.05),
        #         int(height*0.01), int(height*0.01), cv2.BORDER_REFLECT_101),
        #     (256, 256)))
        # patches.append(cv2.resize(image, (256, 256)))
        # patches.append(cv2.resize(center_crop(image, int(height*0.9), int(width*0.9)), (256, 256)))
        # patches.append(cv2.resize(center_crop(image, int(height*0.8), int(width*0.8)), (256, 256)))
        # patches.append(cv2.resize(center_crop(image, int(height*0.7), int(width*0.7)), (256, 256)))
        # patches.append(cv2.resize(center_crop(image, int(height*0.6), int(width*0.6)), (256, 256)))
        # patches.append(cv2.resize(center_crop(image, int(height*0.5), int(width*0.5)), (256, 256)))
        # Rotation if needed
        # patches.append(cv2.rotate(patches[0], cv2.ROTATE_90_CLOCKWISE))
        # patches.append(cv2.rotate(patches[1], cv2.ROTATE_90_CLOCKWISE))
        # patches.append(cv2.rotate(patches[2], cv2.ROTATE_90_CLOCKWISE))
        # patches.append(cv2.rotate(patches[3], cv2.ROTATE_90_CLOCKWISE))

        patches_tsr = torch.stack([to_tensor(patch) for patch in patches], dim=0)
        # log.info(f'{path.name}- input shape: {patches_tsr.shape}')
        beg_time = time.time()
        qrcode_imgs = model.decoder(patches_tsr.to(model.device), normalize=True).sigmoid().cpu()
        # print(f'{path.name}- decoding phase-1 time: {time.time()-beg_time:.5}s')
        beg_time = time.time()
        if cfg.model.model_cfg.code_type == 'qrcode':
            batch_res, batch_pos = wechat_batch_qrcode_decode(qrcode_imgs)
        elif cfg.model.model_cfg.code_type == 'dmtx':
            batch_res, batch_pos = batch_dmtx_decode(qrcode_imgs)
        else:
            raise NotImplementedError
        # print(f'{path.name}- decoding phase-2 time: {time.time()-beg_time:.5}s')

        res = []
        for _res in batch_res:
            # A batch results for one image,
            # If there exists one decoding result, we calculate it.
            if _res:
                res = _res
                break
        log.info(f'{path.name}- decoding results: {res}')
        str_acc.update([res], [[msg_gt.encode('utf-8')]])

        n_batch = len(batch_res)
        n_col = int(n_batch**0.5)
        n_row = int(np.ceil(n_batch/n_col))
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*5))
        for idx, (qrcode_img, res, pos) in enumerate(zip(qrcode_imgs, batch_res, batch_pos)):
            axes[idx//n_col, idx%n_col].imshow(qrcode_img.squeeze(dim=0), cmap='gray')
            axes[idx//n_col, idx%n_col].set_title(res)
            # axes[idx//n_col, idx%n_col].axis('off')
        fig.tight_layout()
        # plt.show()
        plt.savefig(f'{output_dir}/{path.stem}_res.jpg')

        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*5))
        for idx, (patch, res, pos) in enumerate(zip(patches, batch_res, batch_pos)):
            axes[idx//n_col, idx%n_col].imshow(patch, cmap='gray')
            axes[idx//n_col, idx%n_col].set_title(res)
            # axes[idx//n_col, idx%n_col].axis('off')
        fig.tight_layout()
        # plt.show()
        plt.savefig(f'{output_dir}/{path.stem}_orig.jpg')
    print(f'We get BAR: {str_acc.compute()}')


def decode_video_run(model, video_path: str, save_path: str):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 25.0, (1024, 512))

    plt.ion()
    fig = plt.figure()
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (512, 512))
        # rotate the frame if needed.
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Generate batch
        f_height, f_width = frame.shape[0:2]
        crop_sizes = [f_height, int(f_height*0.9), int(f_height*0.8),
                      int(f_height*0.7), int(f_height*0.6), int(f_height*0.5), int(f_height*0.4)]
        batch_frame = []
        batch_frame.append(cv2.resize(frame, (256, 256)))
        batch_frame.append(cv2.resize(center_crop(frame, crop_sizes[1], crop_sizes[1]), (256, 256)))
        batch_frame.append(cv2.resize(center_crop(frame, crop_sizes[2], crop_sizes[2]), (256, 256)))
        batch_frame.append(cv2.resize(center_crop(frame, crop_sizes[3], crop_sizes[3]), (256, 256)))
        batch_frame.append(cv2.resize(center_crop(frame, crop_sizes[4], crop_sizes[4]), (256, 256)))
        batch_frame.append(cv2.resize(center_crop(frame, crop_sizes[5], crop_sizes[5]), (256, 256)))
        batch_frame.append(cv2.resize(center_crop(frame, crop_sizes[6], crop_sizes[6]), (256, 256)))

        # Decode
        to_tensor = transforms.ToTensor()
        inputs = torch.stack([to_tensor(frame) for frame in batch_frame], dim=0)
        outputs = model.decoder(inputs.to(model.device), normalize=True).sigmoid().cpu()
        outputs = image_tensor2numpy(outputs, keep_dims=True)
        batch_data, batch_rect = batch_dmtx_decode(outputs)

        # Draw results
        success_idx = False
        for idx, (crop_size, data, rect) in enumerate(zip(crop_sizes, batch_data, batch_rect)):
            if data:
                success_idx = idx
                data = data[0].decode('utf-8', errors='ignore')
                rect = rect[0]
                left, top, width, height = rect.left, rect.top, rect.width, rect.height
                vertices = np.array([[left, 256 - top - height],
                                     [left + width, 256 - top - height],
                                     [left + width, 256 - top],
                                     [left, 256 - top]], float)

                vertices_before_crop = vertices * (crop_size / 256)
                # vertices_512 = vertices_before_crop
                vertices_512 = vertices_before_crop + (512 - crop_size) // 2

                # cv2.drawMarker(frame, np.int32(vertices_512), color=(255, 0, 0))
                cv2.polylines(frame, np.int32([vertices_512]), thickness=5, color=(0, 255, 0), isClosed=True)
                cv2.putText(frame, data, tuple((vertices_512[0, :]+np.array([0, -15])).astype(np.int32)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                break

        # Show result
        # ax = fig.add_subplot(1, 2, 1)
        # ax.axis('off')
        # ax.imshow(frame)
        # ax.plot()
        # ax = fig.add_subplot(1, 2, 2)
        # ax.axis('off')
        # ax.imshow(outputs[success_idx if success_idx else 3], cmap='gray')
        # ax.plot()
        # plt.pause(0.05)
        # fig.clf()

        # Assemble video
        # pdb.set_trace()
        barcode = outputs[success_idx if success_idx else 3]
        barcode_512 = cv2.copyMakeBorder(barcode, 128, 128, 128, 128, cv2.BORDER_CONSTANT, value=255)
        barcode_512 = np.stack([barcode_512]*3, axis=-1)
        frame = np.concatenate([frame, barcode_512], axis=1)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0.5)

    cap.release()
    out.release()


def simulate_screen_shoot(image_dir: str, output_dir: str, image_size: Tuple[int, int]):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rnd_perspective = RandomPerspective(distortion_scale=0.3, resample='nearest', p=1.)
    illumination = Illumination(p=1.)
    moire = Moire(weight_bound=0.1, p=1.)
    blur = RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.8, 0.8), p=1.)
    noise = RandomGaussianNoise(mean=0., std=0.01, p=1.)
    jpeg = DiffJPEG(height=image_size[0], width=image_size[1], quality=80, p=1.)
    to_tensor = transforms.ToTensor()

    batch_image = []
    image_paths = sorted(str(path) for path in Path(image_dir).glob('*.png'))
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        # image = np.array(image)[:, 128:256, :]      # For PIMoG, as its container are stack of multiple image.
        image = to_tensor(image)
        batch_image.append(image)

    batch_image = torch.stack(batch_image, dim=0)
    batch_image = rnd_perspective(batch_image)

    # batch_image_ary = image_tensor2numpy(batch_image)
    # for idx, image in enumerate(batch_image_ary):
    #     image_name = image_paths[idx].split('/')[-1]
    #     Image.fromarray(image).save(f'{output_dir}/perspective_{image_name}')

    batch_image = illumination(batch_image)
    batch_image = moire(batch_image)
    batch_image = blur(batch_image)
    batch_image = noise(batch_image)
    batch_image = jpeg(batch_image)

    batch_image_ary = image_tensor2numpy(batch_image)
    for idx, image in enumerate(batch_image_ary):
        image_name = image_paths[idx].split('/')[-1]
        Image.fromarray(image).save(f'{output_dir}/simulated_{image_name}')


@hydra.main(version_base='1.3', config_path='../configs', config_name='inference.yaml')
def inference(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.ckpt_path)['state_dict'], strict=False)
    model.freeze()
    model = model.to(device)

    if cfg.task == 'encode':
        # Encoder process
        encode_run(cfg, model, cfg.image_dir, cfg.save_dir)
    elif cfg.task == 'simulate':
        simulate_screen_shoot(
            image_dir=cfg.image_dir,
            output_dir=cfg.output_dir,
            image_size=(400, 400)
        )
    elif cfg.task == 'decode':
        decode_run(cfg, model, cfg.image_dir, cfg.output_dir)
    elif cfg.task == 'decode_video':
        decode_video_run(model, cfg.video_path, cfg.save_path)
    else:
        raise ValueError
    pass


if __name__ == '__main__':
    inference()
