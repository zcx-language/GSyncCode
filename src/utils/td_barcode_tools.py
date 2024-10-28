#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : td_barcode_tools.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/11 11:52

# Import lib here
import cv2
import numpy as np
import qrcode
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pyzbar import pyzbar
from pylibdmtx import pylibdmtx

from typing import Union

from src.utils.image_tools import image_tensor2numpy
from src.utils.sigsev_guard import sigsev_guard
import pdb


def generate_qrcode(string: str):
    qrcode_generator = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=6,
        border=0
    )
    qrcode_generator.add_data('Accept')
    qrcode_generator.make(fit=True)
    image = qrcode_generator.make_image()
    image.show()


def batch_qrcode_decode(images: Union[torch.Tensor, np.ndarray]):
    """Quicker, but less accuracy
    Args:
        images: (B, 1, H, W) or (B, H, W)
    """
    if isinstance(images, torch.Tensor):
        images = image_tensor2numpy(images, keep_dims=True)

    assert len(images.shape) == 3, f'Error, expect inputs are gray images, but got image shaped{images.shape}'
    # (B, H, W)

    data, points = [], []
    for image in images:
        i_data = []
        i_res = pyzbar.decode(image)
        for res in i_res:
            i_data.append(res.data)
        data.append(i_data)
        points.append([])
    return data, points


def wechat_batch_qrcode_decode(images: Union[torch.Tensor, np.ndarray]):
    """Slower, but more accuracy
    Args:
        images: (B, 1, H, W) or (B, H, W)
    """
    if isinstance(images, torch.Tensor):
        images = image_tensor2numpy(images, keep_dims=True)

    assert len(images.shape) == 3, f'Error, expect inputs are gray images, but got image shaped{images.shape}'
    # (B, H, W)

    detector = cv2.wechat_qrcode_WeChatQRCode()
    values, points = [], []
    for image in images:
        i_values, i_points = detector.detectAndDecode(image)
        values.append(list(i_values))
        points.append(list(i_points))
    return values, points


@sigsev_guard(default_value=[], timeout=3)
def dmtx_decode(image: np.ndarray):
    res = pylibdmtx.decode(image)
    return res


def batch_dmtx_decode(images: Union[torch.Tensor, np.ndarray]):
    """Decode dmtx code from image
    Args:
        images: (B, 1, H, W) or (B, H, W)
    """
    if isinstance(images, torch.Tensor):
        images = image_tensor2numpy(images, keep_dims=True)

    assert len(images.shape) == 3, f'Error, expect inputs are gray images, but got image shaped{images.shape}'
    # (B, H, W)

    data, rects = [], []
    for image in images:
        i_data, i_rects = [], []
        i_res = dmtx_decode(image)
        for res in i_res:
            i_data.append(res.data)
            i_rects.append(res.rect)
        data.append(i_data)
        rects.append(i_rects)

    # i_height = height of image
    # Note that the position of (left, top) in pylibdmtx is corresponding to (left, i_height-top) in opencv,
    # as the y-axis forward up in pylibdmtx while y-axis forward down in opencv.
    # But why (left, i_height-top) is the left-bottom vertex? Who knows!
    return data, rects


def t_dmtx():
    from PIL import Image

    # Test batch
    # image_paths = [
    #     '/home/chengxin/Project/ScreenShootResilient/logs/example/test/dmtx1.png',
    #     '/home/chengxin/Project/ScreenShootResilient/logs/example/test/dmtx2.png',
    #     '/home/chengxin/Project/ScreenShootResilient/logs/example/test/dmtx3.png',
    #     '/home/chengxin/Desktop/Accept_qrcode.png',
    # ]
    # images = []
    # for image_path in image_paths:
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     image = cv2.resize(image, (256, 256))
    #     images.append(image)
    # images = np.stack(images, 0)
    # data, rects = batch_dmtx_decode(images)
    # print(data, rects)

    # Test rect
    # image_path = '/home/chengxin/Pictures/2023-05-03_19-01.png'
    image_path = '/home/chengxin/Pictures/2023-05-03_19-59.png'
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    i_height, i_width = image.shape[0:2]
    res = pylibdmtx.decode(image)
    rect = res[0].rect
    left, top, width, height = rect.left, rect.top, rect.width, rect.height
    cv2.drawMarker(image, (left, i_height-top-height), color=(0, 0, 0), thickness=4)
    cv2.drawMarker(image, (left+width, i_height-top-height), color=(0, 0, 0), thickness=3)
    cv2.drawMarker(image, (left+width, i_height-top), color=(0, 0, 0), thickness=2)
    cv2.drawMarker(image, (left, i_height-top), color=(0, 0, 0), thickness=1)
    Image.fromarray(image).show()
    print(res)


def t_qrcode():
    from kornia.augmentation import RandomPerspective
    from torchvision import transforms
    # detector = cv2.wechat_qrcode_WeChatQRCode()
    # qrcode_img_path = '/home/chengxin/Desktop/Accept_qrcode.png'
    # qrcode_img = cv2.cvtColor(cv2.imread(qrcode_img_path), cv2.COLOR_BGR2GRAY)
    # print(wechat_qrcode_decode(qrcode_img))
    # generate_qrcode('Accept')
    # qrcode_generator = qrcode.QRCode(
    #     version=1,
    #     error_correction=qrcode.constants.ERROR_CORRECT_H,
    #     box_size=6,
    #     border=1
    # )
    # qrcode_generator.clear()
    # msg = np.random.binomial(1, .5, 10)
    # qrcode_generator.add_data(''.join(map(str, msg)))
    # qrcode_generator.make(fit=True)
    # qrcode_img = qrcode_generator.make_image()
    # # qrcode_img.show()
    # random_perspective = RandomPerspective(0.5, p=1., resample=1)
    # qrcode_img = np.array(qrcode_img, np.uint8) * 255
    # print(qrcode_img.min(), qrcode_img.max(), qrcode_img.shape)
    # qrcode_img = transforms.ToTensor()(qrcode_img)
    # print(qrcode_img.min(), qrcode_img.max(), qrcode_img.shape)
    # warped_qrcode_img = random_perspective(qrcode_img.unsqueeze(0)).squeeze(0).clamp(0, 1)
    # print(warped_qrcode_img.min(), warped_qrcode_img.max(), warped_qrcode_img.shape)
    # warped_qrcode_img = transforms.ToPILImage()(warped_qrcode_img)
    # warped_qrcode_img.show()
    # qrcode_img_ary = np.expand_dims(np.array(qrcode_img, np.uint8) * 255, axis=0)
    # print(batch_qrcode_decode(qrcode_img_ary)[0])

    image_paths = [
        '/home/chengxin/Project/ScreenShootResilient/logs/example/test/qrcode1.png',
        '/home/chengxin/Project/ScreenShootResilient/logs/example/test/qrcode2.png',
        '/home/chengxin/Project/ScreenShootResilient/logs/example/test/qrcode3.png',
        '/home/chengxin/Project/ScreenShootResilient/logs/example/test/dmtx3.png',
    ]

    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        images.append(image)
    images = np.stack(images, 0)
    data, points = batch_qrcode_decode(images)
    print(data, points)
    pass


def run():
    t_dmtx()
    # t_qrcode()


if __name__ == '__main__':
    run()
