#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : ScreenShootResilient
# @File         : image_tools.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2021/3/26 下午3:02

# Import lib here
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import matplotlib as mpl
from pathlib import Path

from torchvision import transforms
from matplotlib import pyplot as plt
from typing import Union, Optional, List, Tuple


to_tensor = transforms.ToTensor()
img_norm = transforms.Normalize(0.5, 0.5)
img_denorm = transforms.Normalize(-1, 2)


def image_tensor2numpy(tensor: torch.Tensor, normalize: bool = False, value_range: Optional[Tuple[int, int]] = None,
                       scale_each: bool = True, keep_dims: bool = False) -> np.ndarray:
    """Converts a PyTorch tensor to a numpy image, it always squeezes the channel dimension.
    In case the tensor is in the GPU, it will be copied back to CPU.

    Args:
        tensor (torch.Tensor): image of the form :math:`(H, W)`, :math:`(C, H, W)` or :math:`(B, C, H, W)`.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool): If ``True``, scale each image in the batch of
            images separately rather than over all images. Default: ``True``.
        keep_dims (bool): if ``False``, squeeze the input tensor in batch dimension.

    Returns:
        numpy.ndarray: image of the form :math:`(H, W)`, :math:`(H, W, C)` or :math:`(B, H, W, C)`.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError("Input size must be a two, three or four dimensional tensor")

    input_shape = tensor.shape
    image: np.ndarray = tensor.detach().cpu().numpy()

    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        pass
    elif len(input_shape) == 3:
        # (C, H, W) -> (H, W, C)
        if input_shape[0] == 1:
            image = image.squeeze()
        else:
            image = image.transpose((1, 2, 0))
    elif len(input_shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        image = image.transpose((0, 2, 3, 1))
        if input_shape[0] == 1 and not keep_dims:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)

    else:
        raise ValueError("Cannot process tensor with shape {}".format(input_shape))

    if normalize is True:
        image = image.copy()
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img: np.ndarray, low: float, high: float):
            np.clip(img, a_min=low, a_max=high, out=img)
            img.__isub__(low).__itruediv__(max(high - low, 1e-5))

        def norm_range(t: np.ndarray, value_range: Tuple):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if len(input_shape) == 4 and scale_each is True:
            for t in image:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(image, value_range)

    assert image.min() >= 0. and image.max() <= 1., \
        f'Error, expect data in range[0, 1], but got min={image.min()}, max={image.max()}'
    image: np.ndarray = np.clip(image * 255 + 0.5, 0, 255).astype(np.uint8)
    return image


def images_save(images: Union[torch.Tensor, np.ndarray], folder: str, gray: bool = False) -> None:
    """Save torch.Tensor or np.ndarray format images

    Args:
        images (Union[torch.Tensor, np.ndarray]):
            if image is torch.Tensor, need :math:`(H, W)`, :math:`(C, H, W)` or :math:`(B, C, H, W)`.
            if image is np.ndarray, need :math:`(H, W)`, :math:`(H, W, C)` or :math:`(B, H, W, C)`.

        folder (str): the folder that images to save
        gray (bool): indicate the images are gray or not
    """

    if isinstance(images, torch.Tensor):
        images = image_tensor2numpy(images)
    if isinstance(images, np.ndarray):
        images = images.squeeze()
    # here images shape may be (B, H, W, C) or (H, W, C) if gray is False
    # else images shape may be (B, H, W) or (H, W)

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    images_shape = images.shape
    if gray is False:
        if len(images_shape) == 4:
            bits = int(math.log10(images_shape[0])) + 1
            for idx, image in enumerate(images):
                image_path = folder / f'{idx:0{bits}d}.jpg'
                plt.imsave(image_path, image)
        elif len(images_shape) == 3:
            image = images
            image_path = folder / '0.jpg'
            plt.imsave(image_path, image)
        else:
            raise ValueError("Cannot process images with shape {}".format(images_shape))

    else:
        if len(images_shape) == 3:
            bits = int(math.log10(images_shape[0])) + 1
            for idx, image in enumerate(images):
                image_path = folder / f'{idx:0{bits}d}.jpg'
                plt.imsave(image_path, image, cmap='gray')
        elif len(images_shape) == 2:
            image = images
            image_path = folder / '0.jpg'
            plt.imsave(image_path, image, cmap='gray')
        else:
            raise ValueError("Cannot process images with shape {}".format(images_shape))
    return


def image_show(image: Union[torch.Tensor, np.ndarray], figsize: Tuple = (0, 0)) -> None:
    """Show image of torch.Tensor or numpy.array format

    Args:
        image (Union[torch.Tensor, np.ndarray]): the image to show
        figsize (Tuple): the figure size to show

    Returns:
        None
    """

    if isinstance(image, torch.Tensor):
        image = image_tensor2numpy(image)

    # What size does the figure need to be in inches to fit the image?
    if figsize == (0, 0):
        dpi = mpl.rcParams['figure.dpi']
        height, width = image.shape[:2]
        figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(image, cmap='gray')
    plt.show()


def image_color2gray(image: np.ndarray, color: str = 'rgb') -> np.ndarray:
    """ OBSOLETED!!!. Convert image to gray schema

    Args:
        image (np.ndarray):
        color (str):

    Returns:
        np.ndarray
    """

    color = color.lower()
    assert color in {'rgb', 'bgr'}, f'Error, not support color: {color}'
    ndim = image.ndim
    if ndim == 3:
        if color == 'rgb':
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif ndim == 2:
        gray = image.copy()
    else:
        raise ValueError(f'Error, not support image with dimension: {ndim}')

    return gray


def image_split(image: np.ndarray, row: int, col: int, resize=True) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """

    Args:
        image ():
        row ():
        col ():
        resize ():

    Returns:

    """

    height, width = image.shape[:2]
    if resize:
        patch_height, patch_width = height // row, width // col
        height -= height % patch_height
        width -= width % patch_width
        image = cv2.resize(image, (width, height))
    else:
        assert height % row == 0 and width % col == 0
        patch_height, patch_width = height // row, width // col

    patches = []
    offsets = []
    for i in range(row):
        for j in range(col):
            patch = image[i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width, ...]
            patches.append(patch)
            offsets.append((i*patch_height, j*patch_width))
    return patches, offsets


def image_assemble(patches: List[np.ndarray], row: int, col: int) -> np.ndarray:
    """

    Args:
        patches ():
        row ():
        col():

    Returns:

    """

    image = cv2.vconcat([cv2.hconcat([patches[i*col+j] for j in range(col)]) for i in range(row)])
    return image


def image_compression_by_cv2(image: torch.Tensor, quality: int) -> torch.Tensor:
    """Compress images by opencv (Obsoleted)

    Args:
        image (torch.Tensor): N, C, H, W
        quality (int):
    """
    assert image.ndim in {3, 4}, f'Error, need shape of 3(3, H, W) or 4(N, 3, H, W) but get {image.ndim}'
    ary_image = image_tensor2numpy(image)
    if ary_image.ndim == 3:
        _, encoded_img = cv2.imencode('.jpg', ary_image, (int(cv2.IMWRITE_JPEG_QUALITY), quality))
        ary_image = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
        tsr_image = torch.tensor(ary_image, dtype=torch.float32, device=image.device) / 255.
    elif ary_image.ndim == 4:
        for i in range(ary_image.shape[0]):
            _, encoded_img = cv2.imencode('.jpg', ary_image[i], (int(cv2.IMWRITE_JPEG_QUALITY), quality))
            ary_image[i] = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
        tsr_image = torch.tensor(ary_image.transpose((0, 3, 1, 2)), dtype=torch.float32, device=image.device) / 255.
    else:
        raise NotImplementedError
    return tsr_image


class ImageArnold:
    def __init__(self, shuffle_nums: int,
                 a: int = 3,
                 b: int = 3):
        super().__init__()
        self.shuffle_nums = shuffle_nums
        self.a = a
        self.b = b

    def arnold(self, inputs: Union[torch.Tensor, np.ndarray]):
        if isinstance(inputs, torch.Tensor):
            height, width = inputs.shape[-2:]
            outputs = torch.empty_like(inputs)
        elif isinstance(inputs, np.ndarray):
            # TODO
            raise NotImplementedError
        else:
            raise ValueError(f'Error, Need inputs typed torch.Tensor or np.ndarray, but got {type(inputs)}')

        for _ in range(self.shuffle_nums):
            for i in range(height):
                for j in range(width):
                    x = (i + self.b * j) % height
                    y = (self.a * i + (self.a * self.b + 1) * j) % width
                    outputs[..., x, y] = inputs[..., i, j]
        return outputs

    def de_arnold(self, inputs: Union[torch.Tensor, np.ndarray]):
        if isinstance(inputs, torch.Tensor):
            height, width = inputs.shape[-2:]
            outputs = torch.empty_like(inputs)
        elif isinstance(inputs, np.ndarray):
            # TODO
            raise NotImplementedError
        else:
            raise ValueError(f'Error, Need inputs typed torch.Tensor or np.ndarray, but got {type(inputs)}')

        for _ in range(self.shuffle_nums):
            for i in range(height):
                for j in range(width):
                    x = ((self.a * self.b + 1) * i - self.b * j) % height
                    y = (-self.a * i + j) % width
                    outputs[..., x, y] = inputs[..., i, j]
        return outputs


# Calculate the PSNR of two images
def image_psnr(image1: np.ndarray,
               image2: np.ndarray,
               data_range: float = 255.):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(data_range / math.sqrt(mse))


def run():
    from kornia.geometry.transform import translate
    image_path = '/home/chengxin/Desktop/Accept_qrcode.png'
    image_ary = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    image_ary = cv2.resize(image_ary, (128, 128))
    image_tsr = transforms.ToTensor()(image_ary)
    # image_show(image_tsr)
    # image_tsr = image_compression_by_cv2(image_tsr.unsqueeze(0), quality=10)
    # image_show(image_tsr[0])
    image_show(image_tsr)
    image_show(translate(image_tsr.unsqueeze(0), torch.tensor([[10., 10.]]))[0])

    image_arnold = ImageArnold(4)
    permuted_image = image_arnold.arnold(image_tsr.unsqueeze(dim=0))
    image_show(permuted_image[0])
    # permuted_image = permuted_image + torch.randn_like(permuted_image) * 0.2
    permuted_image = translate(permuted_image, torch.tensor([[10., 10.]]))
    permuted_image = permuted_image.clamp(0, 1)
    depermuted_image = image_arnold.de_arnold(permuted_image)
    image_show(depermuted_image[0])

    # image1 = cv2.imread('/home/chengxin/Desktop/lena.png')
    # image1 = cv2.resize(image1, (256, 256))
    # image2 = cv2.imread('/home/chengxin/OpenSource/InvisibleMarkers/data/test/h3.png')
    # print(image_psnr(image1, image2))
    pass


if __name__ == '__main__':
    run()
