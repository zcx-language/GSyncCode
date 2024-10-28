#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : warp_perspective.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : https://github.com/kornia/kornia/blob/master/kornia/augmentation/_2d/geometric/perspective.py
# @CreateTime   : 2023/3/8 14:23

# Import lib here
import torch
from typing import Any, Dict, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample
from kornia.core import Tensor, as_tensor
from kornia.geometry.transform import get_perspective_transform, warp_perspective


class RandomPerspective(GeometricAugmentationBase2D):
    r"""Apply a random perspective transformation to an image tensor with a given probability.

    .. image:: _static/img/RandomPerspective.png

    Args:
        p: probability of the image being perspectively transformed.
        distortion_scale: the degree of distortion, ranged from 0 to 1.
        resample: the interpolation method to use.
        same_on_batch: apply the same transformation across the batch. Default: False.
        align_corners: interpolation flag.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
        sampling_method: ``'basic'`` | ``'area_preserving'``. Default: ``'basic'``
            If ``'basic'``, samples by translating the image corners randomly inwards.
            If ``'area_preserving'``, samples by randomly translating the image corners in any direction.
            Preserves area on average. See https://arxiv.org/abs/2104.03308 for further details.

        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'`` | ``'fill'``.
        fill_value: tensor of shape :math:`(3)` that fills the padding area. Only supported for RGB.
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.warp_pespective`.
    """

    def __init__(
            self,
            distortion_scale: Union[Tensor, float] = 0.5,
            resample: Union[str, int, Resample] = Resample.BILINEAR.name,
            same_on_batch: bool = False,
            align_corners: bool = False,
            p: float = 0.5,
            keepdim: bool = False,
            sampling_method: str = "basic",
            padding_mode: str = 'zeros',
            fill_value: Tensor = torch.zeros(3),    # Only supported for RGB.
            distortion_scale_bound: Optional[float] = None,     # Used for increase the distortion scale by steps.
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PerspectiveGenerator(distortion_scale, sampling_method=sampling_method)

        self.flags: Dict[str, Any] = dict(align_corners=align_corners, resample=Resample.get(resample))

        self.padding_mode = padding_mode
        self.fill_value = torch.tensor([fill_value, fill_value, fill_value], dtype=torch.float32) \
            if isinstance(fill_value, float) or isinstance(fill_value, int) else fill_value

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return get_perspective_transform(params["start_points"].to(input), params["end_points"].to(input))

    def apply_transform(
            self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        _, _, height, width = input.shape
        if not isinstance(transform, Tensor):
            raise TypeError(f'Expected the transform be a Tensor. Gotcha {type(transform)}')

        return warp_perspective(
            input, transform, (height, width), mode=flags["resample"].name.lower(),
            align_corners=flags["align_corners"], padding_mode=self.padding_mode, fill_value=self.fill_value
        )

    def inverse_transform(
            self,
            input: Tensor,
            flags: Dict[str, Any],
            transform: Optional[Tensor] = None,
            size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        return self.apply_transform(
            input,
            params=self._params,
            transform=as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )


def run():
    pass


if __name__ == '__main__':
    run()
