#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : __init__.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/8 09:39

# Import lib here
from .augmenter import Augmenter
from .random_perspective import RandomPerspective
from .illumination import Illumination
from .moire import Moire


def run():
    pass


if __name__ == '__main__':
    run()
