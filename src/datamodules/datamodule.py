#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : datamodule.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/8 23:26

# Import lib here
import functools
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
from typing import Tuple, Optional


class DataModule(LightningDataModule):
    def __init__(self, dataset: functools.partial,
                 dataloader_cfg: DictConfig):
        super().__init__()
        self.dataset = dataset
        self.dataloader_cfg = dataloader_cfg

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Download data if needed."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.train_data:
            self.train_data = self.dataset(stage='train')
        if not self.val_data:
            self.val_data = self.dataset(stage='val')
        if not self.test_data:
            self.test_data = self.dataset(stage='test')

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, **self.dataloader_cfg)

    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle=False, **self.dataloader_cfg)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, **self.dataloader_cfg)


def run():
    pass


if __name__ == '__main__':
    run()
