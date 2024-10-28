#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : datamodules.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/8 23:26

# Import lib here
import functools
import pdb

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from omegaconf import DictConfig
from typing import Tuple, Optional, List, Dict


class DataModules(LightningDataModule):
    def __init__(self, datasets: DictConfig,
                 dataloader_cfg: DictConfig):
        super().__init__()
        self.datasets = datasets
        self.dataloader_cfg = dataloader_cfg

        self.train_datasets: Optional[Dict] = None
        self.valid_datasets: Optional[Dict] = None
        self.test_datasets: Optional[Dict] = None

    def prepare_data(self) -> None:
        """Download data if needed."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.train_datasets:
            self.train_datasets = {}
            for name, dataset in self.datasets.items():
                self.train_datasets[name] = dataset(stage='train')

        if not self.valid_datasets:
            self.valid_datasets = {}
            for name, dataset in self.datasets.items():
                self.valid_datasets[name] = dataset(stage='valid')

        if not self.test_datasets:
            self.test_datasets = {}
            for name, dataset in self.datasets.items():
                self.test_datasets[name] = dataset(stage='test')

    def train_dataloader(self) -> Dict:
        dataloader = {}
        for name, dataset in self.train_datasets.items():
            dataloader[name] = DataLoader(dataset, shuffle=True, **self.dataloader_cfg)
        return dataloader

    def val_dataloader(self):
        dataloader = {}
        for name, dataset in self.valid_datasets.items():
            dataloader[name] = DataLoader(dataset, shuffle=False)
        return CombinedLoader(dataloader, mode='max_size_cycle')

    def test_dataloader(self):
        dataloader = {}
        for name, dataset in self.test_datasets.items():
            dataloader[name] = DataLoader(dataset, shuffle=False)
        return CombinedLoader(dataloader, mode='max_size_cycle')


def run():
    pass


if __name__ == '__main__':
    run()
