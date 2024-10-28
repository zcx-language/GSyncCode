#!/usr/bin/bash

python ./src/train.py \
task_name=hidingbarcode \
tags=default \
datamodule=host_barcode_illum_moire \
model=screenshootresilient \

