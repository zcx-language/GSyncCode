#!/usr/bin/bash

python ./src/train.py \
task_name=hidingbarcode \
tags=v0_haar_haarloss \
datamodule=host_barcode_illum_moire \
model=screenshootresilient \
model.encoder.haar_sampling=true \
model.loss_cfg.l2_yuv_weight=null \
model.loss_cfg.haar_yuv_weight=[0.1,1,1,0.05,0.5,0.5,0.05,0.5,0.5,0.05,0.5,0.5] \
# model.encoder._target_=src.models.encoders.fcn_encoder.FCNEncoder \
# model.decoder._target_=src.models.decoders.fcn_decoder.FCNDecoder \

