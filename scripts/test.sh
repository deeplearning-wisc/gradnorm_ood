#!/usr/bin/env bash

METHOD=$1
OUT_DATA=$2

python test_ood.py \
--name test_${METHOD}_${OUT_DATA} \
--in_datadir dataset/id_data/ILSVRC-2012/val \
--out_datadir dataset/ood_data/${OUT_DATA} \
--model_path checkpoints/pretrained_models/BiT-S-R101x1-flat-finetune.pth.tar \
--batch 256 \
--logdir checkpoints/test_log \
--score ${METHOD}