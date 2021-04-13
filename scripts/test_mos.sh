#!/usr/bin/env bash

OUT_DATA=$1
MODEL=BiT-S-R101x1

python test_mos.py \
--name test_${MODEL}_mos_${OUT_DATA} \
--in_datadir dataset/id_data/ILSVRC-2012/val \
--out_datadir dataset/ood_data/${OUT_DATA} \
--model ${MODEL} \
--model_path checkpoints/finetune/finetune_group_softmax_${MODEL}/bit.pth.tar \
--logdir checkpoints/test_log \
--group_config group_config/taxonomy_level0.npy
