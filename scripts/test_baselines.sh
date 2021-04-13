#!/usr/bin/env bash

METHOD=$1
OUT_DATA=$2
MODEL=BiT-S-R101x1

python test_baselines.py \
--name test_${MODEL}_${METHOD}_${OUT_DATA} \
--in_datadir dataset/id_data/ILSVRC-2012/val \
--out_datadir dataset/ood_data/${OUT_DATA} \
--model ${MODEL} \
--model_path checkpoints/finetune/finetune_flat_softmax_${MODEL}/bit.pth.tar \
--batch 256 \
--logdir checkpoints/test_log \
--score ${METHOD} \
--mahalanobis_param_path checkpoints/tune_mahalanobis/tune_mahalanobis_${MODEL}
