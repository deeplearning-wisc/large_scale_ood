#!/usr/bin/env bash

METHOD=$1
OUT_DATA=$2
MODEL=BiT-S-R101x1

python test_baselines.py \
--name test_${MODEL}_${METHOD}_${OUT_DATA} \
--in_datadir dataset/id_data/ILSVRC-2012 \
--in_data_list data_lists/imagenet2012_val_list.txt \
--out_datadir data_lists/ood_data/${OUT_DATA} \
--out_data_list data_lists/${OUT_DATA}_selected_list.txt \
--model ${MODEL} \
--model_path checkpoints/finetune/finetune_flat_softmax_${MODEL} \
--logdir checkpoints/test_log \
--score ${METHOD}