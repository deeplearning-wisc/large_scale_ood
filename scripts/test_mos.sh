#!/usr/bin/env bash

OUT_DATA=$1
MODEL=BiT-S-R101x1

python test_mos.py \
--name test_${MODEL}_mos_${OUT_DATA} \
--in_datadir dataset/id_data/ILSVRC-2012 \
--in_data_list data_lists/imagenet2012_val_list.txt \
--out_datadir data_lists/ood_data/${OUT_DATA} \
--out_data_list data_lists/${OUT_DATA}_selected_list.txt \
--model ${MODEL} \
--model_path checkpoints/finetune/finetune_group_softmax_${MODEL} \
--logdir checkpoints/test_log \
--group_config group_config/taxonomy_level0.npy