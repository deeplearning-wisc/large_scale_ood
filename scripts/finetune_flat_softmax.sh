#!/usr/bin/env bash

MODEL=BiT-S-R101x1

python finetune.py \
--name finetune_flat_softmax_${MODEL} \
--model ${MODEL} \
--logdir checkpoints/finetune \
--dataset imagenet2012 \
--eval_every 200 \
--datadir dataset/id_data/ILSVRC-2012 \
--train_list data_lists/imagenet2012_train_list.txt \
--val_list data_lists/imagenet2012_val_list.txt \
--num_block_open 0 \
--finetune_type flat_softmax \
