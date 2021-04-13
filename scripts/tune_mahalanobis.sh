#!/usr/bin/env bash

MODEL=BiT-S-R101x1

python tune_mahalanobis_hyperparameter.py \
--name tune_mahalanobis_${MODEL} \
--model ${MODEL} \
--model_path checkpoints/finetune/finetune_flat_softmax_${MODEL}/bit.pth.tar \
--logdir checkpoints/tune_mahalanobis \
--datadir dataset/id_data/ILSVRC-2012 \
--train_list data_lists/imagenet2012_train_list.txt \
--val_list data_lists/imagenet2012_val_list.txt \
