# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# coding: utf-8

import argparse


def argparser():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained model.")
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring and checkpointing.")

    parser.add_argument("--model", help="Which variant to use.", default="BiT-S-R101x1")
    parser.add_argument("--logdir", required=True,
                        help="Where to log training info and checkpoints.")

    parser.add_argument("--bit_pretrained_dir", default='bit_pretrained_models',
                        help="Where to search for pretrained BiT models.")

    parser.add_argument("--dataset", default="imagenet2012")

    parser.add_argument("--batch", type=int, default=512,
                        help="Batch size.")
    parser.add_argument("--batch_split", type=int, default=1,
                        help="Number of batches to compute gradient on before updating weights.")
    parser.add_argument("--base_lr", type=float, default=0.003,
                        help="Base learning-rate for fine-tuning. Most likely default is best.")
    parser.add_argument("--eval_every", type=int, default=None,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--datadir", required=True,
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--no-save", dest="save", action="store_false")

    parser.add_argument("--num_block_open", type=int, choices=[-1, 0, 1, 2, 3, 4], default=0)

    parser.add_argument("--train_list", type=str, help="Data list for training data.")
    parser.add_argument("--val_list", type=str, help="Data list for validation data.")

    # group softmax arguments
    parser.add_argument("--finetune_type", choices=['flat_softmax', 'group_softmax'], default='group_softmax')
    parser.add_argument("--group_config", default="group_config/taxonomy_level0.npy")
    return parser


def get_mixup(dataset_size):
    return 0.0 if dataset_size < 20_000 else 0.1


def get_schedule(dataset_size):
    if dataset_size < 20_000:
        return [100, 200, 300, 400, 500]
    elif dataset_size < 500_000:
        return [500, 3000, 6000, 9000, 10_000]
    else:
        return [500, 6000, 12_000, 18_000, 20_000]


def get_lr(step, dataset_size, base_lr=0.003):
    """Returns learning-rate for `step` or None at the end."""
    supports = get_schedule(dataset_size)
    # Linear warmup
    if step < supports[0]:
        return base_lr * step / supports[0]
    # End of training
    elif step >= supports[-1]:
        return None
    # Staircase decays by factor of 10
    else:
        for s in supports[1:]:
            if s < step:
                base_lr /= 10
        return base_lr

