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
import logging
import logging.config
import os

from utils import hyperrule


def argparser():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained model.")
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring and checkpointing.")

    parser.add_argument("--model", help="Which variant to use.", default="BiT-S-R101x1")
    parser.add_argument("--logdir", required=True,
                        help="Where to log training info (small).")

    parser.add_argument("--bit_pretrained_dir", default='bit_pretrained_models',
                        help="Where to search for pretrained BiT models.")

    parser.add_argument("--dataset",
                        help="Choose the dataset. It should be easy to add your own! "
                             "Don't forget to set --datadir if necessary.")

    parser.add_argument("--batch", type=int, default=512,
                        help="Batch size.")
    parser.add_argument("--batch_split", type=int, default=1,
                        help="Number of batches to compute gradient on before updating weights.")
    parser.add_argument("--base_lr", type=float, default=0.003,
                        help="Base learning-rate for fine-tuning. Most likely default is best.")
    parser.add_argument("--eval_every", type=int, default=None,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    return parser


def setup_logger(args):
    """Creates and returns a fancy logger."""
    # return logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    # Why is setting up proper logging so !@?#! ugly?
    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
        },
        "handlers": {
            "stderr": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "logfile": {
                "level": "DEBUG",
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": os.path.join(args.logdir, args.name, "train.log"),
                "mode": "a",
            }
        },
        "loggers": {
            "": {
                "handlers": ["stderr", "logfile"],
                "level": "DEBUG",
                "propagate": True
            },
        }
    })
    logger = logging.getLogger(__name__)
    logger.flush = lambda: [h.flush() for h in logger.handlers]
    logger.info(args)
    return logger
