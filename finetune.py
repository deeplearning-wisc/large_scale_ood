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
"""Fine-tune a BiT model on some downstream dataset."""
# !/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin  # pylint: disable=g-importing-member

import numpy as np
import torch
import torchvision as tv

import resnetv2

from utils import common
from utils import hyperrule
from dataset import DatasetWithMeta, DatasetWithMetaGroup

from tensorboardX import SummaryWriter


def topk(output, target, ks=(1,)):
    """Returns one boolean vector for each k, whether the target is within the output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i


def mktrainval(args, logger):
    """Returns train and validation datasets."""
    precrop, crop = hyperrule.get_resolution_from_dataset(args.dataset)
    train_tx = tv.transforms.Compose([
        tv.transforms.Resize((precrop, precrop)),
        tv.transforms.RandomCrop((crop, crop)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.dataset == "imagenet2012":
        if args.finetune_type == 'group_softmax':
            train_set = DatasetWithMetaGroup(args.datadir, args.train_list, train_tx, num_group=args.num_groups)
            valid_set = DatasetWithMetaGroup(args.datadir, args.val_list, val_tx, num_group=args.num_groups)
        else:
            train_set = DatasetWithMeta(args.datadir, args.train_list, train_tx)
            valid_set = DatasetWithMeta(args.datadir, args.val_list, val_tx)
    else:
        raise ValueError(f"Sorry, we have not spent time implementing the "
                         f"{args.dataset} dataset in the PyTorch codebase. "
                         f"In principle, it should be easy to add :)")

    logger.info(f"Using a training set with {len(train_set)} images.")
    logger.info(f"Using a validation set with {len(valid_set)} images.")

    micro_batch_size = args.batch // args.batch_split

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=micro_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    if micro_batch_size <= len(train_set):
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=micro_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False)
    else:
        # In the few-shot cases, the total dataset size might be smaller than the batch-size.
        # In these cases, the default sampler doesn't repeat, so we need to make it do that
        # if we want to match the behaviour from the paper.
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
            sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

    return train_set, valid_set, train_loader, valid_loader


def run_eval(model, data_loader, logger, step, writer, group_slices):
    # switch to evaluate mode
    model.eval()

    logger.info("Running validation...")
    logger.flush()

    all_c, all_top1 = [], []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()

            # compute output, measure accuracy and record loss.
            logits = model(x)
            if group_slices:
                c, top1 = cal_acc(logits, y, group_slices)
            else:
                c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
                top1 = topk(logits, y, ks=(1,))[0]

            all_c.extend(c.cpu())  # Also ensures a sync point.
            all_top1.extend(top1.cpu())

    model.train()
    logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
                f"top1 {np.mean(all_top1):.2%}")
    logger.flush()
    writer.add_scalar('Val/loss', np.mean(all_c), step)
    writer.add_scalar('Val/top1', np.mean(all_top1), step)
    return all_c, all_top1


def mixup_data(x, y, l):
    """Returns mixed inputs, pairs of targets, and lambda"""
    indices = torch.randperm(x.shape[0]).to(x.device)

    mixed_x = l * x + (1 - l) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b


def mixup_criterion_flat(criterion, pred, y_a, y_b, l):
    return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def mixup_criterion_group(criterion, pred, y_a, y_b, l, group_slices):
    return l * get_loss(criterion, pred, y_a, group_slices) \
           + (1 - l) * get_loss(criterion, pred, y_b, group_slices)


def get_group_slices(classes_per_group):
    group_slices = []
    start = 0
    for num_cls in classes_per_group:
        end = start + num_cls + 1
        group_slices.append([start, end])
        start = end
    return torch.LongTensor(group_slices)


def cal_acc(logits, labels, group_slices):
    num_groups = group_slices.shape[0]
    loss = 0
    num_samples = logits.shape[0]

    all_group_max_score, all_group_max_class = [], []

    smax = torch.nn.Softmax(dim=-1).cuda()
    cri = torch.nn.CrossEntropyLoss(reduction='none').cuda()

    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]: group_slices[i][1]]
        group_label = labels[:, i]
        loss += cri(group_logit, group_label)

        group_softmax = smax(group_logit)
        group_softmax = group_softmax[:, 1:]    # disregard others category
        group_max_score, group_max_class = torch.max(group_softmax, dim=1)
        group_max_class += 1     # shift the class index by 1

        all_group_max_score.append(group_max_score)
        all_group_max_class.append(group_max_class)

    all_group_max_score = torch.stack(all_group_max_score, dim=1)
    all_group_max_class = torch.stack(all_group_max_class, dim=1)

    final_max_score, max_group = torch.max(all_group_max_score, dim=1)

    pred_cls_within_group = all_group_max_class[torch.arange(num_samples), max_group]
    # print(pred_cls_within_group)

    gt_class, gt_group = torch.max(labels, dim=1)

    selected_groups = (max_group == gt_group)

    pred_acc = torch.zeros(logits.shape[0]).bool().cuda()
    # print(pred_acc)

    pred_acc[selected_groups] = (pred_cls_within_group[selected_groups] == gt_class[selected_groups])

    return loss, pred_acc


def get_loss(criterion, logits, labels, group_slices):
    num_groups = group_slices.shape[0]
    loss = 0
    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]: group_slices[i][1]]
        group_label = labels[:, i]
        loss += criterion(group_logit, group_label)
    return loss


def main(args):
    logger = common.setup_logger(args)
    writer = SummaryWriter(pjoin(args.logdir, args.name, 'tensorboard_log'))

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)

    if args.finetune_type == 'group_softmax':
        classes_per_group = np.load(args.group_config)
        args.num_groups = len(classes_per_group)
        group_slices = get_group_slices(classes_per_group)
        group_slices.cuda()
        num_logits = len(train_set.classes) + args.num_groups
    else:
        classes_per_group, args.num_groups, group_slices = None, None, None
        num_logits = len(train_set.classes)

    model = resnetv2.KNOWN_MODELS[args.model](head_size=num_logits,
                                              zero_head=True,
                                              num_block_open=args.num_block_open)

    model_path = pjoin(args.bit_pretrained_dir, args.model + '.npz')
    logger.info(f"Loading model from {model_path}")
    model.load_from(np.load(model_path))

    logger.info("Moving model onto all GPUs")
    model = torch.nn.DataParallel(model)

    # Optionally resume from a checkpoint.
    # Load it to CPU first as we'll move the model to GPU later.
    # This way, we save a little bit of GPU memory when loading.
    step = 0

    # Note: no weight-decay!
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.SGD(trainable_params, lr=args.base_lr, momentum=0.9)

    # Resume fine-tuning if we find a saved model.
    savename = pjoin(args.logdir, args.name, "bit.pth.tar")
    try:
        logger.info(f"Model will be saved in '{savename}'")
        checkpoint = torch.load(savename, map_location="cpu")
        logger.info(f"Found saved model to resume from at '{savename}'")

        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        logger.info(f"Resumed at step {step}")
    except FileNotFoundError:
        logger.info("Fine-tuning from BiT")

    model = model.cuda()
    optim.zero_grad()

    model.train()

    mixup = hyperrule.get_mixup(len(train_set))
    cri = torch.nn.CrossEntropyLoss().cuda()

    logger.info("Starting finetuning!")
    accum_steps = 0
    mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

    for x, y in recycle(train_loader):

        # Schedule sending to GPU(s)
        x = x.cuda()
        y = y.cuda()

        # Update learning-rate, including stop training if over.
        lr = hyperrule.get_lr(step, len(train_set), args.base_lr)

        if lr is None:
            break
        for param_group in optim.param_groups:
            param_group["lr"] = lr

        if mixup > 0.0:
            x, y_a, y_b = mixup_data(x, y, mixup_l)

        # compute output
        logits = model(x)

        if args.finetune_type == 'group_softmax':
            if mixup > 0.0:
                c = mixup_criterion_group(cri, logits, y_a, y_b, mixup_l, group_slices)
            else:
                c = get_loss(cri, logits, y, group_slices)
        else:
            if mixup > 0.0:
                c = mixup_criterion_flat(cri, logits, y_a, y_b, mixup_l)
            else:
                c = cri(logits, y)
        c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

        # Accumulate grads
        (c / args.batch_split).backward()
        accum_steps += 1

        accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
        logger.info(
            f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  # pylint: disable=logging-format-interpolation
        logger.flush()

        writer.add_scalar('Train/loss', c_num, step)

        # Update params
        if accum_steps == args.batch_split:
            optim.step()
            optim.zero_grad()
        step += 1
        accum_steps = 0
        # Sample new mixup ratio for next batch
        mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

        # Run evaluation and save the model.
        if args.eval_every and step % args.eval_every == 0:
            run_eval(model, valid_loader, logger, step, writer, group_slices)
            if args.save:
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                }, savename)

    # Final eval at end of training.
    run_eval(model, valid_loader, logger, step, writer, group_slices)
    if args.save:
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
        }, savename)


if __name__ == "__main__":
    parser = common.argparser()
    parser.add_argument("--datadir", required=True,
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--no-save", dest="save", action="store_false")

    # parser.add_argument("--fix-backbone", dest="fix_backbone", action="store_true")
    parser.add_argument("--num_block_open", type=int, choices=[-1, 0, 1, 2, 3, 4], default=0)

    parser.add_argument("--train_list", type=str)
    parser.add_argument("--val_list", type=str)

    # group softmax arguments
    parser.add_argument("--finetune_type", choices=['flat_softmax', 'group_softmax'], default='group_softmax')
    parser.add_argument("--group_config", default="group_config/taxonomy_level0.npy")

    main(parser.parse_args())
