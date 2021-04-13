from utils import log
import resnetv2
import torch
import time

import numpy as np

from utils.test_utils import arg_parser, mk_id_ood, get_measures
from finetune import get_group_slices
import os


def cal_ood_score(logits, group_slices):
    num_groups = group_slices.shape[0]

    all_group_ood_score_MOS = []

    smax = torch.nn.Softmax(dim=-1).cuda()
    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]: group_slices[i][1]]

        group_softmax = smax(group_logit)
        group_others_score = group_softmax[:, 0]

        all_group_ood_score_MOS.append(-group_others_score)

    all_group_ood_score_MOS = torch.stack(all_group_ood_score_MOS, dim=1)
    final_max_score_MOS, _ = torch.max(all_group_ood_score_MOS, dim=1)
    return final_max_score_MOS.data.cpu().numpy()


def iterate_data(data_loader, model, group_slices):
    confs_mos = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            logits = model(x)
            conf_mos = cal_ood_score(logits, group_slices)
            confs_mos.extend(conf_mos)

    return np.array(confs_mos)


def run_eval(model, in_loader, out_loader, logger, group_slices):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()

    logger.info("Processing in-distribution data...")
    in_confs = iterate_data(in_loader, model, group_slices)

    logger.info("Processing out-of-distribution data...")
    out_confs = iterate_data(out_loader, model, group_slices)

    in_examples = in_confs.reshape((-1, 1))
    out_examples = out_confs.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

    logger.info('============Results for MOS============')
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))
    
    logger.flush()


def main(args):
    logger = log.setup_logger(args)

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    in_set, out_set, in_loader, out_loader = mk_id_ood(args, logger)

    classes_per_group = np.load(args.group_config)
    args.num_groups = len(classes_per_group)
    group_slices = get_group_slices(classes_per_group)
    group_slices.cuda()
    num_logits = len(in_set.classes) + args.num_groups

    logger.info(f"Loading model from {args.model_path}")
    model = resnetv2.KNOWN_MODELS[args.model](head_size=num_logits)

    state_dict = torch.load(args.model_path)
    model.load_state_dict_custom(state_dict['model'])

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    start_time = time.time()
    run_eval(model, in_loader, out_loader, logger, group_slices)
    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()

    parser.add_argument("--group_config", default="group_config/taxonomy_level0.npy")

    main(parser.parse_args())
