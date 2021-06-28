from utils import log
import resnetv2
import torch
import time
import torchvision as tv
import numpy as np
import argparse
from dataset import DatasetWithMeta
from utils.test_utils import get_measures
import os
from torch.autograd import Variable
from utils.mahalanobis_lib import sample_estimator, get_Mahalanobis_score
import torch.nn as nn
from sklearn.linear_model import LogisticRegressionCV


torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


def mktrainval(args, logger):
    """Returns train and validation datasets."""
    crop = 480

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = DatasetWithMeta(args.datadir, args.train_list, val_tx)
    valid_set = DatasetWithMeta(args.datadir, args.val_list, val_tx)

    logger.info(f"Using a training set with {len(train_set)} images.")
    logger.info(f"Using a validation set with {len(valid_set)} images.")

    micro_batch_size = args.batch

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=micro_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return train_set, valid_set, train_loader, valid_loader


def tune_mahalanobis_hyperparams(args, model, num_classes, train_loader, val_loader, logger):

    save_dir = os.path.join(args.logdir, args.name, 'tmp')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()

    # set information about feature extaction
    temp_x = torch.rand(2, 3, 480, 480)
    temp_x = Variable(temp_x).cuda()
    temp_list = model(x=temp_x, layer_index='all')[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    logger.info('get sample mean and covariance')
    filename = os.path.join(save_dir, 'mean_and_precision.npy')

    if not os.path.exists(filename):
        sample_mean, precision = sample_estimator(model, num_classes, feature_list, train_loader)
        np.save(filename, np.array([sample_mean, precision]))

    sample_mean, precision = np.load(filename, allow_pickle=True)
    sample_mean = [s.cuda() for s in sample_mean]
    precision = [torch.from_numpy(p).float().cuda() for p in precision]

    logger.info('train logistic regression model')
    m = 500

    train_in = []
    train_in_label = []
    train_out = []

    val_in = []
    val_in_label = []
    val_out = []

    cnt = 0
    for data, target in val_loader:
        data = data.numpy()
        target = target.numpy()
        for x, y in zip(data, target):
            cnt += 1
            if cnt <= m:
                train_in.append(x)
                train_in_label.append(y)
            elif cnt <= 2*m:
                val_in.append(x)
                val_in_label.append(y)

            if cnt == 2*m:
                break
        if cnt == 2*m:
            break

    logger.info('In {} {}'.format(len(train_in), len(val_in)))

    criterion = nn.CrossEntropyLoss().cuda()
    adv_noise = 0.05

    args.batch_size = args.batch
    for i in range(int(m/args.batch_size) + 1):
        if i*args.batch_size >= m:
            break
        data = torch.tensor(train_in[i*args.batch_size:min((i+1)*args.batch_size, m)])
        target = torch.tensor(train_in_label[i*args.batch_size:min((i+1)*args.batch_size, m)])
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        # output = model(data)

        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True).cuda()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        train_out.extend(adv_data.cpu().numpy())

    for i in range(int(m/args.batch_size) + 1):
        if i*args.batch_size >= m:
            break
        data = torch.tensor(val_in[i*args.batch_size:min((i+1)*args.batch_size, m)])
        target = torch.tensor(val_in_label[i*args.batch_size:min((i+1)*args.batch_size, m)])
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        # output = model(data)

        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True).cuda()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        val_out.extend(adv_data.cpu().numpy())

    logger.info('Out {} {}'.format(len(train_out),len(val_out)))

    train_lr_data = []
    train_lr_label = []
    train_lr_data.extend(train_in)
    train_lr_label.extend(np.zeros(m))
    train_lr_data.extend(train_out)
    train_lr_label.extend(np.ones(m))
    train_lr_data = torch.tensor(train_lr_data)
    train_lr_label = torch.tensor(train_lr_label)

    best_fpr = 1.1
    best_magnitude = 0.0

    for magnitude in [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]:
        train_lr_Mahalanobis = []
        total = 0
        for data_index in range(int(np.floor(train_lr_data.size(0) / args.batch_size)) + 1):
            if total >= 2*m:
                break
            data = train_lr_data[total : total + args.batch_size].cuda()
            total += args.batch_size
            Mahalanobis_scores = get_Mahalanobis_score(data, model, num_classes, sample_mean, precision, num_output, magnitude)
            train_lr_Mahalanobis.extend(Mahalanobis_scores)

        train_lr_Mahalanobis = np.asarray(train_lr_Mahalanobis, dtype=np.float32)
        regressor = LogisticRegressionCV(n_jobs=-1).fit(train_lr_Mahalanobis, train_lr_label)

        logger.info('Logistic Regressor params: {} {}'.format(regressor.coef_, regressor.intercept_))

        t0 = time.time()
        f1 = open(os.path.join(save_dir, "confidence_mahalanobis_In.txt"), 'w')
        f2 = open(os.path.join(save_dir, "confidence_mahalanobis_Out.txt"), 'w')

    ########################################In-distribution###########################################
        logger.info("Processing in-distribution images")

        count = 0
        all_confidence_scores_in, all_confidence_scores_out = [], []
        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_in[i * args.batch_size : min((i+1) * args.batch_size, m)]).cuda()
            # if j<1000: continue
            batch_size = images.shape[0]
            Mahalanobis_scores = get_Mahalanobis_score(images, model, num_classes, sample_mean, precision, num_output, magnitude)
            confidence_scores_in = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
            all_confidence_scores_in.extend(confidence_scores_in)

            for k in range(batch_size):
                f1.write("{}\n".format(confidence_scores_in[k]))

            count += batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

    ###################################Out-of-Distributions#####################################
        t0 = time.time()
        logger.info("Processing out-of-distribution images")
        count = 0

        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_out[i * args.batch_size : min((i+1) * args.batch_size, m)]).cuda()
            # if j<1000: continue
            batch_size = images.shape[0]

            Mahalanobis_scores = get_Mahalanobis_score(images, model, num_classes, sample_mean, precision, num_output, magnitude)

            confidence_scores_out = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
            all_confidence_scores_out.extend(confidence_scores_out)

            for k in range(batch_size):
                f2.write("{}\n".format(confidence_scores_out[k]))

            count += batch_size
            logger.info("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

        f1.close()
        f2.close()

        # results = metric(save_dir, stypes)
        # print_results(results, stypes)
        # fpr = results['mahalanobis']['FPR']
        all_confidence_scores_in = np.array(all_confidence_scores_in).reshape(-1, 1)
        all_confidence_scores_out = np.array(all_confidence_scores_out).reshape(-1, 1)
        logger.info(all_confidence_scores_in.shape)
        logger.info(all_confidence_scores_out.shape)

        _, _, _, fpr = get_measures(all_confidence_scores_in, all_confidence_scores_out)

        if fpr < best_fpr:
            best_fpr = fpr
            best_magnitude = magnitude
            best_regressor = regressor

    logger.info('Best Logistic Regressor params: {} {}'.format(best_regressor.coef_, best_regressor.intercept_))
    logger.info('Best magnitude: {}'.format(best_magnitude))

    return sample_mean, precision, best_regressor, best_magnitude


def main(args):
    logger = log.setup_logger(args)

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    train_set, val_set, train_loader, val_loader = mktrainval(args, logger)

    logger.info(f"Loading model from {args.model_path}")
    model = resnetv2.KNOWN_MODELS[args.model](head_size=len(train_set.classes))
    state_dict = torch.load(args.model_path)
    model.load_state_dict_custom(state_dict['model'])

    logger.info("Moving model onto all GPUs")
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    logger.info('Tuning hyper-parameters...')
    sample_mean, precision, best_regressor, best_magnitude \
        = tune_mahalanobis_hyperparams(args, model, len(val_set.classes), train_loader, val_loader, logger)

    logger.info('saving results...')
    save_dir = os.path.join(args.logdir, args.name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'results'),
            np.array([sample_mean, precision, best_regressor.coef_, best_regressor.intercept_, best_magnitude]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", required=True)

    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)

    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")

    parser.add_argument("--model", default="BiT-S-R101x1",
                        help="Which variant to use")
    parser.add_argument("--model_path", type=str)

    parser.add_argument("--logdir", required=True,
                        help="Where to log training info (small).")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring and checkpointing.")

    main(parser.parse_args())
