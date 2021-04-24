This is the source code for our paper [MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space]().
Code for model finetuning is adapted from [Google BiT](https://github.com/google-research/big_transfer).

### Usage

#### 1. Dataset Preparation

##### In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in
`./dataset/id_data/ILSVRC-2012/train` and  `./dataset/id_data/ILSVRC-2012/val`, respectively.

##### Out-of-distribution dataset

We have curated 4 OOD datasets from 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf), 
and de-duplicated concepts overlapped with ImageNet-1k.

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset,
which can be download via the following links:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, we use the entire dataset, which can be downloaded from their
[original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Please put all downloaded OOD datasets into `./dataset/ood_data/`.
For more details about these OOD datasets, please check out our [paper]().

#### 2. Pre-trained Model Preparation

Please download the [BiT-S pre-trained model families](https://github.com/google-research/big_transfer)
and put them into the folder `./bit_pretrained_models`.
The backbone used in our paper for main results is `BiT-S-R101x1`.

#### 3. Group-softmax/Flat-softmax Model Finetuning

For group-softmax finetuning (MOS), please run:

```
./scripts/finetune_group_softmax.sh
```

For flat-softmax finetuning (baselines), please run:

```
./scripts/finetune_flat_softmax.sh
```


#### 4. OOD Detection Evaluation

To reproduce our MOS results, please run:
```
./scripts/test_mos.sh iNaturalist(/SUN/Places/Textures)
```

To reproduce baseline approaches, please run:
```
./scripts/test_baselines.sh MSP(/ODIN/Energy/Mahalanobis/KL_Div) iNaturalist(/SUN/Places/Textures)
```

Note: before testing Mahalanobis, make sure you have tuned and saved its hyperparameters first by running:
```
./scripts/tune_mahalanobis.sh
```

### Our Fine-tuned Model

To facilitate the reproduction of the results reported in our paper, we also provide our group-softmax finetuned model 
and flat-softmax finetuned model, which can be downloaded via the following links:
```bash
wget http://pages.cs.wisc.edu/~huangrui/finetuned_model/BiT-S-R101x1-group-finetune.pth.tar
wget http://pages.cs.wisc.edu/~huangrui/finetuned_model/BiT-S-R101x1-flat-finetune.pth.tar
```
After downloading the provided models, you can skip Step 3
and set `--model_path` in scripts in Step 4 accordingly.
