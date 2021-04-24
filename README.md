This is the source code for our paper [MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space]().
Code for model finetuning is adapted from [Google BiT](https://github.com/google-research/big_transfer).

### Usage

#### 1. Dataset Preparation

##### In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in
`./dataset/id_data/ILSVRC-2012/train` and  `./dataset/id_data/ILSVRC-2012/val`, respectively.

##### Out-of-distribution dataset

Please download the 4 OOD datasets we curated from the following links:
[iNaturalist](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz),
[SUN](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz),
[Places](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz),
[Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/),
and put them into `./dataset/ood_data/`. For more details about these OOD datasets, please check out our [paper]().

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

To facilitate the reproduction of the results reported in our paper, we also provide our 
[group-softmax finetuned model](http://pages.cs.wisc.edu/~huangrui/finetuned_model/BiT-S-R101x1-group-finetune.pth.tar) 
and [flat-softmax finetuned model](http://pages.cs.wisc.edu/~huangrui/finetuned_model/BiT-S-R101x1-flat-finetune.pth.tar).
After downloading the provided models, you can skip Step 3
and set `--model_path` in scripts in Step 4 accordingly.
