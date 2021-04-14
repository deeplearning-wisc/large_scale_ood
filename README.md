This is the codebase for our paper [MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space]().

### Usage

#### 1. Download Dataset

##### In-distribution dataset

Please download [ImageNet-1k]() and place the training data and validation data in
`dataset/id_data/ILSVRC-2012/train` and  `dataset/id_data/ILSVRC-2012/val`, respectively.

##### Out-of-distribution dataset

Please download the four OOD datasets from the following links: [iNaturalist](), [SUN](), [Places](), [Textures]().
Then put them into `dataset/ood_data/`.

#### 2. Download Pre-trained BiT Model

Please download the [BiT pre-trained models]() and put them into the folder `bit_pretrained_models`.
The model used in our paper for main results is `BiT-S-R101x1`.

#### 3. Finetune Group(Flat)-softmax Model

For group-softmax finetuning (MOS), please run:

```
./scripts/finetune_group_softmax.sh
```

For flat-softmax finetuning (baselines), please run:

```
./scripts/finetune_flat_softmax.sh
```


#### 4. OOD Detection Evaluation

To reproduce MOS, please run:
```
./scripts/test_mos.sh iNaturalist(/SUN/Places/Textures)
```

To reproduce baselines, please run:
```
./scripts/test_baselines.sh MSP(/ODIN/Energy/Mahalanobis/KL_Div) iNaturalist(/SUN/Places/Textures)
```

Note: before test Mahalanobis, make sure you have run this first:
```
./scripts/tune_mahalanobis.sh
```

### Our Fine-tuned Model

We also provide our [group-softmax finetuned model]() and [flat-softmax finetuned model]() for reproducibility.
In order to reproduce our results, you can download these models and set `--model_path` in the above scripts
to be the path of the models you downloaded.