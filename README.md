 ---

<div align="center">    
 
# Towards Dynamic and Small Objects Refinement for Unsupervised Domain Adaptative Nighttime Segmentation

[![Paper](http://img.shields.io/badge/paper-arxiv.2310.04747-B31B1B.svg)](https://arxiv.org/abs/2310.04747)

</div>

This repository provides the official code for [Towards Dynamic and Small Objects Refinement for Unsupervised Domain Adaptative Nighttime Segmentation](https://arxiv.org/abs/2310.04747). The code is organized using [PyTorch Lightning](https://github.com/Lightning-AI/lightning). 

## Abstract

Nighttime semantic segmentation is essential for various applications, e.g., autonomous driving, which often faces challenges due to poor illumination and the lack of well-annotated datasets. Unsupervised domain adaptation (UDA) has shown potential for addressing the challenges and achieved remarkable results for nighttime semantic segmentation. However, existing methods still face limitations in 1) their reliance on style transfer or relighting models, which struggle to generalize to complex nighttime environments, and 2) their ignorance of dynamic and small objects like vehicles and traffic signs, which are difficult to be directly learned from other domains. This paper proposes a novel UDA method that refines both label and feature levels for dynamic and small objects for nighttime semantic segmentation. First, we propose a dynamic and small object refinement module to complement the knowledge of dynamic and small objects from the source domain to target nighttime domain. These dynamic and small objects are normally context-inconsistent in under-exposed conditions. Then, we design a feature prototype alignment module to reduce the domain gap by deploying contrastive learning between features and prototypes of the same class from different domains, while re-weighting the categories of dynamic and small objects. Extensive experiments on four benchmark datasets demonstrate that our method outperforms prior arts by a large margin for nighttime segmentation. Project page: https://rorisis.github.io/DSRNSS/.

## Usage
### Requirements

The code is run with Python 3.8.13. To install the packages, use:
```bash
pip install -r requirements.txt
```

### Set Data Directory

The following environment variable must be set:
```bash
export DATA_DIR=/path/to/data/dir
```

### Download the Data

Before running the code, download and extract the corresponding datasets to the directory `$DATA_DIR`.

#### UDA
<details>
  <summary>Cityscapes</summary>
  
  Download leftImg8bit_trainvaltest.zip and gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/) and extract them to `$DATA_DIR/Cityscapes`.

  ```
  $DATA_DIR
  ├── Cityscapes
  │   ├── leftImg8bit
  │   │   ├── train
  │   │   ├── val
  │   ├── gtFine
  │   │   ├── train
  │   │   ├── val
  ├── ...
  ```
  Afterwards, run the preparation script:
  ```bash
  python tools/convert_cityscapes.py $DATA_DIR/Cityscapes
  ```
</details>

<details>
  <summary>Dark Zurich</summary>
  
  Download Dark_Zurich_train_anon.zip, Dark_Zurich_val_anon.zip, and Dark_Zurich_test_anon_withoutGt.zip from [here](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/) and extract them to `$DATA_DIR/DarkZurich`.

  ```
  $DATA_DIR
  ├── DarkZurich
  │   ├── rgb_anon
  │   │   ├── train
  │   │   ├── val
  │   │   ├── val_ref
  │   │   ├── test
  │   │   ├── test_ref
  │   ├── gt
  │   │   ├── val
  ├── ...
  ```
</details>

<details>
  <summary>Nighttime Driving</summary>
  
  Download NighttimeDrivingTest.zip from [here](http://people.ee.ethz.ch/~daid/NightDriving/) and extract it to `$DATA_DIR/NighttimeDrivingTest`.

  ```
  $DATA_DIR
  ├── NighttimeDrivingTest
  │   ├── leftImg8bit
  │   │   ├── test
  │   ├── gtCoarse_daytime_trainvaltest
  │   │   ├── test
  ├── ...
  ```
</details>

<details>
  <summary>BDD100k-night</summary>
  
  Download `10k Images` and `Segmentation` from [here](https://bdd-data.berkeley.edu/portal.html#download) and extract them to `$DATA_DIR/bdd100k`.

  ```
  $DATA_DIR
  ├── bdd100k
  │   ├── images
  │   │   ├── 10k
  │   ├── labels
  │   │   ├── sem_seg
  ├── ...
  ```
</details>

<details>
  <summary>ACDC</summary>
  
  Download rgb_anon_trainvaltest.zip and gt_trainval.zip from [here](https://acdc.vision.ee.ethz.ch/download) and extract them to `$DATA_DIR/ACDC`.

  ```
  $DATA_DIR
  ├── ACDC
  │   ├── rgb_anon
  │   │   ├── fog
  │   │   ├── night
  │   │   ├── rain
  │   │   ├── snow
  │   ├── gt
  │   │   ├── fog
  │   │   ├── night
  │   │   ├── rain
  │   │   ├── snow
  ├── ...
  ```
</details>


### DSRNSS Training

Make sure to first download the trained UAWarpC model with the link provided [here](https://drive.google.com/drive/folders/1E-6shGVlVRn8DlgV5hCTOANdETOmzwsZ?usp=drive_link).
Enter the path to the UAWarpC model for `model.init_args.alignment_head.init_args.pretrained` in the config file you intend to run (or save the model to `./pretrained_models/`).

To train DSRNSS on DarkZurich (single GPU, with AMP) use the following command:

```bash
python tools/run.py fit --config configs/cityscapes_darkzurich/dsrnss_hrda.yaml --trainer.gpus 1 --trainer.precision 16
```
Other backbones are following corresponding configs.

### DSRNSS Testing

As mentioned in the previous section, modify the config file by adding the UAWarpC model path.
To evaluate DSRNSS e.g. on the DarkZurich validation set, use the following command:

```bash
python tools/run.py test --config configs/cityscapes_darkzurich/dsrnss_hrda.yaml --ckpt_path /path/to/trained/model --trainer.gpus 1
```

The results can be obainted from tensorbord:
```
tensorboard --logdir lightning_logs/version_x
```

### DSRNSS Predicting

```bash
python tools/run.py predict --config configs/cityscapes_darkzurich/dsrnss_hrda.yaml --ckpt_path /path/to/trained/model --trainer.gpus 1
```
To get test set scores for DarkZurich, predictions are evaluated on the respective evaluation server: [DarkZurich](https://codalab.lisn.upsaclay.fr/competitions/3783).

We also provide pretrained models, which can be downloaded from the link [here](https://drive.google.com/drive/folders/1g5YQcEd1H-cZ_slpHTWKLk9RtKtUAHBj?usp=drive_link). To evaluate them, simply provide them as the argument `--ckpt_path`.

## Citation

If you find this code useful in your research, please consider citing the paper:
```bibtex
@article{pan2023towards,
  title={Towards Dynamic and Small Objects Refinement for Unsupervised Domain Adaptative Nighttime Semantic Segmentation},
  author={Pan, Jingyi and Li, Sihang and Chen, Yucheng and Zhu, Jinjing and Wang, Lin},
  journal={arXiv preprint arXiv:2310.04747},
  year={2023}
}
```

## Credit

The pretrained backbone weights and code are from [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). DAFormer code is from the [original repo](https://github.com/lhoyer/DAFormer). Geometric matching code is from [this repo](https://github.com/PruneTruong/DenseMatching). Refign code is from [this repo](https://github.com/brdav/refign). Local correlation CUDA code is from [this repo](https://github.com/ClementPinard/Pytorch-Correlation-extension).

