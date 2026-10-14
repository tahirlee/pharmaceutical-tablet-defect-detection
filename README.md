# Pharmaceutical Tablet Defect Detection

[![Paper](https://img.shields.io/badge/Paper-Pharmaceutics%202026-blue)](https://www.mdpi.com/1999-4923/18/10/1287)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fpharmaceutics18101287-informational)](https://doi.org/10.3390/pharmaceutics18101287)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-green)](https://doi.org/10.5281/zenodo.22773411)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code for the paper:

> **Artificial Intelligence-Based Quality Control of Pharmaceutical Tablets: A Visual Dataset and Benchmark for Automated Defect Detection**
> Tahir Mahmood, Sabi Ur Rehman, Arslan Niamat, Sung Jae Lee, Kang Ryoung Park
> *Pharmaceutics* 2026, 18(10), 1287 — [https://doi.org/10.3390/pharmaceutics18101287](https://doi.org/10.3390/pharmaceutics18101287)

This repository provides the training, evaluation and augmentation code used to benchmark 12 deep-learning architectures on **PharmaVision (PV)**, a public dataset for binary *normal vs. defective* classification of pharmaceutical tablets, together with a cross-dataset evaluation on **pillQC** without any fine-tuning.

---

## Highlights

- **PharmaVision (PV) dataset** — 640 smartphone images (320 normal / 320 defective) covering multiple products, backgrounds and lighting conditions.
- **Unified benchmark** of 12 CNN and transformer architectures under a single, fixed training protocol.
- **Cross-dataset generalization** to pillQC with no fine-tuning.
- **Efficiency reporting** — parameters, GFLOPs, latency and throughput for every model.
- **Key finding:** no single model wins on accuracy, cross-dataset robustness *and* speed. EfficientNetV2-S gives the best in-domain F1 (88.56%), but it is not statistically separable from Xception or Swin-B; GoogLeNet and EfficientNetV2-S transfer best to pillQC; AlexNet is the fastest.

---

## Dataset

The PharmaVision (PV) dataset is publicly available on Kaggle:

- Record: https://www.kaggle.com/datasets/tahirlee/pharmaceutical-tablets-dataset
- DOI: https://doi.org/10.34740/kaggle/dsv/20550042

| Split | Normal | Defective | Total |
|---|---:|---:|---:|
| Train | 190 | 190 | 380 |
| Val | 30 | 30 | 60 |
| Test | 100 | 100 | 200 |
| **Total** | **320** | **320** | **640** |
| Train (augmented) | 570 | 570 | 1,140 |

- Splits are made at the **physical-tablet level**, and the test set was acquired in a separate session about two weeks apart, to prevent leakage.
- **Defect types** (defective class): chips (111), abrasion / surface erosion (75), cracks (59), discoloration / mottling / specks (27), broken tablets (25), multiple defect types (20), lamination (3).
- **Origin:** 92 naturally occurring production defects and 228 defects introduced manually by pharmacists.
- **Acquisition:** Samsung Galaxy A35 and A05 smartphones at ~30 cm; images resized to 224 × 224.
- The augmented training set (flips, ±12° rotation, ±7% brightness/contrast) is used for training only — Val and Test are never augmented.

### Expected folder layout

```
datasets/
└── pv/
    ├── class_indices.json   # e.g. {"0": "defective", "1": "normal"}
    ├── Train/
    │   ├── defective/
    │   └── normal/
    ├── Val/
    │   ├── defective/
    │   └── normal/
    └── test/
        ├── defective/
        └── normal/
```

For the pillQC cross-dataset evaluation, arrange the 43 *chip* images as `defective/` and the 149 *normal* images as `normal/` under a `test/` folder (the *dirt* class is excluded), and reuse the same `class_indices.json`.

---

## Repository structure

```
├── models/                 # Architecture definitions (AlexNet, VGG, ResNet, DenseNet,
│                           #   Xception, DLA, ShuffleNetV2, MobileNetV3, EfficientNetV2,
│                           #   ViT, Swin, ...) and the model registry in __init__.py
├── utils/
│   ├── train_engin.py      # train / evaluate loops
│   ├── lr_methods.py       # warm-up + cosine LR schedule
│   ├── distrubute_utils.py # distributed-training helpers
│   └── visualization.py
├── data_augmentation.py    # defect-safe offline augmentation of the Train split
├── train.py                # training with F2-based checkpoint selection & early stopping
├── test.py                 # test metrics + params / GFLOPs / latency / FPS
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/tahirlee/pharmaceutical-tablet-defect-detection.git
cd pharmaceutical-tablet-defect-detection

pip install torch torchvision          # the paper used PyTorch 2.1.0 + CUDA 12.1
pip install numpy pillow scikit-learn matplotlib tqdm tensorboard thop timm
```

`thop` is optional and only needed for GFLOPs reporting in `test.py`.

---

## Usage

### 1. (Optional) Build the augmented training set

```bash
python data_augmentation.py \
    --input_dir  datasets/pv/Train \
    --output_dir datasets/pv/Train_augmented \
    --copies_per_image 2
```

Each original image is copied unchanged and receives two mild variants (190 → 570 per class). Then point `Train` at the augmented folder (rename or symlink) and pass `--offline_augmented` when training. The released augmented set on Zenodo can be used instead.

### 2. Train

To reproduce the paper's protocol (ImageNet-pretrained, full fine-tuning, plain cross-entropy, Adam, lr 1e-4 with 1-epoch warm-up and cosine decay, weight decay 1e-4, batch size 16, up to 100 epochs, early stopping after 15 epochs without validation F2 improvement, threshold 0.40, seed 7):

```bash
python train.py \
    --model efficient_v2_small \
    --data_path datasets/pv \
    --num_classes 2 \
    --epochs 100 --batch_size 16 --lr 1e-4 --weight_decay 1e-4 --patience 15 \
    --loss_type ce --no_class_weights \
    --no_freeze_features --no_freeze_classifier_base \
    --offline_augmented \
    --positive_class_index 0 --val_threshold 0.40
```

Set `--positive_class_index` to the index of `defective` in your `class_indices.json`. The best checkpoint (by validation F2) is saved to `trained_models/weights/<model>/<model>.pth`.

> **Note:** some `train.py` defaults (focal loss, frozen backbone, class weighting) differ from the paper's protocol; pass the flags above explicitly to reproduce the reported results.

### 3. Evaluate

```bash
# PV test set
python test.py --model efficient_v2_small --data_path datasets/pv --test_subfolder test \
    --pos_label defective --val_threshold 0.40

# pillQC cross-dataset (no fine-tuning)
python test.py --model efficient_v2_small --data_path datasets/pillQC --test_subfolder test \
    --pos_label defective --val_threshold 0.40
```

`test.py` reports accuracy, precision, recall, F1 and the confusion matrix (defective = positive class), plus total/trainable parameters, GFLOPs, average latency and FPS.

### Benchmarked models and `--model` keys

| Paper name | `--model` |
|---|---|
| AlexNet | `alexnet` |
| VGG-19 | `vgg_big` |
| GoogLeNet | `googlenet` |
| ResNet-50 | `resnet` |
| DenseNet-169 | `densenet` |
| Xception | `xception` |
| DLA-34 | `dla` |
| ShuffleNetV2 (1.0×) | `shufflenet` |
| MobileNetV3-small | `mobilenet_v3` |
| EfficientNetV2-S | `efficient_v2_small` |
| ViT-B/32 | `vision_transformer_small` |
| Swin-B | `swin_transformer` |

---

## Results

All results use a decision threshold of 0.40 with *defective* as the positive class. Values are percentages.

### In-domain (PV test set) — selected models

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| **EfficientNetV2-S** | 88.12 | 89.00 | **88.56** |
| Xception | 83.49 | 91.00 | 87.08 |
| Swin-B | – | – | 85.71 |
| ShuffleNetV2 | 92.86 | 78.00 | 84.78 |
| VGG-19 | – | – | 84.51 |
| MobileNetV3-small | – | – | 71.28 |
| AlexNet | – | – | 67.91 |

EfficientNetV2-S reaches 88.50% accuracy. See Table 7 of the [paper](https://doi.org/10.3390/pharmaceutics18101287) for the full results of all 12 models and confidence intervals.

### Cross-dataset (pillQC, no fine-tuning)

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| AlexNet | 56.77 | 22.97 | 39.53 | 29.06 |
| VGG-19 | 88.02 | 73.81 | 72.09 | 72.94 |
| **GoogLeNet** | 89.58 | 78.05 | 74.42 | **76.19** |
| ResNet-50 | 90.10 | 96.15 | 58.14 | 72.46 |
| DenseNet-169 | 83.33 | 78.95 | 34.88 | 48.39 |
| Xception | 88.54 | 83.87 | 60.47 | 70.27 |
| DLA-34 | 86.46 | 79.31 | 53.49 | 63.89 |
| ShuffleNetV2 | 76.56 | 47.83 | 51.16 | 49.44 |
| MobileNetV3-small | 83.33 | 86.67 | 30.23 | 44.83 |
| EfficientNetV2-S | **90.62** | 93.10 | 62.79 | 75.00 |
| ViT-B/32 | 54.17 | 27.72 | 65.12 | 38.89 |
| Swin-B | 80.21 | 61.90 | 30.23 | 40.62 |

### Computational efficiency (NVIDIA RTX 3060, batch size 1)

| Model | Params (M) | GFLOPs | Median latency (ms) | Throughput (FPS) |
|---|---:|---:|---:|---:|
| AlexNet | 57.01 | 1.42 | 1.66 | **602.41** |
| VGG-19 | 139.58 | 39.26 | 6.81 | 146.84 |
| GoogLeNet | 5.60 | 3.02 | 15.35 | 65.15 |
| ResNet-50 | 23.51 | 8.26 | 12.46 | 80.26 |
| DenseNet-169 | 12.48 | 6.87 | 40.23 | 24.86 |
| Xception | 20.81 | 9.19 | 11.86 | 84.32 |
| DLA-34 | 15.27 | 6.18 | 7.99 | 125.16 |
| ShuffleNetV2 | **1.26** | 0.30 | 12.54 | 79.74 |
| MobileNetV3-small | 1.52 | **0.12** | 9.26 | 107.99 |
| EfficientNetV2-S | 20.18 | 5.80 | 33.68 | 29.69 |
| ViT-B/32 | 87.46 | 5.90 | 10.40 | 96.15 |
| Swin-B | 86.74 | 20.44 | 41.19 | 24.28 |

Throughput is computed as 1000 / median latency.

---

## Citation

If you use this code or the PharmaVision dataset, please cite:

```bibtex
@article{mahmood2026pharmavision,
  title   = {Artificial Intelligence-Based Quality Control of Pharmaceutical Tablets:
             A Visual Dataset and Benchmark for Automated Defect Detection},
  author  = {Mahmood, Tahir and Rehman, Sabi Ur and Niamat, Arslan and
             Lee, Sung Jae and Park, Kang Ryoung},
  journal = {Pharmaceutics},
  volume  = {18},
  number  = {10},
  pages   = {1287},
  year    = {2026},
  publisher = {MDPI},
  doi     = {10.3390/pharmaceutics18101287}
}
```

---

## License

The code in this repository is released under the [MIT License](LICENSE). Please refer to the Zenodo record for the dataset's license terms.

## Contact

Tahir Mahmood — Division of Electronics and Electrical Engineering, Dongguk University, Seoul, Republic of Korea
Corresponding author: Kang Ryoung Park, Dongguk University
