# Brain Tumor Segmentation on BraTS 2020 Using SWIN-UNET, MEDSAM and VLM

A comparative deep learning study evaluating three model architectures — **Swin-UNet**, **MedSAM**, and a **VLM adapter** — for multi-class glioma segmentation on multimodal MRI scans. Models are trained to delineate three clinically distinct tumor sub-regions: Necrotic Core (NCR/NET), Peritumoral Edema (ED), and Enhancing Tumor (ET).

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tumor Sub-regions](#tumor-sub-regions)
- [Models](#models)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Data Preparation](#data-preparation)
- [Train / Val / Test Split](#train--val--test-split)
- [Running the Pipeline](#running-the-pipeline)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Findings](#key-findings)
- [Limitations](#limitations)

---

## Overview

This project implements an end-to-end segmentation pipeline on the [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/) dataset. Each patient volume contains four MRI modalities (T1, T1Gd, T2, FLAIR), and the goal is to produce voxel-level predictions for four classes — background, NCR/NET, edema, and enhancing tumor — across all axial slices.

The pipeline covers:

- Exploratory data analysis and class distribution visualization
- Before/after preprocessing comparison with intensity histograms
- Parallel training of three architectures with identical data splits and loss functions
- Per-epoch logging of accuracy, precision, recall, F1, and per-class Dice
- Post-training evaluation including confusion matrices, HD95, and per-class accuracy
- Comparative analysis across all three models on train, val, and test splits

---

## Dataset

**Source:** [Brain Tumor Segmentation (BraTS2020) — Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
hosted by [@awsaf49](https://www.kaggle.com/awsaf49), derived from the official BraTS 2020 Challenge data.

| Property | Value |
|---|---|
| Kaggle dataset | `awsaf49/brats2020-training-data` |
| Volumes | 369 pre-operative MRI patient cases |
| Modalities | T1, T1Gd (contrast), T2, FLAIR |
| Format | HDF5 (`.h5`), pre-processed and converted from NIfTI |
| Resolution | 1 mm³ isotropic |
| Annotations | Manual segmentation, approved by experienced neuro-radiologists |
| Institutions | 19 contributing clinical sites |
| Metadata | `volume_no`, `slice_no`, `target` stored per slice |

All volumes are skull-stripped, co-registered to a common anatomical template, and resampled to 1 mm³. The Kaggle version packages all slices into HDF5 format for reduced memory overhead. Each file contains one patient volume with keys `t1`, `t1ce`, `t2`, `flair`, and `mask`.

**Download the dataset:**

```bash
# Using the Kaggle API (recommended)
pip install kaggle
kaggle datasets download -d awsaf49/brats2020-training-data
unzip brats2020-training-data.zip -d ./data/brats2020/
```

Or download manually from [kaggle.com/datasets/awsaf49/brats2020-training-data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) and extract into `./data/brats2020/`.

> **Kaggle API setup:** Place your `kaggle.json` credentials file at `~/.kaggle/kaggle.json` before running the command above. Instructions at [kaggle.com/docs/api](https://www.kaggle.com/docs/api).

---

## Tumor Sub-regions

BraTS uses a 4-class label scheme. This project remaps label 4 → 3 for contiguous indexing:

| Class | Original Label | Remapped | Description | Approx. Voxel % |
|---|---|---|---|---|
| Background | 0 | 0 | Healthy brain, CSF, white matter — everything outside the tumor | ~98.7% |
| NCR/NET | 1 | 1 | Necrotic and non-enhancing tumor core — dead/inactive tissue at the center | ~0.25% |
| Edema | 2 | 2 | Peritumoral edema — swelling surrounding the tumor mass | ~0.77% |
| Enhancing Tumor | 4 | 3 | Active tumor rim, visibly bright on T1Gd contrast scans | ~0.26% |

> **Note:** Background accounts for ~98.7% of all voxels. Overall accuracy is therefore not a meaningful headline metric — per-class Dice scores and HD95 are the primary evaluation criteria, consistent with BraTS challenge standards.

The three clinically reported composite regions are:
- **Whole Tumor (WT)** = ED + ET + NCR/NET
- **Tumor Core (TC)** = ET + NCR/NET
- **Enhancing Tumor (ET)** = ET only

---

## Models

### Swin-UNet
Swin Transformer encoder with a UNet-style convolutional decoder. Window-based multi-head self-attention captures long-range spatial dependencies across the brain slice, which is critical for irregular glioma shapes. Pretrained on ImageNet; fine-tuned end-to-end on BraTS.

### MedSAM
Segment Anything Model (SAM) with a ViT-B backbone, adapted for medical image segmentation. Trained here in a dense prediction setting without explicit bounding-box prompts. Six transformer encoder layers process 4-channel MRI input.

### VLM Adapter
Vision-Language Model style encoder inspired by BioViL-T / CLIP, with a CNN visual backbone and a segmentation head. Pretrained visual representations provide medical imaging priors that improve generalization on unseen patient data.

All three models:
- Accept input shape `(B, 4, 128, 128)` — 4 MRI modalities, resized to 128×128
- Produce output shape `(B, 4, 128, 128)` — 4-class logit maps
- Are trained with the same **Combined Dice + Cross-Entropy loss** to handle class imbalance

---

## Project Structure

```
brats_pipeline/
│
├── section1_config.py               # Hyperparameters, paths, device, split ratios
├── section2_dataset.py              # HDF5 Dataset class + volume-level train/val/test split
├── section3_eda.py                  # Class distribution, ground truth samples, modality grid
├── section4_preprocessing_viz.py   # Before/after normalization, histograms, augmentation viz
├── section5_models.py               # SwinUNet, MedSAM, VLM architectures + CombinedLoss
├── section6_training.py             # Training loop, per-epoch metrics, curve plots
├── section7_evaluation.py           # Confusion matrix, per-class accuracy, HD95, reports
├── section8_comparative_and_main.py # Comparative plots, final table, main() runner
│
├── data/
│   └── brats2020/                   # Place .h5 patient volumes here
│
├── outputs/
│   ├── eda/                         # Class distribution, GT sample, modality grid figures
│   ├── preprocessing/               # Before/after normalization figures
│   ├── training/                    # Loss/metric curves per model + 3-model comparison
│   ├── predictions/                 # Train & test prediction overlays per model
│   └── evaluation/                  # Confusion matrices, reports, final comparison table
│
└── checkpoints/                     # Best model .pth checkpoint per architecture
```

---

## Setup & Installation

**Requirements:** Python 3.9+, CUDA 11.8+ recommended

```bash
# Clone the repository
git clone https://github.com/your-username/brats2020-segmentation.git
cd brats2020-segmentation

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn scikit-learn scipy h5py Pillow
```

**Tested environment:**

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.1.0 |
| CUDA | 11.8 |
| scikit-learn | 1.3.2 |
| h5py | 3.10.0 |

---

## Data Preparation

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) — HDF5 files are already pre-converted, no NIfTI processing needed
2. Extract and place all `.h5` files in `./data/brats2020/`
3. Verify at least one file loads correctly: `python -c "import h5py; f=h5py.File('./data/brats2020/BraTS20_Training_001.h5','r'); print(list(f.keys()))"`

Expected HDF5 structure per file:

```
BraTS20_Training_001.h5
  ├── t1      → shape (155, H, W), float32
  ├── t1ce    → shape (155, H, W), float32
  ├── t2      → shape (155, H, W), float32
  ├── flair   → shape (155, H, W), float32
  └── mask    → shape (155, H, W), uint8   [labels: 0, 1, 2, 4]
```

---

## Train / Val / Test Split

Splits are performed at **volume (patient) level** — never at slice level — to prevent data leakage.

> If split at slice level, slices from the same patient would appear in both training and test sets. Since adjacent slices share anatomy, the model would be tested on data it has effectively already seen, inflating all metrics.

| Split | Ratio | Approx. Volumes (n=369) |
|---|---|---|
| Train | 70% | ~258 volumes / ~40,000 slices |
| Validation | 15% | ~55 volumes / ~8,500 slices |
| Test | 15% | ~56 volumes / ~8,700 slices |

Implementation uses two-step `sklearn.train_test_split`:

```python
# Step 1: carve out train
train_paths, temp_paths = train_test_split(all_volumes, test_size=0.30, random_state=42)

# Step 2: split remainder equally into val and test
val_paths, test_paths = train_test_split(temp_paths, test_size=0.50, random_state=42)
```

To change ratios, edit `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO` in `section1_config.py`.

---

## Running the Pipeline

**Run the full pipeline end-to-end:**

```bash
python section8_comparative_and_main.py
```

This sequentially runs EDA → preprocessing viz → trains all 3 models → evaluates on all splits → generates comparative analysis. Estimated runtime: 8–14 hours on a single A100 GPU for 50 epochs.

**Run individual sections:**

```bash
# EDA only (no training required)
python section3_eda.py

# Preprocessing visualizations only
python section4_preprocessing_viz.py

# Train a single model
python -c "
from section1_config import *
from section2_dataset import get_dataloaders
from section5_models import get_model
from section6_training import train_model
loaders = get_dataloaders(DATA_DIR)
model = get_model('swin_unet', device=DEVICE)
train_model(model, 'swin_unet', loaders[0], loaders[1], DEVICE)
"
```

**Key hyperparameters** (edit in `section1_config.py`):

```python
IMG_SIZE     = 128       # spatial resize
BATCH_SIZE   = 8
NUM_EPOCHS   = 50
LR           = 1e-4
WEIGHT_DECAY = 1e-5
```

---

## Results

All metrics reported on the **held-out test split** (15% of volumes, never seen during training or validation).

### Overall performance

| Model | Accuracy | Precision | Recall | F1 (macro) | Mean Dice | Inference (ms/slice) |
|---|---|---|---|---|---|---|
| **Swin-UNet** | **0.9957** | **0.8291** | **0.7980** | **0.8125** | **0.7505** | 6.70 |
| VLM | 0.9954 | 0.8239 | 0.7711 | 0.7958 | 0.7283 | **6.19** |
| MedSAM | 0.9935 | 0.7429 | 0.6851 | 0.7110 | 0.6154 | 8.68 |

### Per-class Dice score (test split)

| Model | NCR/NET | Edema | Enhancing Tumor | Mean Dice |
|---|---|---|---|---|
| **Swin-UNet** | 0.6743 | **0.7876** | **0.7896** | **0.7505** |
| VLM | 0.6455 | 0.7735 | 0.7659 | 0.7283 |
| MedSAM | 0.5306 | 0.6764 | 0.6392 | 0.6154 |

### Hausdorff Distance 95 — boundary precision (lower is better)

| Model | HD95 NCR (px) | HD95 Edema (px) | HD95 ET (px) |
|---|---|---|---|
| MedSAM | **7.81** | 24.01 | 3.00 |
| **Swin-UNet** | 8.54 | **19.03** | **1.00** |
| VLM | 8.60 | 19.86 | 1.41 |

### Generalization gap (train F1 → test F1)

| Model | Train F1 | Test F1 | Absolute Drop | Relative Drop |
|---|---|---|---|---|
| **VLM** | 0.8807 | 0.7958 | −0.0849 | **9.6%** |
| Swin-UNet | 0.9049 | 0.8125 | −0.0924 | 10.2% |
| MedSAM | 0.8180 | 0.7110 | −0.1070 | 13.1% |

---

## Evaluation Metrics

| Metric | What it measures | Where reported |
|---|---|---|
| **Dice Score** | Overlap between predicted and ground truth mask. Range 0–1, higher is better | Per class, per model |
| **HD95** | 95th percentile of boundary point distances in pixels. Lower = tighter boundary | Per tumor class, test split |
| **Precision** | Of all voxels predicted as class X, how many truly are X | Per class per split |
| **Recall** | Of all true class X voxels, how many did the model find | Per class per split |
| **F1 (macro)** | Harmonic mean of precision and recall, averaged equally across all 4 classes | Per split |
| **Per-class accuracy** | Fraction of true class X voxels correctly labeled, independent of other classes | Test split bar chart |
| **Confusion matrix** | 4×4 matrix showing all prediction/label combinations | Row-normalized, test split |

> **Note:** Overall accuracy is not a primary metric due to severe class imbalance. Background constitutes ~98.7% of all voxels, meaning a trivial model predicting only background would still achieve ~98.7% accuracy. Dice and HD95 are the clinically meaningful measures.

---

## Key Findings

**Swin-UNet achieves the best test performance** with Mean Dice 0.7505 and F1 0.8125. Its window-based multi-head self-attention captures the long-range spatial context needed to differentiate tumor sub-regions at irregular boundaries.

**VLM generalizes best** with only a 9.6% relative F1 drop from training to test — smaller than both other models. Pretrained visual representations from large-scale medical imaging corpora transfer robustly to unseen patients. It is also the fastest model at 6.19 ms/slice.

**MedSAM underperforms in this setting** due to architectural mismatch. SAM is designed for prompt-guided segmentation; deployed without spatial prompts in a dense multi-class setting, its decoder cannot localize sub-regions reliably. NCR/NET recall collapses to 0.466 on test — a 40.3% drop from training — the most severe generalization failure observed.

**NCR/NET is universally the hardest class** across all three models. All models drop 23–33% on NCR/NET F1 from train to test. This reflects the class's small voxel count (0.25%), isointense MRI signal, and patient-specific shape variability — not a model-specific failure. This is a known open problem in BraTS that persists even in top competition entries.

**Enhancing Tumor is the easiest class** to segment accurately. Its bright, distinct signal on T1Gd contrast scans creates a strong, consistent feature for all models to learn. Swin-UNet achieves HD95 of just 1.0 px for ET — meaning predicted boundaries are within 1mm of the radiologist annotation.

---

## Limitations

- **2D slice-wise processing:** All models operate on individual axial slices. 3D volumetric context, particularly cross-slice information, would improve NCR/NET boundary delineation where the core shape is defined across multiple slices.
- **MedSAM without prompts:** MedSAM is evaluated without bounding-box prompts, which is not its intended use case. Integrating an automated prompt generator (e.g., a lightweight 2D detector producing bounding boxes per class) would be the correct configuration for a fair comparison.
- **Class imbalance:** Despite Combined Dice + CE loss, NCR/NET remains severely underrepresented. Weighted sampling or targeted online hard example mining could improve small-class performance.
- **Single dataset:** All results are on BraTS 2020. Generalizability to other glioma datasets or clinical MRI protocols (different scanners, field strengths) has not been tested.
