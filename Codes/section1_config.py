# =============================================================================
# BRATS 2020 BRAIN TUMOR SEGMENTATION — COMPLETE PIPELINE
# =============================================================================
# Dataset   : BraTS 2020 (HDF5 format)
# Task      : Multi-class segmentation — ET (label 4), ED (label 2), NCR/NET (label 1)
# Models    : VLM (BioViL-T) | MedSAM | Swin-UNet
# Split     : 70% Train / 15% Validation / 15% Test  (volume-level, no slice leakage)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

import os
import h5py
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)

import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {DEVICE}")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\Amma\Desktop\Brain_tumor_detection_23036\archive (2)\BraTS2020_training_data\content\data"
OUTPUT_DIR = r"C:\Users\Amma\Desktop\Brain_tumor_detection_23036\archive (2)\outputs"
CHECKPOINT_DIR = "./checkpoints"

for d in [OUTPUT_DIR, CHECKPOINT_DIR,
          f"{OUTPUT_DIR}/eda",
          f"{OUTPUT_DIR}/preprocessing",
          f"{OUTPUT_DIR}/training",
          f"{OUTPUT_DIR}/predictions",
          f"{OUTPUT_DIR}/evaluation"]:
    os.makedirs(d, exist_ok=True)

# ── Dataset constants ─────────────────────────────────────────────────────────
MODALITIES    = ["t1", "t1ce", "t2", "flair"]   # 4 MRI channels
NUM_CLASSES   = 4                                 # 0=BG, 1=NCR/NET, 2=ED, 4→3=ET
CLASS_NAMES   = ["Background", "NCR/NET", "Edema", "Enhancing Tumor"]
CLASS_COLORS  = ["#000000", "#FF4444", "#44CC44", "#FFFF00"]  # BraTS standard

# ── Training hyperparameters ──────────────────────────────────────────────────
IMG_SIZE      = 128      # spatial resize (H x W)
BATCH_SIZE    = 8
NUM_EPOCHS    = 50
LR            = 1e-4
WEIGHT_DECAY  = 1e-5

# ── Train / Val / Test split ratios ──────────────────────────────────────────
# Split is performed at VOLUME level (not slice level) to prevent data leakage.
# A volume = one patient's full 3D scan. All slices of a volume stay in one split.
# Ratios: 70% train | 15% validation | 15% test
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15

print("[Config] Configuration loaded successfully.")
print(f"[Config] Split → Train: {int(TRAIN_RATIO*100)}% | "
      f"Val: {int(VAL_RATIO*100)}% | Test: {int(TEST_RATIO*100)}%")