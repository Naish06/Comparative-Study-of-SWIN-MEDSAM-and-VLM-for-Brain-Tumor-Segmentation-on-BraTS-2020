# =============================================================================
# SECTION 2 — DATASET LOADER & TRAIN / VAL / TEST SPLIT
# =============================================================================
# HDF5 structure (actual dataset):
#   Each .h5 file = one 2-D axial slice (NOT a full 3-D volume).
#   Filename pattern: volume_XXX_slice_YYY.h5
#   Keys:
#       'image'  → shape (240, 240, 4)  float64   (4 MRI modalities: T1, T1ce, T2, FLAIR)
#       'mask'   → shape (240, 240, 3)  uint8     (3 binary channels, disjoint)
#                  ch0 = NCR/NET  (label 1)
#                  ch1 = Edema    (label 2)
#                  ch2 = ET       (label 3, originally label 4 in BraTS)
#
# Label mapping (multi-channel binary → single integer label map):
#   Background (no channel active) → 0
#   ch0 active → 1 (NCR/NET)
#   ch1 active → 2 (Edema)
#   ch2 active → 3 (Enhancing Tumor)
#
# Split is performed at VOLUME (patient) level — NOT at slice level.
# =============================================================================

import os, h5py, random, re, numpy as np, torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────
# 2.1  Convert 3-channel binary mask → single integer label map
# ─────────────────────────────────────────────────────────────────
def mask_channels_to_labelmap(mask_3ch: np.ndarray) -> np.ndarray:
    """
    Converts (H, W, 3) binary mask → (H, W) integer label map.
    ch0 → 1 (NCR/NET), ch1 → 2 (Edema), ch2 → 3 (ET), else 0 (BG).
    Channels are disjoint, so simple weighted sum works.
    """
    label_map = np.zeros(mask_3ch.shape[:2], dtype=np.int64)
    label_map[mask_3ch[:, :, 0] > 0] = 1   # NCR/NET
    label_map[mask_3ch[:, :, 1] > 0] = 2   # Edema
    label_map[mask_3ch[:, :, 2] > 0] = 3   # Enhancing Tumor
    return label_map


# ─────────────────────────────────────────────────────────────────
# 2.2  Group slice files by volume
# ─────────────────────────────────────────────────────────────────
def group_slices_by_volume(data_dir: str):
    """
    Scans data_dir for .h5 files with pattern volume_XXX_slice_YYY.h5.
    Returns:
        volume_dict : dict {volume_name: [sorted list of slice file paths]}
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
    volume_dict = defaultdict(list)

    pattern = re.compile(r"^(volume_\d+)_slice_(\d+)\.h5$")
    for fname in all_files:
        m = pattern.match(fname)
        if m:
            vol_name = m.group(1)
            slice_idx = int(m.group(2))
            volume_dict[vol_name].append((slice_idx, os.path.join(data_dir, fname)))

    # Sort slices within each volume by slice index
    for vol_name in volume_dict:
        volume_dict[vol_name].sort(key=lambda x: x[0])
        volume_dict[vol_name] = [path for _, path in volume_dict[vol_name]]

    return dict(volume_dict)


# ─────────────────────────────────────────────────────────────────
# 2.3  BraTS Slice-Level HDF5 Dataset
# ─────────────────────────────────────────────────────────────────
class BraTSDataset(Dataset):
    """
    Loads 2-D axial slices from BraTS HDF5 files (one file per slice).

    Args:
        slice_paths : list of .h5 file paths (one per 2-D slice)
        img_size    : (H, W) spatial resize target
        augment     : apply random flip / rotation during training
        phase       : 'train' | 'val' | 'test'  (controls augmentation)
    """

    def __init__(self, slice_paths, img_size=128, augment=False, phase="train"):
        self.img_size    = img_size
        self.augment     = augment
        self.phase       = phase
        self.slice_paths = slice_paths

        print(f"[Dataset] {phase}: {len(slice_paths)} slices loaded")

    def __len__(self):
        return len(self.slice_paths)

    def _load_slice(self, fpath):
        """Load image and mask from a single slice .h5 file."""
        with h5py.File(fpath, "r") as f:
            # image: (240, 240, 4) → transpose to (4, 240, 240) channels-first
            image = f["image"][()].astype(np.float32)           # (H, W, 4)
            image = np.transpose(image, (2, 0, 1))              # (4, H, W)

            # mask: (240, 240, 3) binary channels → single label map (H, W)
            mask_3ch = f["mask"][()]                             # (H, W, 3)
            mask = mask_channels_to_labelmap(mask_3ch)           # (H, W) int64
        return image, mask

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Per-channel z-score normalization (skip zero-padded regions)."""
        out = np.zeros_like(image)
        for c in range(image.shape[0]):
            ch = image[c]
            nonzero = ch[ch > 0]
            if nonzero.size > 0:
                mu, sigma = nonzero.mean(), nonzero.std() + 1e-8
                out[c] = (ch - mu) / sigma
            else:
                out[c] = ch
        return out

    def _resize(self, image: np.ndarray, mask: np.ndarray):
        """Resize to (img_size, img_size) using bilinear for image, nearest for mask."""
        from PIL import Image as PILImage
        sz = self.img_size
        resized_img = np.zeros((image.shape[0], sz, sz), dtype=np.float32)
        for c in range(image.shape[0]):
            pil = PILImage.fromarray(image[c]).resize((sz, sz), PILImage.BILINEAR)
            resized_img[c] = np.array(pil)
        pil_mask = PILImage.fromarray(mask.astype(np.uint8)).resize((sz, sz), PILImage.NEAREST)
        resized_mask = np.array(pil_mask).astype(np.int64)
        return resized_img, resized_mask

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        """Random horizontal flip + random 90° rotation."""
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask  = np.flip(mask, axis=1).copy()
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask  = np.rot90(mask,  k=k, axes=(0, 1)).copy()
        return image, mask

    def __getitem__(self, idx):
        fpath = self.slice_paths[idx]
        image, mask = self._load_slice(fpath)

        # Normalize
        image = self._normalize(image)

        # Resize
        image, mask = self._resize(image, mask)

        # Augment (training only)
        if self.augment and self.phase == "train":
            image, mask = self._augment(image, mask)

        return torch.tensor(image, dtype=torch.float32), \
               torch.tensor(mask,  dtype=torch.long)


# ─────────────────────────────────────────────────────────────────
# 2.4  Volume-level Train / Val / Test split
# ─────────────────────────────────────────────────────────────────
def build_splits(data_dir: str, train_r=0.70, val_r=0.15, seed=42):
    """
    Split at VOLUME (patient) level to prevent data leakage.
    Groups slice files by volume, splits volume names, then returns
    flat lists of slice paths per split.

    Returns:
        train_paths, val_paths, test_paths  — flat lists of slice .h5 file paths
    """
    volume_dict = group_slices_by_volume(data_dir)
    volume_names = sorted(volume_dict.keys())

    if len(volume_names) == 0:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    n = len(volume_names)

    # First split: train vs (val + test)
    train_vols, temp_vols = train_test_split(
        volume_names, test_size=(1 - train_r), random_state=seed, shuffle=True
    )
    # Second split: val vs test (equal halves of the remainder)
    val_vols, test_vols = train_test_split(
        temp_vols, test_size=0.5, random_state=seed
    )

    # Flatten volume names → list of slice paths
    train_paths = [p for v in train_vols for p in volume_dict[v]]
    val_paths   = [p for v in val_vols   for p in volume_dict[v]]
    test_paths  = [p for v in test_vols  for p in volume_dict[v]]

    print(f"\n[Split] Total volumes : {n}")
    print(f"[Split] Train volumes : {len(train_vols)} ({len(train_vols)/n*100:.1f}%) → {len(train_paths)} slices")
    print(f"[Split] Val   volumes : {len(val_vols)}   ({len(val_vols)/n*100:.1f}%) → {len(val_paths)} slices")
    print(f"[Split] Test  volumes : {len(test_vols)}  ({len(test_vols)/n*100:.1f}%) → {len(test_paths)} slices")
    return train_paths, val_paths, test_paths


# ─────────────────────────────────────────────────────────────────
# 2.5  DataLoader factory
# ─────────────────────────────────────────────────────────────────
def get_dataloaders(data_dir, img_size=128, batch_size=8, num_workers=0):
    """
    Returns train_loader, val_loader, test_loader + raw path lists.
    Note: num_workers=0 by default on Windows to avoid multiprocessing issues.
    """
    train_paths, val_paths, test_paths = build_splits(data_dir)

    train_ds = BraTSDataset(train_paths, img_size=img_size, augment=True,  phase="train")
    val_ds   = BraTSDataset(val_paths,   img_size=img_size, augment=False, phase="val")
    test_ds  = BraTSDataset(test_paths,  img_size=img_size, augment=False, phase="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"\n[DataLoader] Train batches : {len(train_loader)}")
    print(f"[DataLoader] Val   batches : {len(val_loader)}")
    print(f"[DataLoader] Test  batches : {len(test_loader)}")

    return (train_loader, val_loader, test_loader,
            train_paths, val_paths, test_paths)

if __name__ == "__main__":
    from section1_config import DATA_DIR, IMG_SIZE, BATCH_SIZE
    print(f"--- Running Dataset Test ---")
    get_dataloaders(DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    print("--- Dataset Module Completed Successfully ---")