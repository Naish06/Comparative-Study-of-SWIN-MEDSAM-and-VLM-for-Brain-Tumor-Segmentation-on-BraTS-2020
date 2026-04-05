# =============================================================================
# SECTION 4 — PREPROCESSING VISUALIZATION
# =============================================================================
# Adapted for slice-level .h5 files:
#   Each .h5 has 'image' (240,240,4) and 'mask' (240,240,3) binary channels.
#   image channels: [0]=T1, [1]=T1ce, [2]=T2, [3]=FLAIR
#
# Shows sample images BEFORE and AFTER each preprocessing step:
#   • Z-score normalization  (per modality, non-zero voxels)
#   • Augmentation samples   (flip, rotation)
#   • Intensity histograms   (pre vs post normalization)
#
# Outputs saved to:  ./outputs/preprocessing/
# =============================================================================

import os, h5py, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUT_PREP = "./outputs/preprocessing"


def _load_slice_data(fpath):
    """Load image and integer label map from a slice-level .h5 file."""
    with h5py.File(fpath, "r") as f:
        image = f["image"][()].astype(np.float32)   # (H, W, 4)
        mask_3ch = f["mask"][()]                      # (H, W, 3) binary
    label_map = np.zeros(mask_3ch.shape[:2], dtype=np.int64)
    label_map[mask_3ch[:, :, 0] > 0] = 1
    label_map[mask_3ch[:, :, 1] > 0] = 2
    label_map[mask_3ch[:, :, 2] > 0] = 3
    return image, label_map


# ─────────────────────────────────────────────────────────────────
# 4a  Before vs After Normalization
# ─────────────────────────────────────────────────────────────────
def plot_before_after_normalization(volume_paths, n_volumes=2, save=True):
    """
    For each slice: show raw vs z-score normalized for all 4 modalities.
    """
    print("[Preprocessing] Plotting before/after normalization ...")
    # Pick slices with tumor content
    candidates = random.sample(volume_paths, min(100, len(volume_paths)))
    sampled = []
    for fpath in candidates:
        _, mask = _load_slice_data(fpath)
        if (mask > 0).sum() > 100:
            sampled.append(fpath)
        if len(sampled) >= n_volumes:
            break

    mod_labels = ["T1", "T1Gd", "T2", "FLAIR"]

    for vol_idx, fpath in enumerate(sampled):
        image, mask = _load_slice_data(fpath)

        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        vname = os.path.basename(fpath).replace(".h5", "")
        fig.suptitle(f"Before / After normalization — {vname}", fontsize=13, fontweight="bold")

        row_labels = ["Raw (original)", "Z-score normalized", "Difference (abs)"]
        for row in range(3):
            axes[row, 0].set_ylabel(row_labels[row], fontsize=9, fontweight="bold")

        for col, lbl in enumerate(mod_labels):
            axes[0, col].set_title(lbl, fontsize=11, fontweight="bold")

            sl_raw = image[:, :, col]    # (H, W)

            # Z-score on non-zero voxels
            nonzero = sl_raw[sl_raw > 0]
            if nonzero.size > 0:
                mu, sigma = nonzero.mean(), nonzero.std() + 1e-8
                sl_norm = (sl_raw - mu) / sigma
            else:
                sl_norm = sl_raw.copy()

            sl_diff = np.abs(sl_norm - (sl_raw / (sl_raw.max() + 1e-8)))
            raw_disp = (sl_raw - sl_raw.min()) / (sl_raw.max() - sl_raw.min() + 1e-8)

            axes[0, col].imshow(raw_disp, cmap="gray");  axes[0, col].axis("off")
            axes[1, col].imshow(sl_norm,  cmap="gray");  axes[1, col].axis("off")
            axes[2, col].imshow(sl_diff,  cmap="hot");   axes[2, col].axis("off")

        plt.tight_layout()
        if save:
            path = f"{OUTPUT_PREP}/4a_before_after_norm_vol{vol_idx}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[Preprocessing] Saved → {path}")
        plt.close()


# ─────────────────────────────────────────────────────────────────
# 4b  Intensity Histogram (pre vs post normalization)
# ─────────────────────────────────────────────────────────────────
def plot_intensity_histograms(volume_paths, n_volumes=30, save=True):
    """
    Overlay intensity histograms across N slices before and after z-score norm.
    One subplot per modality.
    """
    print("[Preprocessing] Plotting intensity histograms ...")
    sampled = random.sample(volume_paths, min(n_volumes, len(volume_paths)))
    mod_labels = ["T1", "T1Gd", "T2", "FLAIR"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Intensity histogram: raw vs z-score normalized", fontsize=13, fontweight="bold")

    for col, lbl in enumerate(mod_labels):
        axes[0, col].set_title(lbl, fontsize=11, fontweight="bold")
        raw_vals, norm_vals = [], []

        for fpath in sampled:
            with h5py.File(fpath, "r") as f:
                image = f["image"][()].astype(np.float32)   # (H, W, 4)
            vol = image[:, :, col].ravel()
            vol = vol[vol > 0]
            if vol.size == 0:
                continue
            norm_vol = (vol - vol.mean()) / (vol.std() + 1e-8)
            raw_vals.append(vol)
            norm_vals.append(norm_vol)

        if len(raw_vals) == 0:
            continue

        raw_all  = np.concatenate(raw_vals)
        norm_all = np.concatenate(norm_vals)

        # Clip outliers for display
        r_lo, r_hi = np.percentile(raw_all,  [1, 99])
        n_lo, n_hi = np.percentile(norm_all, [1, 99])

        axes[0, col].hist(raw_all.clip(r_lo, r_hi),  bins=80, color="#378ADD", alpha=0.7, density=True)
        axes[0, col].set_ylabel("Density" if col == 0 else "")
        axes[0, col].set_xlabel("Intensity")

        axes[1, col].hist(norm_all.clip(n_lo, n_hi), bins=80, color="#1D9E75", alpha=0.7, density=True)
        axes[1, col].set_ylabel("Density" if col == 0 else "")
        axes[1, col].set_xlabel("Normalized intensity")

    axes[0, 0].set_ylabel("Raw", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Z-scored", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save:
        path = f"{OUTPUT_PREP}/4b_intensity_histograms.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Preprocessing] Saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 4c  Augmentation Samples
# ─────────────────────────────────────────────────────────────────
def plot_augmentation_samples(volume_paths, n_samples=3, save=True):
    """
    Display original slice vs augmented variants side by side.
    Augmentations: horizontal flip, 90° rotation, combined.
    """
    print("[Preprocessing] Plotting augmentation samples ...")
    # Pick slices with tumor
    candidates = random.sample(volume_paths, min(100, len(volume_paths)))
    sampled = []
    for fpath in candidates:
        _, mask = _load_slice_data(fpath)
        if (mask > 0).sum() > 100:
            sampled.append(fpath)
        if len(sampled) >= n_samples:
            break

    n_samples = len(sampled)
    if n_samples == 0:
        print("[Preprocessing] No tumor slices found — skipping augmentation plot.")
        return

    fig, axes = plt.subplots(n_samples, 4, figsize=(15, n_samples * 3.5))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Augmentation samples — FLAIR channel", fontsize=13, fontweight="bold")

    col_titles = ["Original", "Horizontal flip", "90° rotation", "Flip + rotate"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row, fpath in enumerate(sampled):
        image, _ = _load_slice_data(fpath)
        flair = image[:, :, 3].astype(np.float32)   # FLAIR channel
        sl = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)

        augs = [
            sl,
            np.flip(sl, axis=1),
            np.rot90(sl, k=1),
            np.rot90(np.flip(sl, axis=1), k=1)
        ]
        for col, aug in enumerate(augs):
            axes[row, col].imshow(aug, cmap="gray")
            axes[row, col].axis("off")

    plt.tight_layout()
    if save:
        path = f"{OUTPUT_PREP}/4c_augmentation_samples.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Preprocessing] Saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 4d  Run all preprocessing visualizations
# ─────────────────────────────────────────────────────────────────
def run_preprocessing_viz(volume_paths):
    print("\n" + "="*60)
    print("SECTION 4 — PREPROCESSING VISUALIZATION")
    print("="*60)
    plot_before_after_normalization(volume_paths)
    plot_intensity_histograms(volume_paths)
    plot_augmentation_samples(volume_paths)
    print("[Preprocessing] All figures saved to ./outputs/preprocessing/")

if __name__ == "__main__":
    from section1_config import DATA_DIR
    from section2_dataset import build_splits
    print(f"--- Running Preprocessing Viz Test ---")
    train_paths, _, _ = build_splits(DATA_DIR)
    run_preprocessing_viz(train_paths)
    print("--- Preprocessing Viz Module Completed Successfully ---")