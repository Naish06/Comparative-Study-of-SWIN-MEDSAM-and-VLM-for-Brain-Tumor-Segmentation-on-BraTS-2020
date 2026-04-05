# =============================================================================
# SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
# Adapted for slice-level .h5 files:
#   Each .h5 has 'image' (240,240,4) and 'mask' (240,240,3) binary channels.
#
# Outputs saved to:  ./outputs/eda/
#   3a_class_distribution.png
#   3b_ground_truth_samples.png
#   3c_modality_grid.png
# =============================================================================

import os, h5py, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

OUTPUT_EDA   = "./outputs/eda"
CLASS_NAMES  = ["Background", "NCR/NET", "Edema", "Enhancing Tumor"]
CLASS_COLORS = ["#1a1a1a", "#FF4444", "#44CC44", "#FFFF00"]
CMAP_SEG     = ListedColormap(CLASS_COLORS)


def _load_slice_data(fpath):
    """Load image and integer label map from a slice-level .h5 file."""
    with h5py.File(fpath, "r") as f:
        image = f["image"][()].astype(np.float32)   # (H, W, 4)
        mask_3ch = f["mask"][()]                      # (H, W, 3) binary
    # Convert 3-channel binary → integer label map
    label_map = np.zeros(mask_3ch.shape[:2], dtype=np.int64)
    label_map[mask_3ch[:, :, 0] > 0] = 1   # NCR/NET
    label_map[mask_3ch[:, :, 1] > 0] = 2   # Edema
    label_map[mask_3ch[:, :, 2] > 0] = 3   # ET
    return image, label_map


# ─────────────────────────────────────────────────────────────────
# 3a  Class Distribution Bar Chart
# ─────────────────────────────────────────────────────────────────
def plot_class_distribution(volume_paths, n_sample=500, save=True):
    """
    Count voxel-level label frequencies across a sample of slices.
    Plots a grouped bar chart (log scale) showing severe class imbalance.
    """
    print("[EDA] Computing class distribution ...")
    counts = np.zeros(4, dtype=np.int64)       # {0,1,2,3}

    sampled = random.sample(volume_paths, min(n_sample, len(volume_paths)))
    for fpath in sampled:
        _, mask = _load_slice_data(fpath)
        for c in range(4):
            counts[c] += (mask == c).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Class distribution — BraTS 2020 (voxel level)", fontsize=14, fontweight="bold")

    # Linear scale (background dominates)
    ax = axes[0]
    bars = ax.bar(CLASS_NAMES, counts, color=CLASS_COLORS, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Voxel count")
    ax.set_title("Linear scale")
    ax.tick_params(axis="x", rotation=15)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"{cnt:,}", ha="center", va="bottom", fontsize=9)

    # Log scale (shows tumor classes clearly)
    ax2 = axes[1]
    ax2.bar(CLASS_NAMES, counts, color=CLASS_COLORS, edgecolor="white", linewidth=0.5, log=True)
    ax2.set_ylabel("Voxel count (log)")
    ax2.set_title("Log scale — tumor classes visible")
    ax2.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    if save:
        path = f"{OUTPUT_EDA}/3a_class_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[EDA] Saved → {path}")
    plt.close()
    return counts


# ─────────────────────────────────────────────────────────────────
# 3b  Ground Truth Samples Display
# ─────────────────────────────────────────────────────────────────
def plot_ground_truth_samples(volume_paths, n_samples=4, save=True):
    """
    Displays N sample slices: FLAIR | GT mask | color overlay.
    Only picks slices that contain tumor (mask > 0).
    """
    print("[EDA] Plotting ground truth samples ...")
    # Filter for slices with tumor
    candidates = random.sample(volume_paths, min(200, len(volume_paths)))
    tumor_slices = []
    for fpath in candidates:
        _, mask = _load_slice_data(fpath)
        if (mask > 0).sum() > 100:
            tumor_slices.append(fpath)
        if len(tumor_slices) >= n_samples:
            break

    if len(tumor_slices) == 0:
        print("[EDA] No tumor slices found — skipping GT samples plot.")
        return

    n_samples = len(tumor_slices)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 3.5))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Ground truth: FLAIR | Annotation mask | Overlay", fontsize=13, fontweight="bold")

    col_titles = ["FLAIR (input)", "Segmentation mask", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, fpath in enumerate(tumor_slices):
        image, mask = _load_slice_data(fpath)
        # FLAIR is channel index 3 (last channel)
        flair = image[:, :, 3]
        flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)

        ax0, ax1, ax2 = axes[row, 0], axes[row, 1], axes[row, 2]

        ax0.imshow(flair_norm, cmap="gray")
        ax0.axis("off")
        vname = os.path.basename(fpath).replace(".h5", "")
        ax0.set_ylabel(f"{vname}", fontsize=7)

        ax1.imshow(mask, cmap=CMAP_SEG, vmin=0, vmax=3, interpolation="nearest")
        ax1.axis("off")

        ax2.imshow(flair_norm, cmap="gray")
        masked = np.ma.masked_where(mask == 0, mask)
        ax2.imshow(masked, cmap=CMAP_SEG, vmin=0, vmax=3, alpha=0.55, interpolation="nearest")
        ax2.axis("off")

    # Legend
    patches = [mpatches.Patch(color=c, label=n) for c, n in zip(CLASS_COLORS[1:], CLASS_NAMES[1:])]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save:
        path = f"{OUTPUT_EDA}/3b_ground_truth_samples.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[EDA] Saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 3c  Modality Grid (T1 / T1ce / T2 / FLAIR)
# ─────────────────────────────────────────────────────────────────
def plot_modality_grid(volume_paths, n_volumes=3, save=True):
    """
    Shows all 4 MRI modalities side-by-side for N slices.
    image channels: [0]=T1, [1]=T1ce, [2]=T2, [3]=FLAIR
    """
    print("[EDA] Plotting modality grid ...")
    # Pick slices with tumor
    candidates = random.sample(volume_paths, min(200, len(volume_paths)))
    sampled = []
    for fpath in candidates:
        _, mask = _load_slice_data(fpath)
        if (mask > 0).sum() > 100:
            sampled.append(fpath)
        if len(sampled) >= n_volumes:
            break

    if len(sampled) == 0:
        print("[EDA] No tumor slices found — skipping modality grid.")
        return

    n_volumes = len(sampled)
    mod_labels = ["T1 (native)", "T1Gd (contrast)", "T2", "FLAIR"]

    fig, axes = plt.subplots(n_volumes, 5, figsize=(17, n_volumes * 3.5))
    if n_volumes == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("MRI modality comparison — same axial slice", fontsize=13, fontweight="bold")

    for col, lbl in enumerate(mod_labels):
        axes[0, col].set_title(lbl, fontsize=10, fontweight="bold")
    axes[0, 4].set_title("GT mask", fontsize=10, fontweight="bold")

    for row, fpath in enumerate(sampled):
        image, mask = _load_slice_data(fpath)

        for col in range(4):
            sl = image[:, :, col]
            sl_norm = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
            axes[row, col].imshow(sl_norm, cmap="gray")
            axes[row, col].axis("off")

        axes[row, 4].imshow(mask, cmap=CMAP_SEG, vmin=0, vmax=3, interpolation="nearest")
        axes[row, 4].axis("off")

        vname = os.path.basename(fpath).replace(".h5", "")
        axes[row, 0].set_ylabel(f"{vname}", fontsize=7)

    patches = [mpatches.Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    if save:
        path = f"{OUTPUT_EDA}/3c_modality_grid.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[EDA] Saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 3d  Run all EDA
# ─────────────────────────────────────────────────────────────────
def run_eda(volume_paths):
    print("\n" + "="*60)
    print("SECTION 3 — EXPLORATORY DATA ANALYSIS")
    print("="*60)
    counts = plot_class_distribution(volume_paths)
    plot_ground_truth_samples(volume_paths)
    plot_modality_grid(volume_paths)
    print("[EDA] All EDA figures saved to ./outputs/eda/")
    return counts

if __name__ == "__main__":
    from section1_config import DATA_DIR
    from section2_dataset import build_splits
    print(f"--- Running EDA Test ---")
    train_paths, val_paths, test_paths = build_splits(DATA_DIR)
    all_paths = train_paths + val_paths + test_paths
    run_eda(all_paths)
    print("--- EDA Module Completed Successfully ---")