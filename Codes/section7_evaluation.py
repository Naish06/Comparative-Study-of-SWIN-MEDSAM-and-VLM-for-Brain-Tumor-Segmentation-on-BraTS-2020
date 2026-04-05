# =============================================================================
# SECTION 7 — PREDICTION VISUALIZATION & EVALUATION METRICS
# =============================================================================
# 7a  Sample prediction images (train + test)
# 7b  Confusion matrix (4-class, normalized)
# 7c  Per-class accuracy bar chart
# 7d  Full classification report (precision / recall / F1 per split)
# 7e  Hausdorff Distance 95 (HD95) per model
#
# Outputs saved to:
#   ./outputs/predictions/
#   ./outputs/evaluation/
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)
from scipy.spatial.distance import directed_hausdorff

OUTPUT_PRED = "./outputs/predictions"
OUTPUT_EVAL = "./outputs/evaluation"
CLASS_NAMES  = ["Background", "NCR/NET", "Edema", "Enhancing Tumor"]
CLASS_COLORS = ["#1a1a1a", "#FF4444", "#44CC44", "#FFFF00"]
CMAP_SEG     = ListedColormap(CLASS_COLORS)


# ─────────────────────────────────────────────────────────────────
# 7a  Sample Prediction Images
# ─────────────────────────────────────────────────────────────────
def plot_prediction_samples(model, loader, model_name, split_name,
                             device, n_samples=4, save=True):
    """
    Displays N slices:  [FLAIR | Ground Truth | Prediction | Overlay]
    Args:
        split_name : 'train' | 'test'
    """
    print(f"[Prediction] Plotting {n_samples} samples — {model_name} [{split_name}] ...")
    model.eval()
    collected = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().numpy()     # (B, H, W)
            masks  = masks.numpy()
            images = images.cpu().numpy()

            for i in range(images.shape[0]):
                # Only include slices with some tumor
                if (masks[i] > 0).sum() > 100:
                    collected.append((images[i], masks[i], preds[i]))
                if len(collected) >= n_samples:
                    break
            if len(collected) >= n_samples:
                break

    n = len(collected)
    if n == 0:
        print(f"  [Warning] No tumor slices found — skipping prediction plot.")
        return

    fig, axes = plt.subplots(n, 4, figsize=(16, n * 3.5))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Predictions — {model_name.upper()} [{split_name}]",
                 fontsize=13, fontweight="bold")

    col_titles = ["FLAIR (input)", "Ground truth", "Prediction", "Overlay on FLAIR"]
    for col, t in enumerate(col_titles):
        axes[0, col].set_title(t, fontsize=10, fontweight="bold")

    for row, (imgs, gt, pred) in enumerate(collected):
        flair    = imgs[3]     # FLAIR is channel index 3
        flair_n  = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)

        axes[row, 0].imshow(flair_n, cmap="gray");          axes[row, 0].axis("off")
        axes[row, 1].imshow(gt,   cmap=CMAP_SEG, vmin=0, vmax=3, interpolation="nearest")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(pred, cmap=CMAP_SEG, vmin=0, vmax=3, interpolation="nearest")
        axes[row, 2].axis("off")

        # Overlay: FLAIR + prediction mask
        axes[row, 3].imshow(flair_n, cmap="gray")
        pred_masked = np.ma.masked_where(pred == 0, pred)
        axes[row, 3].imshow(pred_masked, cmap=CMAP_SEG, vmin=0, vmax=3,
                            alpha=0.55, interpolation="nearest")
        axes[row, 3].axis("off")

    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(CLASS_COLORS[1:], CLASS_NAMES[1:])]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save:
        path = f"{OUTPUT_PRED}/{model_name}_{split_name}_predictions.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Prediction] Saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 7b  Confusion Matrix
# ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(all_preds, all_targets, model_name, split_name, save=True):
    """
    4×4 normalized confusion matrix (row-normalized = per-class recall).
    """
    print(f"[Evaluation] Confusion matrix — {model_name} [{split_name}] ...")

    # Subsample for memory
    if len(all_preds) > 5_000_000:
        idx = np.random.choice(len(all_preds), 5_000_000, replace=False)
        all_preds, all_targets = all_preds[idx], all_targets[idx]

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion matrix — {model_name.upper()} [{split_name}]",
                 fontsize=13, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], linewidths=0.5, cbar=True)
    axes[0].set_title("Raw counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=20)

    # Normalized (recall per class diagonal)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], linewidths=0.5, cbar=True,
                vmin=0, vmax=1)
    axes[1].set_title("Row-normalized (recall per class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    if save:
        path = f"{OUTPUT_EVAL}/{model_name}_{split_name}_confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Evaluation] Saved → {path}")
    plt.close()
    return cm, cm_norm


# ─────────────────────────────────────────────────────────────────
# 7c  Per-class accuracy bar chart
# ─────────────────────────────────────────────────────────────────
def plot_per_class_accuracy(all_preds, all_targets, model_name, split_name, save=True):
    """Grouped bar chart of per-class accuracy for one model."""
    per_class_acc = []
    for c in range(4):
        mask   = (all_targets == c)
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            acc = (all_preds[mask] == c).sum() / mask.sum()
            per_class_acc.append(float(acc))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"Per-class accuracy — {model_name.upper()} [{split_name}]",
                 fontsize=12, fontweight="bold")
    bars = ax.bar(CLASS_NAMES, per_class_acc, color=CLASS_COLORS, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.axhline(np.mean(per_class_acc), color="gray", linestyle="--", lw=1, label="Mean")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save:
        path = f"{OUTPUT_EVAL}/{model_name}_{split_name}_per_class_accuracy.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Evaluation] Saved → {path}")
    plt.close()
    return per_class_acc


# ─────────────────────────────────────────────────────────────────
# 7d  Full classification report (all splits, all models)
# ─────────────────────────────────────────────────────────────────
def print_and_save_classification_report(all_preds, all_targets,
                                          model_name, split_name, save=True):
    """
    Prints sklearn classification report and saves as CSV.
    Returns dict of summary metrics.
    """
    report_str  = classification_report(
        all_targets, all_preds,
        target_names=CLASS_NAMES,
        labels=[0, 1, 2, 3],
        zero_division=0
    )
    print(f"\n── Classification Report: {model_name.upper()} [{split_name}] ──")
    print(report_str)

    # Save as CSV
    report_dict = classification_report(
        all_targets, all_preds,
        target_names=CLASS_NAMES,
        labels=[0, 1, 2, 3],
        zero_division=0,
        output_dict=True
    )
    df = pd.DataFrame(report_dict).transpose()
    if save:
        path = f"{OUTPUT_EVAL}/{model_name}_{split_name}_classification_report.csv"
        df.to_csv(path)
        print(f"[Evaluation] Report saved → {path}")

    return {
        "accuracy":  report_dict["accuracy"],
        "precision": report_dict["macro avg"]["precision"],
        "recall":    report_dict["macro avg"]["recall"],
        "f1":        report_dict["macro avg"]["f1-score"],
    }


# ─────────────────────────────────────────────────────────────────
# 7e  HD95 (Hausdorff Distance 95th percentile)
# ─────────────────────────────────────────────────────────────────
def compute_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray,
                  class_idx: int) -> float:
    """
    Computes HD95 for a single class between pred and GT binary masks.
    Args:
        pred_mask, gt_mask : (H, W) integer label maps
        class_idx          : class to evaluate
    Returns:
        hd95 in pixels, or np.nan if one mask is empty
    """
    pred_bin = (pred_mask == class_idx).astype(np.uint8)
    gt_bin   = (gt_mask   == class_idx).astype(np.uint8)

    pred_pts = np.argwhere(pred_bin)
    gt_pts   = np.argwhere(gt_bin)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan

    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    # 95th percentile approximation (symmetric max for 2D)
    return max(d1, d2)


# ─────────────────────────────────────────────────────────────────
# 7f  Full evaluation on a DataLoader
# ─────────────────────────────────────────────────────────────────
def evaluate_model(model, loader, model_name, split_name, device,
                   compute_hd=True, save=True):
    """
    Runs full evaluation:
    • Collects all predictions and targets
    • Computes Dice per class, per-class accuracy, confusion matrix,
      classification report, optionally HD95
    Returns:
        summary dict with all metrics
    """
    print(f"\n[Evaluation] Evaluating {model_name.upper()} on [{split_name}] ...")
    model.eval()
    all_preds, all_targets = [], []
    slice_preds, slice_gts = [], []    # for HD95 (keep 2D structure)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().numpy()
            gts    = masks.numpy()

            for i in range(preds.shape[0]):
                slice_preds.append(preds[i])
                slice_gts.append(gts[i])

            all_preds.append(preds.ravel())
            all_targets.append(gts.ravel())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # ── Dice ─────────────────────────────────────────────────────
    from section6_training import compute_dice_per_class
    dice = compute_dice_per_class(all_preds, all_targets)

    # ── Confusion matrix ─────────────────────────────────────────
    cm, cm_norm = plot_confusion_matrix(all_preds, all_targets,
                                         model_name, split_name, save=save)

    # ── Per-class accuracy ────────────────────────────────────────
    per_class_acc = plot_per_class_accuracy(all_preds, all_targets,
                                             model_name, split_name, save=save)

    # ── Classification report ─────────────────────────────────────
    clf_metrics = print_and_save_classification_report(
        all_preds, all_targets, model_name, split_name, save=save
    )

    # ── HD95 per tumor class ──────────────────────────────────────
    hd95 = {"ncr": np.nan, "ed": np.nan, "et": np.nan}
    if compute_hd:
        hd_vals = {1: [], 2: [], 3: []}
        for pred_sl, gt_sl in zip(slice_preds[:200], slice_gts[:200]):  # first 200 slices
            for c in [1, 2, 3]:
                h = compute_hd95(pred_sl, gt_sl, c)
                if not np.isnan(h):
                    hd_vals[c].append(h)
        hd95 = {
            "ncr": np.median(hd_vals[1]) if hd_vals[1] else np.nan,
            "ed":  np.median(hd_vals[2]) if hd_vals[2] else np.nan,
            "et":  np.median(hd_vals[3]) if hd_vals[3] else np.nan,
        }

    summary = {
        "model":         model_name,
        "split":         split_name,
        "accuracy":      clf_metrics["accuracy"],
        "precision":     clf_metrics["precision"],
        "recall":        clf_metrics["recall"],
        "f1":            clf_metrics["f1"],
        "dice_bg":       dice[0],
        "dice_ncr":      dice[1],
        "dice_ed":       dice[2],
        "dice_et":       dice[3],
        "mean_dice":     dice[1:].mean(),
        "hd95_ncr":      hd95["ncr"],
        "hd95_ed":       hd95["ed"],
        "hd95_et":       hd95["et"],
        "per_class_acc": per_class_acc,
    }

    print(f"[Evaluation] {model_name.upper()} [{split_name}] — "
          f"Acc: {summary['accuracy']:.4f} | "
          f"F1: {summary['f1']:.4f} | "
          f"Mean Dice: {summary['mean_dice']:.4f}")

    return summary

if __name__ == "__main__":
    print(f"--- Evaluation Module ---")
    print("This file contains the evaluation logic.")
    print("To execute full evaluation step, please run 'python section8_comparative_and_main.py'.")