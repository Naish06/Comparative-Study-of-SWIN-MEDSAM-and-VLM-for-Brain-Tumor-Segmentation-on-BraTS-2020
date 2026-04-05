# =============================================================================
# SECTION 6 — TRAINING LOOP WITH MONITORING
# =============================================================================
# Per-epoch logging: loss, accuracy, precision, recall, F1, Dice per class
# Saves:
#   • Best checkpoint per model
#   • Training history CSV
#   • Training curve plots  → ./outputs/training/
# =============================================================================

import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

OUTPUT_TRAIN = "./outputs/training"
CKPT_DIR     = "./checkpoints"


# ─────────────────────────────────────────────────────────────────
# 6a  Dice coefficient per class
# ─────────────────────────────────────────────────────────────────
def compute_dice_per_class(preds: np.ndarray, targets: np.ndarray,
                            n_classes=4, smooth=1e-6) -> np.ndarray:
    """
    Args:
        preds   : (N,) flattened predicted labels
        targets : (N,) flattened true labels
    Returns:
        dice per class shape (n_classes,)
    """
    dice = np.zeros(n_classes)
    for c in range(n_classes):
        pred_c   = (preds   == c).astype(np.float32)
        target_c = (targets == c).astype(np.float32)
        inter    = (pred_c * target_c).sum()
        union    = pred_c.sum() + target_c.sum()
        dice[c]  = (2 * inter + smooth) / (union + smooth)
    return dice


# ─────────────────────────────────────────────────────────────────
# 6b  Single epoch train / validate
# ─────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, phase="train"):
    """
    Runs one epoch of train or validation.
    Returns: dict with all metrics for this epoch.
    """
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)                           # (B, C, H, W)
            loss   = criterion(logits, masks)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

            preds = logits.argmax(dim=1).cpu().numpy().ravel()   # flatten
            tgts  = masks.cpu().numpy().ravel()
            
            # Subsample by 10x to prevent OOM errors (accumulating the full dataset takes > 20GB RAM!)
            all_preds.append(preds[::10])
            all_targets.append(tgts[::10])

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Subsample for speed (10M voxels max)
    if len(all_preds) > 10_000_000:
        idx = np.random.choice(len(all_preds), 10_000_000, replace=False)
        all_preds, all_targets = all_preds[idx], all_targets[idx]

    acc       = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="macro",
                                 zero_division=0, labels=[0,1,2,3])
    recall    = recall_score(all_targets, all_preds, average="macro",
                              zero_division=0, labels=[0,1,2,3])
    f1        = f1_score(all_targets, all_preds, average="macro",
                          zero_division=0, labels=[0,1,2,3])
    dice      = compute_dice_per_class(all_preds, all_targets)
    mean_dice = dice[1:].mean()    # exclude background

    metrics = {
        "loss":          total_loss / len(loader),
        "accuracy":      acc,
        "precision":     precision,
        "recall":        recall,
        "f1":            f1,
        "mean_dice":     mean_dice,
        "dice_bg":       dice[0],
        "dice_ncr":      dice[1],
        "dice_ed":       dice[2],
        "dice_et":       dice[3],
    }
    return metrics


# ─────────────────────────────────────────────────────────────────
# 6c  Full training loop
# ─────────────────────────────────────────────────────────────────
def train_model(model, model_name, train_loader, val_loader,
                device, num_epochs=50, lr=1e-4, weight_decay=1e-5):
    """
    Full training loop with:
    • Dice + CE loss
    • Adam optimizer + CosineAnnealing scheduler
    • Best-val-Dice checkpoint saving
    • Per-epoch CSV logging

    Returns:
        history : pd.DataFrame with all epoch metrics (train + val)
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    print(f"{'='*60}")

    from section5_models import CombinedLoss
    criterion = CombinedLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_dice = -np.inf
    history = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, "train")
        val_metrics   = run_epoch(model, val_loader,   criterion, None,      device, "val")

        scheduler.step()
        elapsed = time.time() - t0

        # ── Assemble log row ──────────────────────────────────────
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "time_s": elapsed}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        history.append(row)

        # ── Console log ───────────────────────────────────────────
        print(f"[{model_name}] Ep {epoch:03d}/{num_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Dice: {val_metrics['mean_dice']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"{elapsed:.1f}s")

        # ── Save best checkpoint ──────────────────────────────────
        if val_metrics["mean_dice"] > best_val_dice:
            best_val_dice = val_metrics["mean_dice"]
            ckpt_path = f"{CKPT_DIR}/{model_name}_best.pth"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_dice":    best_val_dice,
                "optimizer":   optimizer.state_dict(),
            }, ckpt_path)
            print(f"  [Checkpoint] Best model saved → {ckpt_path}  (Dice={best_val_dice:.4f})")

    history_df = pd.DataFrame(history)
    csv_path   = f"{OUTPUT_TRAIN}/{model_name}_history.csv"
    history_df.to_csv(csv_path, index=False)
    print(f"\n[Training] History saved → {csv_path}")
    return history_df


# ─────────────────────────────────────────────────────────────────
# 6d  Plot training curves
# ─────────────────────────────────────────────────────────────────
def plot_training_curves(history_df, model_name, save=True):
    """
    Plots 6-panel training dashboard:
    Loss | Accuracy | Precision | Recall | F1 | Per-class Dice
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training curves — {model_name.upper()}", fontsize=14, fontweight="bold")

    epochs = history_df["epoch"].values

    panels = [
        ("loss",      "Loss",           "Loss"),
        ("accuracy",  "Accuracy",       "Accuracy"),
        ("precision", "Precision",      "Precision"),
        ("recall",    "Recall",         "Recall"),
        ("f1",        "F1 (macro)",     "F1"),
        (None,        "Dice per class", "Dice"),
    ]

    for ax, (key, title, ylabel) in zip(axes.ravel(), panels):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

        if key is not None:
            ax.plot(epochs, history_df[f"train_{key}"], label="Train", color="#378ADD", lw=1.5)
            ax.plot(epochs, history_df[f"val_{key}"],   label="Val",   color="#D85A30", lw=1.5)
            ax.legend(fontsize=9)
        else:
            # Per-class dice (val only)
            colors = ["#888780", "#FF4444", "#44CC44", "#FFFF00"]
            labels = ["BG", "NCR/NET", "Edema", "Enhancing Tumor"]
            for c, (col, lbl) in enumerate(zip(colors, labels)):
                col_key = f"val_dice_{'bg' if c==0 else ['ncr','ed','et'][c-1]}"
                if col_key in history_df:
                    ax.plot(epochs, history_df[col_key], label=lbl, color=col, lw=1.5)
            ax.legend(fontsize=8)

    plt.tight_layout()
    if save:
        path = f"{OUTPUT_TRAIN}/{model_name}_training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Training] Curves saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 6e  Overlay training curves for all 3 models (comparison)
# ─────────────────────────────────────────────────────────────────
def plot_model_comparison_curves(histories: dict, save=True):
    """
    Args:
        histories : dict {model_name: DataFrame}
    """
    import matplotlib.pyplot as plt

    model_colors = {
        "swin_unet": "#378ADD",
        "medsam":    "#D85A30",
        "vlm":       "#7F77DD",
    }
    metrics = ["loss", "accuracy", "f1", "mean_dice"]
    titles  = ["Loss", "Accuracy", "F1 (macro)", "Mean Dice (no BG)"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Model comparison — validation metrics", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes, metrics, titles):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        for mname, df in histories.items():
            col   = f"val_{metric}"
            color = model_colors.get(mname, "gray")
            ax.plot(df["epoch"], df[col], label=mname.upper(), color=color, lw=2)
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save:
        path = f"{OUTPUT_TRAIN}/comparison_val_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Training] Comparison curves saved → {path}")
    plt.close()

if __name__ == "__main__":
    print(f"--- Training Module ---")
    print("This file contains the training loop and metrics.")
    print("To execute full training, please run 'python section8_comparative_and_main.py'.")