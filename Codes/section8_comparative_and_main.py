# =============================================================================
# SECTION 8 — COMPARATIVE ANALYSIS & FINAL SUMMARY TABLE
# =============================================================================
# Aggregates results from all 3 models into:
#   • Side-by-side bar charts (Dice, F1, Accuracy per class)
#   • Final comparison table (CSV + styled PNG)
#   • Per-class accuracy grouped bar chart (all models together)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_EVAL = "./outputs/evaluation"


# ─────────────────────────────────────────────────────────────────
# 8a  Grouped per-class accuracy bar chart (all models)
# ─────────────────────────────────────────────────────────────────
def plot_per_class_accuracy_comparison(summaries: list, save=True):
    """
    Args:
        summaries : list of dicts from evaluate_model() for each model
    """
    print("[Comparative] Plotting per-class accuracy comparison ...")
    model_names = [s["model"].upper() for s in summaries]
    class_names = ["Background", "NCR/NET", "Edema", "Enhancing Tumor"]
    class_colors = ["#888780", "#FF4444", "#44CC44", "#FFFF00"]

    n_models  = len(summaries)
    n_classes = 4
    x         = np.arange(n_classes)
    width     = 0.22
    offsets   = np.linspace(-width, width, n_models)

    model_style = ["#378ADD", "#D85A30", "#7F77DD"]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_title("Per-class accuracy — all models (test split)",
                 fontsize=13, fontweight="bold")

    for idx, (s, offset, mc) in enumerate(zip(summaries, offsets, model_style)):
        accs = s["per_class_acc"]
        bars = ax.bar(x + offset, accs, width=width * 0.9,
                      label=s["model"].upper(), color=mc, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        path = f"{OUTPUT_EVAL}/8a_per_class_accuracy_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Comparative] Saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 8b  Dice score comparison bar chart
# ─────────────────────────────────────────────────────────────────
def plot_dice_comparison(summaries: list, save=True):
    """Bar chart of Dice per class for each model."""
    print("[Comparative] Plotting Dice comparison ...")
    metrics_keys = ["dice_ncr", "dice_ed", "dice_et", "mean_dice"]
    labels       = ["NCR/NET", "Edema", "Enhancing Tumor", "Mean (no BG)"]
    model_colors = ["#378ADD", "#D85A30", "#7F77DD"]

    x      = np.arange(len(metrics_keys))
    width  = 0.22
    n      = len(summaries)
    offsets = np.linspace(-width, width, n)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Dice score comparison — test split", fontsize=13, fontweight="bold")

    for s, offset, mc in zip(summaries, offsets, model_colors):
        vals = [s[k] for k in metrics_keys]
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      label=s["model"].upper(), color=mc, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Dice")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        path = f"{OUTPUT_EVAL}/8b_dice_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Comparative] Saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 8c  Final comparison table (CSV + heatmap)
# ─────────────────────────────────────────────────────────────────
def build_final_comparison_table(summaries: list, inference_times: dict = None, save=True):
    """
    Args:
        summaries       : list of eval summary dicts
        inference_times : dict {model_name: ms_per_slice}
    """
    print("[Comparative] Building final comparison table ...")
    rows = []
    for s in summaries:
        row = {
            "Model":          s["model"].upper(),
            "Accuracy":       round(s["accuracy"],  4),
            "Precision":      round(s["precision"], 4),
            "Recall":         round(s["recall"],    4),
            "F1 (macro)":     round(s["f1"],        4),
            "Dice NCR/NET":   round(s["dice_ncr"],  4),
            "Dice Edema":     round(s["dice_ed"],   4),
            "Dice ET":        round(s["dice_et"],   4),
            "Mean Dice":      round(s["mean_dice"], 4),
            "HD95 NCR (px)":  round(s["hd95_ncr"], 2) if not np.isnan(s.get("hd95_ncr", np.nan)) else "N/A",
            "HD95 ED (px)":   round(s["hd95_ed"],  2) if not np.isnan(s.get("hd95_ed",  np.nan)) else "N/A",
            "HD95 ET (px)":   round(s["hd95_et"],  2) if not np.isnan(s.get("hd95_et",  np.nan)) else "N/A",
        }
        if inference_times:
            row["Inference (ms/slice)"] = inference_times.get(s["model"], "N/A")
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    print("\n" + "="*70)
    print("FINAL COMPARISON TABLE")
    print("="*70)
    print(df.to_string())

    if save:
        csv_path = f"{OUTPUT_EVAL}/8c_final_comparison_table.csv"
        df.to_csv(csv_path)
        print(f"\n[Comparative] Table saved → {csv_path}")

        # Heatmap (numeric columns only)
        numeric_df = df.select_dtypes(include=[float, int])
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.set_title("Final model comparison — test split", fontsize=13, fontweight="bold")
        sns.heatmap(numeric_df, annot=True, fmt=".4f", cmap="YlGnBu",
                    linewidths=0.5, ax=ax, cbar=True, vmin=0, vmax=1)
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        hm_path = f"{OUTPUT_EVAL}/8c_final_comparison_heatmap.png"
        plt.savefig(hm_path, dpi=150, bbox_inches="tight")
        print(f"[Comparative] Heatmap saved → {hm_path}")
        plt.close()

    return df


# =============================================================================
# SECTION 9 — MAIN RUNNER
# =============================================================================
# Orchestrates all sections end-to-end:
#   1. Config + paths
#   2. Dataset loading + split
#   3. EDA
#   4. Preprocessing viz
#   5. Train all 3 models
#   6. Evaluate on all 3 splits
#   7. Comparative analysis
# =============================================================================

def main():
    import torch
    import time
    from section1_config  import (DATA_DIR, DEVICE, IMG_SIZE, BATCH_SIZE,
                                   NUM_EPOCHS, LR, WEIGHT_DECAY)
    from section2_dataset  import get_dataloaders
    from section3_eda      import run_eda
    from section4_preprocessing_viz import run_preprocessing_viz
    from section5_models   import get_model
    from section6_training import train_model, plot_training_curves, plot_model_comparison_curves
    from section7_evaluation import evaluate_model, plot_prediction_samples

    print("\n" + "="*60)
    print("BRATS 2020 — FULL SEGMENTATION PIPELINE")
    print("="*60)

    # ── Load data ─────────────────────────────────────────────────
    (train_loader, val_loader, test_loader,
     train_paths, val_paths, test_paths) = get_dataloaders(
        DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE
    )

    all_paths = train_paths + val_paths + test_paths

    # ── Section 3: EDA ────────────────────────────────────────────
    run_eda(all_paths)

    # ── Section 4: Preprocessing viz ─────────────────────────────
    run_preprocessing_viz(train_paths)

    # ── Section 5-6-7: Train + Evaluate each model ───────────────
    MODEL_NAMES = ["swin_unet", "medsam", "vlm"]
    histories   = {}
    summaries   = []
    inf_times   = {}

    for model_name in MODEL_NAMES:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_name.upper()}")
        print(f"{'#'*60}")

        # Initialize
        model = get_model(model_name, num_classes=4, img_size=IMG_SIZE, device=DEVICE)

        # Train
        history = train_model(
            model, model_name, train_loader, val_loader,
            device=DEVICE, num_epochs=NUM_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY
        )
        histories[model_name] = history
        plot_training_curves(history, model_name)

        # Load best checkpoint for evaluation
        ckpt_path = f"./checkpoints/{model_name}_best.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            print(f"[Runner] Loaded best checkpoint from epoch {ckpt['epoch']}")

        # Measure inference time (ms per slice)
        model.eval()
        dummy = torch.randn(1, 4, IMG_SIZE, IMG_SIZE).to(DEVICE)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = model(dummy)
        inf_times[model_name] = round((time.time() - t0) / 50 * 1000, 2)

        # Prediction samples: train + test
        plot_prediction_samples(model, train_loader, model_name, "train", DEVICE, n_samples=4)
        plot_prediction_samples(model, test_loader,  model_name, "test",  DEVICE, n_samples=4)

        # Full evaluation on test split
        test_summary = evaluate_model(model, test_loader, model_name, "test", DEVICE)
        summaries.append(test_summary)

        # Also evaluate train + val (for classification report completeness)
        evaluate_model(model, train_loader, model_name, "train", DEVICE, compute_hd=False)
        evaluate_model(model, val_loader,   model_name, "val",   DEVICE, compute_hd=False)

    # ── Section 6: Model comparison curves ───────────────────────
    plot_model_comparison_curves(histories)

    # ── Section 8: Comparative analysis ──────────────────────────
    print("\n" + "="*60)
    print("SECTION 8 — COMPARATIVE ANALYSIS")
    print("="*60)
    plot_per_class_accuracy_comparison(summaries)
    plot_dice_comparison(summaries)
    build_final_comparison_table(summaries, inference_times=inf_times)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"All outputs in ./outputs/")
    print("="*60)


if __name__ == "__main__":
    main()