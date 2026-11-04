"""
Independent Model Evaluation on Held-Out Test Set.
Computes real test accuracy, precision, recall, F1, confusion matrix, and ROC-AUC.
Generates publication-quality metric plots without hardcoding or fabricating results.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, get_device, save_json
from src.dataset import (
    CLASSES,
    ID_TO_CLASS,
    BrainTumorDataset,
    load_split_from_csv,
    get_transforms
)
from models.hybrid_model import HybridBrainTumorModel

logger = logging.getLogger("brain_tumor_project.evaluation")


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir_metrics: str = "results/metrics",
    output_dir_figures: str = "results/figures"
) -> Dict[str, Any]:
    """
    Evaluates model on test_loader and saves verifiable metrics and figures.
    """
    Path(output_dir_metrics).mkdir(parents=True, exist_ok=True)
    Path(output_dir_figures).mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    logger.info("Running inference on held-out test dataset...")
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())
            all_probs.extend(probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # 1. Overall Metrics
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # 2. Per-Class Classification Report
    class_names = [ID_TO_CLASS[i] for i in range(len(CLASSES))]
    cls_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

    # 4. ROC-AUC (One-vs-Rest)
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
        if len(np.unique(y_true)) > 1 and y_probs.shape[1] == len(CLASSES):
            roc_auc_macro = float(roc_auc_score(y_true_bin, y_probs, multi_class="ovr", average="macro"))
        else:
            roc_auc_macro = None
    except Exception as e:
        logger.warning(f"Could not compute multi-class ROC-AUC: {e}")
        roc_auc_macro = None

    metrics_result = {
        "test_samples": int(len(y_true)),
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "roc_auc_macro": roc_auc_macro,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist()
    }

    # Save metrics JSON
    metrics_path = Path(output_dir_metrics) / "test_metrics.json"
    save_json(metrics_result, str(metrics_path))

    # --- Plot 1: Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(f"Test Confusion Matrix (Acc: {acc*100:.2f}%)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Diagnostic Class", fontsize=12)
    plt.ylabel("Ground Truth Class", fontsize=12)
    plt.tight_layout()
    cm_path = Path(output_dir_figures) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # --- Plot 2: Per-Class Metrics Bar Chart ---
    per_class_p = [cls_report[c]["precision"] for c in class_names]
    per_class_r = [cls_report[c]["recall"] for c in class_names]
    per_class_f1 = [cls_report[c]["f1-score"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, per_class_p, width, label="Precision", color="#3498db")
    plt.bar(x, per_class_r, width, label="Recall", color="#2ecc71")
    plt.bar(x + width, per_class_f1, width, label="F1-Score", color="#e74c3c")
    plt.xlabel("Brain Tumor Category", fontsize=12)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.title("Per-Class Evaluation Metrics on Held-Out Test Set", fontsize=14, fontweight="bold")
    plt.xticks(x, class_names, fontsize=11)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()
    bar_path = Path(output_dir_figures) / "per_class_metrics.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # --- Plot 3: Multi-Class ROC Curves ---
    if roc_auc_macro is not None:
        plt.figure(figsize=(9, 7))
        colors = ["#e74c3c", "#3498db", "#9b59b6", "#2ecc71"]
        for i, color in zip(range(len(CLASSES)), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            class_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f"{class_names[i]} (AUC = {class_auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(f"Multi-Class ROC Curves (Macro AUC = {roc_auc_macro:.3f})", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        roc_path = Path(output_dir_figures) / "roc_curves.png"
        plt.savefig(roc_path, dpi=300)
        plt.close()

    logger.info(
        f"\n================ TEST EVALUATION SUMMARY ================\n"
        f"Test Samples: {len(y_true)}\n"
        f"Accuracy:     {acc * 100:.2f}%\n"
        f"Macro F1:     {f1_macro:.4f}\n"
        f"Weighted F1:  {f1_weighted:.4f}\n"
        f"Macro ROC-AUC: {f'{roc_auc_macro:.4f}' if roc_auc_macro else 'N/A'}\n"
        f"Figures saved in: {output_dir_figures}\n"
        f"=========================================================="
    )
    return metrics_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Hybrid Brain Tumor Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(args.device if args.device != "auto" else config["training"].get("device", "auto"))

    checkpoint_path = args.checkpoint or str(Path(config["training"]["checkpoint_dir"]) / config["training"]["checkpoint_filename"])
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train model first via python training/train.py")

    test_csv = args.test_csv or str(Path(config["dataset"]["processed_dir"]) / "test_split.csv")
    if not Path(test_csv).is_file():
        raise FileNotFoundError(f"Test split CSV not found at: {test_csv}")

    test_samples = load_split_from_csv(test_csv)
    img_size = config["dataset"]["image_size"]
    test_transform = get_transforms(image_size=img_size, is_training=False)
    test_dataset = BrainTumorDataset(test_samples, transform=test_transform)

    batch_size = config["evaluation"].get("test_batch_size", 4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model without downloading base weights since checkpoint contains all weights
    eval_config = dict(config)
    eval_config["model"] = dict(config.get("model", {}))
    eval_config["model"]["cnn_pretrained"] = False
    eval_config["model"]["vit_pretrained"] = False
    model = HybridBrainTumorModel.from_config(eval_config)
    logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Run evaluation
    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir_metrics=config["evaluation"].get("metrics_dir", "results/metrics"),
        output_dir_figures=config["evaluation"].get("figures_dir", "results/figures")
    )


if __name__ == "__main__":
    main()
