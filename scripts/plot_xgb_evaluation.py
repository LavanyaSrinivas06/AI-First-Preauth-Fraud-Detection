#!/usr/bin/env python3
"""
Generate evaluation plots for the tuned XGBoost model.

Inputs (must already exist):
- artifacts/xgb_model.pkl          (trained model from train_xgboost.py)
- artifacts/xgb_metrics.json       (metrics + optimal threshold)
- data/processed/train.csv
- data/processed/val.csv
- data/processed/test.csv

Outputs (all saved to artifacts/plots/):
- xgb_roc_val.png
- xgb_roc_test.png
- xgb_pr_val.png
- xgb_pr_test.png
- xgb_confusion_val.png
- xgb_confusion_test.png
- xgb_f1_vs_threshold_val.png
- xgb_score_distribution_val.png
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
)


# -------------------------------------------------------------------
# Paths and loading helpers
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
PLOT_DIR = ARTIFACT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = ARTIFACT_DIR / "xgb_model.pkl"
METRICS_PATH = ARTIFACT_DIR / "xgb_metrics.json"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]

    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"]

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_model_and_threshold():
    model = joblib.load(MODEL_PATH)

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    # Fallback if threshold key missing
    threshold = float(metrics.get("threshold", 0.5))
    return model, threshold


# -------------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------------

def plot_roc(model, X, y, split_name: str, out_path: Path):
    proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{split_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {split_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pr(model, X, y, split_name: str, out_path: Path):
    proba = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"{split_name} (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve – {split_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion(model, X, y, threshold: float, split_name: str, out_path: Path):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y, preds)

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0 (non-fraud)", "Pred 1 (fraud)"],
        yticklabels=["True 0 (non-fraud)", "True 1 (fraud)"],
    )
    plt.title(f"Confusion Matrix – {split_name} (t={threshold:.3f})")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_f1_vs_threshold(model, X, y, split_name: str, out_path: Path):
    proba = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = []

    for t in thresholds:
        preds = (proba >= t).astype(int)
        f1s.append(f1_score(y, preds, zero_division=0))

    f1s = np.array(f1s)
    best_idx = int(np.argmax(f1s))
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, f1s)
    plt.axvline(best_t, linestyle="--", label=f"Best t={best_t:.3f}, F1={best_f1:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title(f"F1 vs Threshold – {split_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_score_distribution(model, X, y, split_name: str, out_path: Path):
    proba = model.predict_proba(X)[:, 1]

    plt.figure(figsize=(6, 4))
    plt.hist(
        proba[y == 0],
        bins=50,
        alpha=0.7,
        label="Class 0 (non-fraud)",
    )
    plt.hist(
        proba[y == 1],
        bins=50,
        alpha=0.7,
        label="Class 1 (fraud)",
    )
    plt.xlabel("Predicted fraud probability")
    plt.ylabel("Count")
    plt.title(f"Score Distribution by Class – {split_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

def main():
    print("🔄 Loading data, model, and threshold...")
    (_, _), (X_val, y_val), (X_test, y_test) = load_data()
    model, threshold = load_model_and_threshold()
    print("✔ Loaded.")

    # ROC curves
    print("📈 Saving ROC curves...")
    plot_roc(model, X_val, y_val, "Validation", PLOT_DIR / "xgb_roc_val.png")
    plot_roc(model, X_test, y_test, "Test", PLOT_DIR / "xgb_roc_test.png")

    # Precision–Recall curves
    print("📈 Saving Precision–Recall curves...")
    plot_pr(model, X_val, y_val, "Validation", PLOT_DIR / "xgb_pr_val.png")
    plot_pr(model, X_test, y_test, "Test", PLOT_DIR / "xgb_pr_test.png")

    # Confusion matrices
    print("📊 Saving confusion matrix heatmaps...")
    plot_confusion(
        model, X_val, y_val, threshold, "Validation", PLOT_DIR / "xgb_confusion_val.png"
    )
    plot_confusion(
        model, X_test, y_test, threshold, "Test", PLOT_DIR / "xgb_confusion_test.png"
    )

    # F1 vs threshold (validation only)
    print("📊 Saving F1 vs threshold plot (validation)...")
    plot_f1_vs_threshold(
        model, X_val, y_val, "Validation", PLOT_DIR / "xgb_f1_vs_threshold_val.png"
    )

    # Score distribution (validation only)
    print("📊 Saving score distribution plot (validation)...")
    plot_score_distribution(
        model, X_val, y_val, "Validation", PLOT_DIR / "xgb_score_distribution_val.png"
    )

    print("\n✅ All plots saved under:", PLOT_DIR)


if __name__ == "__main__":
    main()
