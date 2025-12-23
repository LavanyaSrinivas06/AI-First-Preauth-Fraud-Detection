#!/usr/bin/env python3
"""
Evaluate the trained Autoencoder using the stored dual thresholds (review/block).

Inputs:
- artifacts/autoencoder_model.keras
- artifacts/ae_thresholds.json
- data/processed/val_nosmote.csv
- data/processed/test_nosmote.csv

Outputs: console evaluation + example rows
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

from tensorflow.keras.models import load_model


# ------------------------------------------------------
# Paths
# ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "autoencoder_model.keras"
THRESHOLDS_PATH = ARTIFACT_DIR / "ae_thresholds.json"

VAL_PATH  = DATA_DIR / "val_nosmote.csv"
TEST_PATH = DATA_DIR / "test_nosmote.csv"


# ------------------------------------------------------
# Helper: Reconstruction error
# ------------------------------------------------------
def reconstruction_errors(model, X):
    X = np.asarray(X).astype("float32")
    X_pred = model.predict(X, verbose=0)
    return np.mean((X - X_pred) ** 2, axis=1)


def triage(errors: np.ndarray, t_review: float, t_block: float) -> np.ndarray:
    decisions = np.zeros(len(errors), dtype=int)
    decisions[errors >= t_review] = 1
    decisions[errors >= t_block] = 2
    return decisions


def print_binary_eval(name: str, y_true: np.ndarray, pred: np.ndarray, scores: np.ndarray, label: str):
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    cm = confusion_matrix(y_true, pred)

    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    print(f"\n--- {name.upper()} | {label} ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, pred, digits=4))


def show_examples(df_X: pd.DataFrame, y_true: np.ndarray, errors: np.ndarray, pred_bin: np.ndarray, title: str, k: int = 5):
    df = df_X.copy()
    df["error"] = errors
    df["y_true"] = y_true
    df["y_pred"] = pred_bin

    tp = df[(df.y_true == 1) & (df.y_pred == 1)]
    fn = df[(df.y_true == 1) & (df.y_pred == 0)]
    fp = df[(df.y_true == 0) & (df.y_pred == 1)]

    print(f"\n===== {title.upper()} EXAMPLES =====")

    print(f"\nTrue positives (fraud caught) – {len(tp)} rows")
    print(tp.head(k)[["error", "y_true", "y_pred"]])

    print(f"\nFalse negatives (fraud MISSED) – {len(fn)} rows")
    print(fn.head(k)[["error", "y_true", "y_pred"]])

    print(f"\nFalse positives (non-fraud flagged) – {len(fp)} rows")
    print(fp.head(k)[["error", "y_true", "y_pred"]])


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():
    print("\n🔄 Loading model, thresholds, and data...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f"Missing thresholds file: {THRESHOLDS_PATH}")
    if not VAL_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Missing val/test NO-SMOTE CSVs. Run preprocessing first.")

    model = load_model(MODEL_PATH, compile=False)

    with open(THRESHOLDS_PATH, "r") as f:
        thr = json.load(f)
    t_review = float(thr["review"])
    t_block = float(thr["block"])

    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"].values

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"].values

    print("✔ Loaded.")
    print(f"Thresholds: review={t_review:.6f}, block={t_block:.6f}")

    # ---------------- VAL ----------------
    val_errors = reconstruction_errors(model, X_val.values)
    val_flagged = (val_errors >= t_review).astype(int)  # review OR block
    val_blocked = (val_errors >= t_block).astype(int)   # block only

    print("\n===== VALIDATION =====")
    print_binary_eval("validation", y_val, val_flagged, val_errors, "FLAGGED (review+block)")
    print_binary_eval("validation", y_val, val_blocked, val_errors, "BLOCKED ONLY")

    # Examples for flagged
    show_examples(X_val, y_val, val_errors, val_flagged, "validation flagged", k=5)

    # ---------------- TEST ----------------
    test_errors = reconstruction_errors(model, X_test.values)
    test_flagged = (test_errors >= t_review).astype(int)
    test_blocked = (test_errors >= t_block).astype(int)

    print("\n===== TEST =====")
    print_binary_eval("test", y_test, test_flagged, test_errors, "FLAGGED (review+block)")
    print_binary_eval("test", y_test, test_blocked, test_errors, "BLOCKED ONLY")

    show_examples(X_test, y_test, test_errors, test_flagged, "test flagged", k=5)

    # Triage counts (approve/review/block)
    tri_test = triage(test_errors, t_review, t_block)
    approve = int((tri_test == 0).sum())
    review = int((tri_test == 1).sum())
    block = int((tri_test == 2).sum())

    print("\n✅ Triage counts on TEST")
    print(f"  Approve: {approve}")
    print(f"  Review : {review}")
    print(f"  Block  : {block}")


if __name__ == "__main__":
    main()
