import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras.models import load_model


# ------------------------------------------------------
# Paths
# ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "autoencoder_model.h5"
THRESHOLD_PATH = ARTIFACT_DIR / "ae_threshold.txt"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"


# ------------------------------------------------------
# Helper: Reconstruction error
# ------------------------------------------------------
def reconstruction_errors(model, X):
    X = np.asarray(X)
    X_pred = model.predict(X, verbose=0)
    return np.mean((X - X_pred) ** 2, axis=1)


# ------------------------------------------------------
# Load everything
# ------------------------------------------------------
def load_all():
    print("\n🔄 Loading model, threshold, and data...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Missing threshold: {THRESHOLD_PATH}")

   #model = load_model(MODEL_PATH)
    model = load_model(MODEL_PATH, compile=False)



    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())

    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"].values

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"].values

    print("✔ Loaded.")

    return model, threshold, X_val, y_val, X_test, y_test


# ------------------------------------------------------
# Evaluate on a split
# ------------------------------------------------------
def evaluate_split(name, model, threshold, X, y):
    print(f"\n===== {name.upper()} EVALUATION =====")

    errors = reconstruction_errors(model, X)
    preds = (errors > threshold).astype(int)

    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    roc_auc = roc_auc_score(y, errors)
    pr_auc = average_precision_score(y, errors)
    cm = confusion_matrix(y, preds)

    print(f"Threshold: {threshold:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y, preds))

    return errors, preds


# ------------------------------------------------------
# Show TP/FP/FN examples
# ------------------------------------------------------
def show_examples(X, y_true, errors, y_pred, name, k=5):
    df = X.copy()
    df["error"] = errors
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    print(f"\n===== {name.upper()} EXAMPLES =====")

    tp = df[(df.y_true == 1) & (df.y_pred == 1)]
    fn = df[(df.y_true == 1) & (df.y_pred == 0)]
    fp = df[(df.y_true == 0) & (df.y_pred == 1)]

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
    model, threshold, X_val, y_val, X_test, y_test = load_all()

    # Validation
    val_errors, val_preds = evaluate_split(
        "validation", model, threshold, X_val, y_val
    )
    show_examples(
        X_val.copy(), y_val, val_errors, val_preds, "validation"
    )

    # Test
    test_errors, test_preds = evaluate_split(
        "test", model, threshold, X_test, y_test
    )
    show_examples(
        X_test.copy(), y_test, test_errors, test_preds, "test"
    )


if __name__ == "__main__":
    main()
