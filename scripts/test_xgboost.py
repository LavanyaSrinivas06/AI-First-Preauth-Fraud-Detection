#scripts/test_xgboost.py
#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

"""
Quick sanity test for the trained XGBoost model.

- Loads model + threshold + processed val/test data
- Checks how many frauds are caught / missed
- Prints example rows for manual inspection
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from src.utils.data_loader import load_dataset

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = ARTIFACT_DIR / "xgb_model.pkl"
METRICS_PATH = ARTIFACT_DIR / "xgb_metrics.json"


# -------------------------------------------------------------------
# Loading helpers
# -------------------------------------------------------------------
def load_data():
    """
    Load validation and test CSVs and split into:
    - full DataFrames (val_df, test_df)
    - features (X_val, X_test)
    - labels (y_val, y_test)
    """
    val_df, X_val, y_val = load_dataset(VAL_PATH)
    test_df, X_test, y_test = load_dataset(TEST_PATH)
    return val_df, X_val, y_val, test_df, X_test, y_test



def load_model_and_threshold():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Please run `train_xgboost.py` before loading the model."
        )

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found at {METRICS_PATH}. "
            f"Please ensure training completed successfully."
        )

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

    try:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load metrics JSON at {METRICS_PATH}: {e}")

    threshold = float(metrics.get("threshold", 0.5))
    return model, threshold



# -------------------------------------------------------------------
# Simple evaluation
# -------------------------------------------------------------------
def evaluate_split(model, X, y, threshold: float, split_name: str):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)

    print(f"\n===== {split_name.upper()} SUMMARY =====")
    print(f"Threshold: {threshold:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y, preds, digits=4))

    return proba, preds


# -------------------------------------------------------------------
# Show example transactions (TP / FN / FP)
# -------------------------------------------------------------------
def show_examples(df, y_true, proba, preds, split_name: str, k: int = 5):
    """
    Print example transactions for manual inspection grouped by prediction outcome.

    Shows:
      - True positives: fraud correctly detected
      - False negatives: fraud missed
      - False positives: non-fraud flagged as fraud

    Args:
        df: Original DataFrame with all features.
        y_true: True labels (Series or array-like, 0/1).
        proba: Predicted fraud probabilities from the model.
        preds: Binary predicted labels (0/1) after thresholding.
        split_name: Name of the split ("validation" or "test").
        k: Max number of examples to show from each group.
    """
    df_local = df.copy()
    df_local["y_true"] = y_true.values
    df_local["y_pred"] = preds
    df_local["proba"] = proba

    # True positives: fraud correctly detected
    tp = df_local[(df_local["y_true"] == 1) & (df_local["y_pred"] == 1)]
    # False negatives: fraud missed
    fn = df_local[(df_local["y_true"] == 1) & (df_local["y_pred"] == 0)]
    # False positives: non-fraud flagged as fraud
    fp = df_local[(df_local["y_true"] == 0) & (df_local["y_pred"] == 1)]

    def _print_block(name, df_block):
        print(f"\n{name} – {len(df_block)} rows")
        if len(df_block) > 0:
            cols_to_show = ["proba", "y_true", "y_pred"]
            print(df_block.head(min(k, len(df_block)))[cols_to_show])

    print(f"\n===== {split_name.upper()} EXAMPLES =====")

    _print_block("True positives (fraud correctly caught)", tp)
    _print_block("False negatives (fraud MISSED)", fn)
    _print_block("False positives (non-fraud flagged)", fp)



# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print("🔄 Loading data, model, and threshold...")
    val_df, X_val, y_val, test_df, X_test, y_test = load_data()
    model, threshold = load_model_and_threshold()
    print("✔ Loaded.")

    # Validation set check
    val_proba, val_preds = evaluate_split(model, X_val, y_val, threshold, "validation")
    show_examples(val_df, y_val, val_proba, val_preds, "validation", k=5)

    # Test set check
    test_proba, test_preds = evaluate_split(model, X_test, y_test, threshold, "test")
    show_examples(test_df, y_test, test_proba, test_preds, "test", k=5)


if __name__ == "__main__":
    main()
