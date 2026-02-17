from statsmodels.stats.contingency_tables import mcnemar
import numpy as np


import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import json
from pathlib import Path

"""
validation_stats.py

Perform McNemar's test to compare two binary classifiers (XGBoost vs Hybrid)
on the same test dataset.

Usage:
- Call perform_mcnemar(y_true, y_pred_xgb, y_pred_hybrid)
- Or run this file directly to see an example.
"""



def perform_mcnemar(y_true, y_pred_xgb, y_pred_hybrid, alpha=0.05):
    """
    Compute 2x2 contingency table and perform McNemar's test.

    Assumptions:
    - y_true, y_pred_xgb, y_pred_hybrid are array-like of 0/1 labels and same length.

    Table layout (rows = XGBoost wrong(0)/correct(1), cols = Hybrid wrong(0)/correct(1)):
            Hybrid wrong  Hybrid correct
    XGB wrong      n00           n10
    XGB correct    n01           n11

    Where:
    - n01 = XGBoost correct & Hybrid wrong
    - n10 = XGBoost wrong & Hybrid correct
    """
    y_true = np.asarray(y_true).ravel()
    y_xgb = np.asarray(y_pred_xgb).ravel()
    y_hybrid = np.asarray(y_pred_hybrid).ravel()

    if not (len(y_true) == len(y_xgb) == len(y_hybrid)):
        raise ValueError("Input arrays must have the same length.")

    # Convert to boolean correct indicators
    xgb_correct = (y_xgb == y_true)
    hybrid_correct = (y_hybrid == y_true)

    # Counts
    n11 = int(np.sum(xgb_correct & hybrid_correct))
    n00 = int(np.sum(~xgb_correct & ~hybrid_correct))
    n01 = int(np.sum(xgb_correct & ~hybrid_correct))  # XGB correct, Hybrid wrong
    n10 = int(np.sum(~xgb_correct & hybrid_correct))  # XGB wrong, Hybrid correct

    # Build contingency table in the orientation expected: [[n00, n10],[n01, n11]]
    table = np.array([[n00, n10],
                      [n01, n11]])

    print("Contingency table (rows = XGB wrong/correct, cols = Hybrid wrong/correct):")
    print(table)
    print(f"n01 (XGB correct, Hybrid wrong) = {n01}")
    print(f"n10 (XGB wrong, Hybrid correct) = {n10}")

    # Choose exact test if discordant counts are small
    discordant = n01 + n10
    exact = True if discordant <= 25 else False

    result = mcnemar(table, exact=exact)
    stat = result.statistic
    pvalue = result.pvalue

    print(f"\nMcNemar test (exact={exact}):")
    print(f"Statistic = {stat}")
    print(f"P-value = {pvalue}")

    # Interpretation
    if pvalue < alpha:
        # Determine direction of improvement
        if n01 > n10:
            direction = "XGBoost performs significantly better than Hybrid (more XGB-correct / Hybrid-wrong cases)."
        elif n10 > n01:
            direction = "Hybrid performs significantly better than XGBoost (more Hybrid-correct / XGB-wrong cases)."
        else:
            direction = "Significant difference detected, but discordant counts are equal — check data."
        print(f"Result: p < {alpha} → statistically significant. {direction}")
    else:
        print(f"Result: p >= {alpha} → NOT statistically significant (no evidence of improvement).")

    return {"table": table, "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "statistic": stat, "pvalue": pvalue, "exact": exact}


if __name__ == "__main__":
    # === CONFIGURATION ===
    # Paths should match your artifacts and test set
    XGB_MODEL_PATH = Path("artifacts/models/xgb_model.pkl")
    AE_MODEL_PATH = Path("archive/artifacts_old_or_duplicates/autoencoder_model.h5")
    AE_THRESHOLD_PATH = Path("artifacts/thresholds/ae_thresholds.json")
    XGB_METRICS_PATH = Path("artifacts/xgb_metrics.json")

    # Find test set
    TEST_DATA_PATH = None
    for candidate in [
        "data/processed/test.csv",
        "data/processed/test_set.csv",
        "data/processed/val.csv",
        "data/processed/validation.csv",
        "data/processed/eval.csv",
        "data/processed/evaluation.csv",
        "artifacts/reports/base_dataset_summary.csv",
        "demo_eval_results.csv",
        "docs/hybrid_decision_table.csv",
    ]:
        if Path(candidate).exists():
            TEST_DATA_PATH = Path(candidate)
            break
    if TEST_DATA_PATH is None:
        import glob
        files = glob.glob("data/processed/*.csv")
        if files:
            TEST_DATA_PATH = Path(files[0])
    if TEST_DATA_PATH is None:
        raise FileNotFoundError("No test set CSV found in common locations. Please provide a test set CSV.")

    # Load test set
    test_df = pd.read_csv(TEST_DATA_PATH)
    label_col = None
    for col in ["label", "Class", "target", "y_true"]:
        if col in test_df.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"Test set {TEST_DATA_PATH} must have a ground truth column named 'label', 'Class', 'target', or 'y_true'. Columns found: {list(test_df.columns)}")

    X_test = test_df.drop(columns=[label_col])
    y_true = test_df[label_col].values.astype(int)

    # Load models and thresholds
    xgb_model = joblib.load(XGB_MODEL_PATH)
    ae_model = load_model(AE_MODEL_PATH, compile=False)
    with open(AE_THRESHOLD_PATH) as f:
        ae_thresholds = json.load(f)
    with open(XGB_METRICS_PATH) as f:
        xgb_metrics = json.load(f)
    xgb_thr = xgb_metrics.get("threshold", 0.5)
    ae_block = ae_thresholds.get("block", 0.18)
    xgb_t_low = xgb_thr
    xgb_t_high = xgb_thr

    # Preprocessing (if needed)
    PREPROCESSOR_PATH = Path("artifacts/preprocess/preprocessor.pkl")
    if PREPROCESSOR_PATH.exists():
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        X_test_proc = preprocessor.transform(X_test)
    else:
        X_test_proc = X_test.values

    # XGBoost predictions (binary)
    xgb_probs = xgb_model.predict_proba(X_test_proc)[:, 1]
    y_pred_xgb = (xgb_probs >= xgb_t_high).astype(int)

    # Autoencoder predictions (for hybrid)
    ae_recon = ae_model.predict(X_test_proc)
    ae_errors = np.mean(np.square(X_test_proc - ae_recon), axis=1)

    # Hybrid decision logic (same as in comparative_eval.py)
    def hybrid_decision(px, ae):
        if px < xgb_t_low:
            return 0  # APPROVE
        if px >= xgb_t_high:
            return 1  # BLOCK
        if ae >= ae_block:
            return 1  # BLOCK
        return 0  # REVIEW treated as not fraud for metrics

    y_pred_hybrid = np.array([hybrid_decision(px, ae) for px, ae in zip(xgb_probs, ae_errors)])

    # Run McNemar's test
    perform_mcnemar(y_true, y_pred_xgb, y_pred_hybrid)