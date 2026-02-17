# /experiments/comparative_eval.py

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from pathlib import Path
import json
from tensorflow.keras.models import load_model

# === CONFIGURATION ===
# Update these paths if your artifacts are elsewhere
XGB_MODEL_PATH = Path("artifacts/models/xgb_model.pkl")
# Use the Keras .h5 model from the archive (or move it to artifacts/ if you prefer)
AE_MODEL_PATH = Path("archive/artifacts_old_or_duplicates/autoencoder_model.h5")
AE_THRESHOLD_PATH = Path("artifacts/thresholds/ae_thresholds.json")
XGB_METRICS_PATH = Path("artifacts/xgb_metrics.json")
import glob

# Try to find a test set CSV in common locations
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
    # Try any CSV in data/processed/
    files = glob.glob("data/processed/*.csv")
    if files:
        TEST_DATA_PATH = Path(files[0])
if TEST_DATA_PATH is None:
    raise FileNotFoundError("No test set CSV found in common locations. Please provide a test set CSV.")
RESULTS_CSV_PATH = Path("experiments/comparative_eval_results.csv")

# === LOAD MODELS AND THRESHOLDS ===

if not XGB_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing XGBoost model at {XGB_MODEL_PATH}")
if not AE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing Autoencoder model at {AE_MODEL_PATH}")
if not TEST_DATA_PATH.exists():
    raise FileNotFoundError(f"Missing test dataset at {TEST_DATA_PATH}")
if not AE_THRESHOLD_PATH.exists():
    raise FileNotFoundError(f"Missing AE thresholds at {AE_THRESHOLD_PATH}")

if not XGB_METRICS_PATH.exists():
    raise FileNotFoundError(f"Missing XGB metrics at {XGB_METRICS_PATH}")

xgb_model = joblib.load(XGB_MODEL_PATH)
ae_model = load_model(AE_MODEL_PATH, compile=False)

with open(AE_THRESHOLD_PATH) as f:
    ae_thresholds = json.load(f)
with open(XGB_METRICS_PATH) as f:
    xgb_metrics = json.load(f)

# Use the threshold from metrics file for both low and high (single threshold logic)
xgb_thr = xgb_metrics.get("threshold", 0.5)
xgb_t_low = xgb_thr
xgb_t_high = xgb_thr
ae_review = ae_thresholds.get("review", 0.12)
ae_block = ae_thresholds.get("block", 0.307)  # Updated block threshold

# === LOAD TEST DATA ===

test_df = pd.read_csv(TEST_DATA_PATH)
label_col = None
for col in ["label", "Class", "target", "y_true"]:
    if col in test_df.columns:
        label_col = col
        break
if label_col is None:
    raise ValueError(f"Test set {TEST_DATA_PATH} must have a ground truth column named 'label', 'Class', 'target', or 'y_true'. Columns found: {list(test_df.columns)}")

X_test = test_df.drop(columns=[label_col])
y_test = test_df[label_col].values

# === PREPROCESSING ===
# If you have a preprocessing pipeline, load and apply it here
PREPROCESSOR_PATH = Path("artifacts/preprocess/preprocessor.pkl")
if PREPROCESSOR_PATH.exists():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_test_proc = preprocessor.transform(X_test)
else:
    # If no pipeline, assume X_test is already preprocessed
    X_test_proc = X_test.values

# === XGBOOST PREDICTIONS ===
xgb_probs = xgb_model.predict_proba(X_test_proc)[:, 1]
xgb_preds = (xgb_probs >= xgb_t_high).astype(int)

# === AUTOENCODER PREDICTIONS ===
# Keras model expects numpy arrays
ae_recon = ae_model.predict(X_test_proc)
ae_errors = np.mean(np.square(X_test_proc - ae_recon), axis=1)
ae_preds = (ae_errors >= ae_block).astype(int)

# === HYBRID DECISION ENGINE ===
def hybrid_decision(p_xgb, ae_err):
    if p_xgb < xgb_t_low:
        return 0  # APPROVE
    if p_xgb >= xgb_t_high:
        return 1  # BLOCK
    if ae_err >= ae_block:
        return 1  # BLOCK
    return 2  # REVIEW

hybrid_preds = []
for px, ae in zip(xgb_probs, ae_errors):
    dec = hybrid_decision(px, ae)
    # For metrics, treat REVIEW as not fraud (0)
    hybrid_preds.append(1 if dec == 1 else 0)
hybrid_preds = np.array(hybrid_preds)

# === METRICS FUNCTION ===
def compute_metrics(y_true, y_pred, y_score):
    return {
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist()
    }

results = {
    "XGBoost": compute_metrics(y_test, xgb_preds, xgb_probs),
    "Autoencoder": compute_metrics(y_test, ae_preds, ae_errors),
    "Hybrid": compute_metrics(y_test, hybrid_preds, xgb_probs)  # Use xgb_probs for ROC-AUC
}


# === SAVE RESULTS ===
df_results = pd.DataFrame([
    {
        "Model": k,
        "Precision": v["Precision"],
        "Recall": v["Recall"],
        "F1-score": v["F1-score"],
        "ROC-AUC": v["ROC-AUC"],
        "Confusion Matrix": v["Confusion Matrix"]
    }
    for k, v in results.items()
])
df_results.to_csv(RESULTS_CSV_PATH, index=False)

# === EXPORT PER-TRANSACTION PREDICTIONS FOR STATISTICAL TESTING ===
test_pred_df = pd.DataFrame({
    "y_true": y_test,
    "xgb_flag": xgb_preds,
    "hybrid_flag": hybrid_preds
})
test_pred_path = Path("artifacts/test_predictions.csv")
test_pred_path.parent.mkdir(parents=True, exist_ok=True)
test_pred_df.to_csv(test_pred_path, index=False)
print(f"Per-transaction predictions saved to {test_pred_path}")

# === PRETTY PRINT ===
print("\nComparative Evaluation Results:\n")
print(df_results.to_string(index=False))

print(f"\nResults saved to {RESULTS_CSV_PATH}")
