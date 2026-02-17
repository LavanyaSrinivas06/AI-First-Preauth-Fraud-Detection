"""
Ablation Study: Contribution of Autoencoder Gray-Zone Routing

Evaluates three configurations:
A) XGBoost baseline
B) Threshold-only (no AE)
C) Full Hybrid (AE in gray zone)

- Loads preprocessing, models, and test set
- Computes metrics and confusion matrices
- Performs McNemar's test (XGB vs Hybrid, Threshold-only vs Hybrid)
- Prints all results in thesis-ready format
- Reproducible (random seed=42)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from tensorflow.keras.models import load_model

np.random.seed(42)

# === CONFIGURATION ===

PREPROCESS_PATH = Path("artifacts/preprocess/preprocess.joblib")
XGB_MODEL_PATH = Path("artifacts/models/xgb_model.pkl")
AE_MODEL_PATH = Path("artifacts/models/autoencoder_model.keras")
TEST_PATH = Path("data/processed/test_renamed.csv")

xgb_t_low = 0.05
xgb_t_high = 0.80
xgb_main_threshold = 0.307
ae_threshold = 0.692

# === LOAD ARTIFACTS ===
preprocessor = joblib.load(PREPROCESS_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)
ae_model = load_model(AE_MODEL_PATH, compile=False)
test_df = pd.read_csv(TEST_PATH)

label_col = None
for col in ["label", "Class", "target", "y_true"]:
    if col in test_df.columns:
        label_col = col
        break
if label_col is None:
    raise ValueError(f"Test set must have a ground truth column. Columns found: {list(test_df.columns)}")

X_test = test_df.drop(columns=[label_col])
y_true = test_df[label_col].values.astype(int)
X_test_proc = preprocessor.transform(X_test)

# === XGBOOST BASELINE ===
xgb_probs = xgb_model.predict_proba(X_test_proc)[:, 1]
xgb_pred = (xgb_probs >= xgb_main_threshold).astype(int)

# === THRESHOLD-ONLY (NO AE) ===
thresh_only_pred = np.zeros_like(xgb_probs, dtype=int)
thresh_only_pred[xgb_probs < xgb_t_low] = 0  # Approve
thresh_only_pred[xgb_probs >= xgb_t_high] = 1  # Block
mask_gray = (xgb_probs >= xgb_t_low) & (xgb_probs < xgb_t_high)
thresh_only_pred[mask_gray] = (xgb_probs[mask_gray] >= xgb_main_threshold).astype(int)

# === FULL HYBRID ===
ae_recon = ae_model.predict(X_test_proc)
ae_errors = np.mean(np.square(X_test_proc - ae_recon), axis=1)
hybrid_pred = np.zeros_like(xgb_probs, dtype=int)
hybrid_pred[xgb_probs < xgb_t_low] = 0  # Approve
hybrid_pred[xgb_probs >= xgb_t_high] = 1  # Block
mask_gray = (xgb_probs >= xgb_t_low) & (xgb_probs < xgb_t_high)
hybrid_pred[mask_gray] = (ae_errors[mask_gray] >= ae_threshold).astype(int)

# === METRICS FUNCTION ===
def compute_metrics(y_true, y_pred):
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist()
    }

results = {
    "XGBoost": compute_metrics(y_true, xgb_pred),
    "Threshold-only": compute_metrics(y_true, thresh_only_pred),
    "Hybrid": compute_metrics(y_true, hybrid_pred)
}

# === PRINT METRICS ===
print("\nAblation Study Results\n")
for model, metrics in results.items():
    print(f"{model}:")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1:        {metrics['F1']:.4f}")
    print(f"  Confusion Matrix: {metrics['Confusion Matrix']}")
    print()

# === FINAL COMPARISON TABLE ===
print("| Model | Precision | Recall | F1 |")
for model, metrics in results.items():
    print(f"| {model} | {metrics['Precision']:.4f} | {metrics['Recall']:.4f} | {metrics['F1']:.4f} |")

# === MCNEMAR'S TEST ===
def mcnemar_report(y_true, pred_a, pred_b, label_a, label_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    n11 = int(np.sum(correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    table = np.array([[n00, n10], [n01, n11]])
    result = mcnemar(table, exact=True)
    print(f"\nMcNemar's Test: {label_a} vs {label_b}")
    print(f"Contingency Table:\n{table}")
    print(f"n01 ({label_a} correct, {label_b} wrong) = {n01}")
    print(f"n10 ({label_a} wrong, {label_b} correct) = {n10}")
    print(f"p-value = {result.pvalue:.5f}")
    if result.pvalue < 0.05:
        print("Result: Statistically significant difference (p < 0.05)")
    else:
        print("Result: No statistically significant difference (p >= 0.05)")

mcnemar_report(y_true, xgb_pred, hybrid_pred, "XGBoost", "Hybrid")
mcnemar_report(y_true, thresh_only_pred, hybrid_pred, "Threshold-only", "Hybrid")
