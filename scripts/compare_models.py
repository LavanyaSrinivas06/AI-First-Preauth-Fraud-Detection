import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)

from tensorflow.keras.models import load_model


# ==========================
# Paths
# ==========================
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "test.csv"
PREPROCESS_PATH = ROOT / "artifacts" / "preprocess.joblib"

XGB_MODEL_PATH = ROOT / "artifacts" / "xgb_model.pkl"
AE_MODEL_PATH = ROOT / "artifacts" / "autoencoder_model.h5"
AE_THRESHOLD_PATH = ROOT / "artifacts" / "ae_threshold.txt"

PLOT_DIR = ROOT / "artifacts" / "plots" / "model_comparison"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = ROOT / "docs" / "model_comparison_report.md"


# ==========================
# Load data & models
# ==========================
print("🔄 Loading test data, preprocessing, and models...")

df = pd.read_csv(DATA_PATH)
X_test = df.drop(columns=["Class"])
y_test = df["Class"].values

preprocess = joblib.load(PREPROCESS_PATH)
X_test_processed = X_test.values

xgb_model = joblib.load(XGB_MODEL_PATH)
#autoencoder = load_model(AE_MODEL_PATH)
autoencoder = load_model(AE_MODEL_PATH, compile=False)


with open(AE_THRESHOLD_PATH, "r") as f:
    ae_threshold = float(f.read().strip())

print("✔ Loaded all artifacts.")


# ==========================
# XGBoost Evaluation
# ==========================
print("\n=== XGBoost Evaluation ===")

xgb_proba = xgb_model.predict_proba(X_test_processed)[:, 1]
xgb_pred = (xgb_proba >= 0.5).astype(int)

metrics_xgb = {
    "precision": precision_score(y_test, xgb_pred),
    "recall": recall_score(y_test, xgb_pred),
    "f1": f1_score(y_test, xgb_pred),
    "roc_auc": roc_auc_score(y_test, xgb_proba),
    "pr_auc": average_precision_score(y_test, xgb_proba),
    "confusion_matrix": confusion_matrix(y_test, xgb_pred).tolist(),
}


# ==========================
# Autoencoder Evaluation
# ==========================
print("\n=== Autoencoder Evaluation ===")

X_recon = autoencoder.predict(X_test_processed, verbose=0)
ae_errors = np.mean((X_test_processed - X_recon)**2, axis=1)

ae_pred = (ae_errors >= ae_threshold).astype(int)

metrics_ae = {
    "precision": precision_score(y_test, ae_pred, zero_division=0),
    "recall": recall_score(y_test, ae_pred, zero_division=0),
    "f1": f1_score(y_test, ae_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, ae_errors),
    "pr_auc": average_precision_score(y_test, ae_errors),
    "confusion_matrix": confusion_matrix(y_test, ae_pred).tolist(),
}


# ==========================
# Save ROC & PR Curves
# ==========================
print("📈 Saving ROC and Precision–Recall plots...")

plt.figure(figsize=(10, 6))
RocCurveDisplay.from_predictions(y_test, xgb_proba, name="XGBoost")
RocCurveDisplay.from_predictions(y_test, ae_errors, name="Autoencoder")
plt.title("ROC Curve Comparison")
plt.savefig(PLOT_DIR / "roc_comparison.png")
plt.close()

plt.figure(figsize=(10, 6))
PrecisionRecallDisplay.from_predictions(y_test, xgb_proba, name="XGBoost")
PrecisionRecallDisplay.from_predictions(y_test, ae_errors, name="Autoencoder")
plt.title("Precision–Recall Curve Comparison")
plt.savefig(PLOT_DIR / "pr_comparison.png")
plt.close()


# ==========================
# Save Confusion Matrices
# ==========================
cm_xgb = np.array(metrics_xgb["confusion_matrix"])
cm_ae = np.array(metrics_ae["confusion_matrix"])

plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plt.imshow(cm_xgb, cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(cm_ae, cmap="Greens")
plt.title("Autoencoder Confusion Matrix")
plt.colorbar()

plt.savefig(PLOT_DIR / "confusion_matrices.png")
plt.close()


# ==========================
# Save comparison report (Markdown)
# ==========================
print("📝 Writing model_comparison_report.md ...")

with open(REPORT_PATH, "w") as f:
    f.write("# Model Comparison Report (XGBoost vs Autoencoder)\n\n")
    f.write("## Metrics Comparison Table\n")
    f.write("| Metric | XGBoost | Autoencoder |\n")
    f.write("|--------|---------|-------------|\n")
    for m in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        f.write(f"| {m} | {metrics_xgb[m]:.4f} | {metrics_ae[m]:.4f} |\n")

    f.write("\n## Confusion Matrices\n")
    f.write(f"**XGBoost:** {metrics_xgb['confusion_matrix']}\n\n")
    f.write(f"**Autoencoder:** {metrics_ae['confusion_matrix']}\n\n")

    f.write("## Notes\n")
    f.write("- XGBoost excels at precision and stable detection of known fraud patterns.\n")
    f.write("- Autoencoder excels at recall (finding unseen patterns) but has more false positives.\n")
    f.write("- Ensemble fusion may be explored in Phase 3.\n")

print("✔ Comparison complete.")
