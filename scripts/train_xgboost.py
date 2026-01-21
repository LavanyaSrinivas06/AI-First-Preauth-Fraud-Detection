#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

"""
Train XGBoost model with hyperparameter tuning and threshold optimization.
Saves:
- artifacts/xgb_model.pkl
- artifacts/xgb_metrics.json
- artifacts/plots/xgb_feature_importance.png
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, plot_importance
from src.utils.data_loader import load_dataset


from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import RandomizedSearchCV

# ---------------------------
# Paths
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
PLOT_DIR = ARTIFACT_DIR / "plots"

ARTIFACT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# XGBoost should train on SMOTEd data
TRAIN_PATH = DATA_DIR / "train.csv"          # SMOTEd
VAL_PATH   = DATA_DIR / "val.csv"            # not smoted
TEST_PATH  = DATA_DIR / "test.csv"           # not smoted
USED_SMOTE = True



# ---------------------------
# Load Data
# ---------------------------
print("📥 Loading processed datasets...")

# Optional safety: check files exist
for path, name in [(TRAIN_PATH, "Training"), (VAL_PATH, "Validation"), (TEST_PATH, "Test")]:
    if not path.exists():
        raise FileNotFoundError(f"{name} data not found at {path}. "
                                f"Please run the preprocessing script first.")

# Use shared loader from src.utils.data_loader
train_df, X_train, y_train = load_dataset(TRAIN_PATH)
val_df, X_val, y_val = load_dataset(VAL_PATH)
test_df, X_test, y_test = load_dataset(TEST_PATH)

print("✔ Data loaded.")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ---------------------------
# Save feature schema (ORDER MATTERS)
# ---------------------------
FEATURES_PATH = ARTIFACT_DIR / "features.json"

# Case A: X_train is a DataFrame (best)
if hasattr(X_train, "columns"):
    feature_names = list(X_train.columns)

# Case B: X_train is a NumPy array (no names). Use CSV columns as source of truth.
else:
    # train_df likely contains the full row including label; adjust label column name if different
    label_candidates = ["Class", "label", "is_fraud", "fraud"]
    label_col = next((c for c in label_candidates if c in train_df.columns), None)
    if label_col is None:
        raise RuntimeError("Could not find label column in train_df to derive feature schema.")

    feature_names = [c for c in train_df.columns if c != label_col]

with open(FEATURES_PATH, "w") as f:
    json.dump(feature_names, f, indent=2)

print(f"✔ Saved: {FEATURES_PATH} ({len(feature_names)} features)")



# ---------------------------
# Baseline Model
# ---------------------------
print("\n🚀 Training baseline XGBoost model...")

#scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
scale_pos_weight = 1.0 if USED_SMOTE else (y_train == 0).sum() / (y_train == 1).sum()

XGB_CONFIG = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "scale_pos_weight": scale_pos_weight,
    "n_jobs": -1,
    "tree_method": "hist"
}

baseline_xgb = XGBClassifier(**XGB_CONFIG)

baseline_xgb.fit(X_train, y_train)
print("✔ Baseline model trained.")

# ---------------------------
# Hyperparameter Tuning
# ---------------------------
print("\n🎯 Starting hyperparameter search...")

param_grid = {
    "max_depth": [3, 4, 5,6],
    "learning_rate": [0.01, 0.03, 0.05],
    "n_estimators": [150, 200, 250, 300],  
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

tuner = RandomizedSearchCV(
    estimator=XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        tree_method="hist",
    ),
    param_distributions=param_grid,
    n_iter=30,
    scoring="average_precision",
    cv=3,
    verbose=1,
    n_jobs=-1,
)

tuner.fit(X_train, y_train)
print("✔ Hyperparameter tuning complete.")

best_params = tuner.best_params_
print("Best params:", best_params)

best_xgb = tuner.best_estimator_

# ---------------------------
# Threshold Optimization
# ---------------------------
print("\n📌 Optimizing probability threshold...")

val_probs = best_xgb.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.001, 0.5, 500)
best_score = -1.0   # ✅ INITIALIZE
best_t = 0.5

for t in thresholds:
    preds = (val_probs >= t).astype(int)
    score = fbeta_score(y_val, preds, beta=2, zero_division=0)
    if score > best_score:
        best_score = score
        best_t = t

print(f"✔ Best Threshold: {best_t:.4f}")
print(f"✔ Best F2: {best_score:.4f}")


# ---------------------------
# Final Evaluation Function
# ---------------------------
def evaluate(model, X, y, t):
    """
    Compute evaluation metrics for a given model, dataset, and threshold.
    """
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise RuntimeError(
            "Error during evaluation: model.predict_proba(X) failed. "
            "Check that X has the expected shape and that the model is fitted. "
            f"Original error: {e}"
        )

    preds = (probs >= t).astype(int)

    return {
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, probs),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "classification_report": classification_report(y, preds, output_dict=True),
    }

# ---------------------------
# Evaluate on Validation + Test
# ---------------------------
print("\n📊 Computing final metrics...")

metrics_val = evaluate(best_xgb, X_val, y_val, best_t)
metrics_test = evaluate(best_xgb, X_test, y_test, best_t)


# ---------------------------
# Weights & Biases logging
# ---------------------------
import wandb

wandb.init(
    project="ai-first-preauth-fraud",
    name="xgboost-preauth-training",
    config={
        **best_params,
        "used_smote": USED_SMOTE,
        "scale_pos_weight": scale_pos_weight,
        "model_type": "XGBoost",
    }
)

wandb.log({
    # Validation metrics
    "val/roc_auc": metrics_val["roc_auc"],
    "val/precision": metrics_val["precision"],
    "val/recall": metrics_val["recall"],
    "val/f1": metrics_val["f1"],

    # Test metrics
    "test/roc_auc": metrics_test["roc_auc"],
    "test/precision": metrics_test["precision"],
    "test/recall": metrics_test["recall"],
    "test/f1": metrics_test["f1"],

    # Threshold
    "decision/best_threshold": best_t,
})

# Probability distribution
wandb.log({
    "val/xgb_probability_distribution": wandb.Histogram(val_probs),
})

# Decision flow simulation (aligns with your preauth logic)
approve_rate = (val_probs < best_t).mean()
review_rate = ((val_probs >= best_t) & (val_probs < 0.80)).mean()
block_rate = (val_probs >= 0.80).mean()

wandb.log({
    "decision_flow/approve_rate": approve_rate,
    "decision_flow/review_rate": review_rate,
    "decision_flow/block_rate": block_rate,
})

wandb.finish()


# ---------------------------
# Save Model + Metrics
# ---------------------------
print("\n💾 Saving model and metrics...")

MODEL_DIR = ARTIFACT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(best_xgb, MODEL_DIR / "xgb_model.pkl")

with open(ARTIFACT_DIR / "xgb_metrics.json", "w") as f:
    json.dump(
        {
            "threshold": best_t,
            "validation": metrics_val,
            "test": metrics_test,
            "best_params": best_params,
        },
        f,
        indent=4,
    )

print("✔ Saved: artifacts/xgb_model.pkl")
print("✔ Saved: artifacts/xgb_metrics.json")

# ---------------------------
# Feature Importance Plot
# ---------------------------
print("\n📈 Saving feature importance plot...")

plt.figure(figsize=(14, 10))
plot_importance(best_xgb, max_num_features=20)
plt.tight_layout()
plt.savefig(PLOT_DIR / "xgb_feature_importance.png")
plt.close()

print("✔ Saved: artifacts/plots/xgb_feature_importance.png")

print("\n🎉 Training pipeline complete!")
