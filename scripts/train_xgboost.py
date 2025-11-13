#!/usr/bin/env python3
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
from sklearn.metrics import (
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

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

# ---------------------------
# Load Data
# ---------------------------
print("📥 Loading processed datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]

X_val = val_df.drop(columns=["Class"])
y_val = val_df["Class"]

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

print("✔ Data loaded.")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ---------------------------
# Baseline Model
# ---------------------------
print("\n🚀 Training baseline XGBoost model...")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

baseline_xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    tree_method="hist"
)

baseline_xgb.fit(X_train, y_train)
print("✔ Baseline model trained.")

# ---------------------------
# Hyperparameter Tuning
# ---------------------------
print("\n🎯 Starting hyperparameter search...")

param_grid = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05],
    "n_estimators": [150, 200, 250],
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
    n_iter=10,
    scoring="f1",
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

thresholds = np.linspace(0.1, 0.9, 50)
best_f1 = 0
best_t = 0.5

for t in thresholds:
    preds = (val_probs >= t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print(f"✔ Best Threshold: {best_t:.4f}")
print(f"✔ Best F1: {best_f1:.4f}")

# ---------------------------
# Final Evaluation Function
# ---------------------------
def evaluate(model, X, y, t):
    probs = model.predict_proba(X)[:, 1]
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
# Save Model + Metrics
# ---------------------------
print("\n💾 Saving model and metrics...")

joblib.dump(best_xgb, ARTIFACT_DIR / "xgb_model.pkl")

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
