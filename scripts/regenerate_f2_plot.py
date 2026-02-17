#!/usr/bin/env python3
"""Regenerate the F2-vs-threshold plot with best values annotated on the figure."""

from pathlib import Path
import json
import numpy as np
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH   = ROOT / "artifacts" / "models" / "xgb_model.pkl"
METRICS_PATH = ROOT / "artifacts" / "xgb_metrics.json"
VAL_PATH     = ROOT / "data" / "processed" / "val.csv"
OUT_PATH     = ROOT / "docs" / "figures" / "models" / "04_xgboost" / "xgb_f2_vs_threshold_val.png"

# Load
model = joblib.load(MODEL_PATH)
val = pd.read_csv(VAL_PATH)
y_val = val["Class"]
X_val = val.drop(columns=["Class"])
val_probs = model.predict_proba(X_val)[:, 1]

# Sweep
thresholds = np.linspace(0.001, 0.5, 500)
f2_scores = [fbeta_score(y_val, (val_probs >= t).astype(int), beta=2, zero_division=0)
             for t in thresholds]

best_idx = int(np.argmax(f2_scores))
best_t   = float(thresholds[best_idx])
best_f2  = float(f2_scores[best_idx])

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

ax.plot(thresholds, f2_scores, color="#5C6BC0", linewidth=2, label="F2 Score")
ax.axvline(best_t, color="#EF5350", linewidth=1.3, linestyle="--", alpha=0.8)
ax.plot(best_t, best_f2, 'o', color="#EF5350", markersize=8, zorder=5)

# Annotate best value on the plot
ax.annotate(
    f"Best Threshold = {best_t:.3f}\nF2 Score = {best_f2:.3f}",
    xy=(best_t, best_f2),
    xytext=(best_t + 0.07, best_f2 - 0.06),
    fontsize=10, fontweight="bold", color="#37474F",
    arrowprops=dict(arrowstyle="-|>", color="#90A4AE", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E1", edgecolor="#FFB74D", lw=1),
)

ax.set_xlabel("Decision Threshold", fontsize=11, color="#37474F")
ax.set_ylabel("F2 Score", fontsize=11, color="#37474F")
ax.set_title("F2 Score vs Decision Threshold (Validation Set)", fontsize=12,
             fontweight="bold", color="#37474F", pad=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="lower left")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Best threshold: {best_t:.4f}")
print(f"Best F2 score:  {best_f2:.4f}")
print(f"Saved: {OUT_PATH}")
