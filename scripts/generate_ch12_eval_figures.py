#!/usr/bin/env python3
"""
Chapter 12 — Evaluation Figures for Master's Thesis
====================================================
Generates all eight evaluation plots from held-out test set artifacts.

Inputs (all pre-computed):
    data/processed/test.csv              – preprocessed test features + Class
    data/processed/val.csv               – preprocessed validation features + Class
    data/processed/test_raw.csv          – raw test rows (contains Time column)
    artifacts/models/xgb_model.pkl       – trained XGBoost model
    artifacts/ae_errors/ae_test_errors.npy   – AE reconstruction errors (test)
    artifacts/ae_errors/ae_val_errors.npy    – AE reconstruction errors (val)
    artifacts/thresholds/ae_thresholds.json  – operational AE thresholds

Outputs → docs/figures/thesis_diagrams/ch12_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

# ── paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "thesis_diagrams"
OUT.mkdir(parents=True, exist_ok=True)

TEST_CSV   = ROOT / "data" / "processed" / "test.csv"
VAL_CSV    = ROOT / "data" / "processed" / "val.csv"
RAW_TEST   = ROOT / "data" / "processed" / "test_raw.csv"
XGB_MODEL  = ROOT / "artifacts" / "models" / "xgb_model.pkl"
AE_TEST_ERR = ROOT / "artifacts" / "ae_errors" / "ae_test_errors.npy"
AE_VAL_ERR  = ROOT / "artifacts" / "ae_errors" / "ae_val_errors.npy"
AE_THRESH   = ROOT / "artifacts" / "thresholds" / "ae_thresholds.json"
XGB_METRICS = ROOT / "artifacts" / "xgb_metrics.json"

# ── load data ───────────────────────────────────────────────────────────
test_df = pd.read_csv(TEST_CSV)
y_test  = test_df["Class"].to_numpy()
X_test  = test_df.drop(columns=["Class"])

val_df  = pd.read_csv(VAL_CSV)
y_val   = val_df["Class"].to_numpy()
X_val   = val_df.drop(columns=["Class"])

raw_test = pd.read_csv(RAW_TEST)
time_col = raw_test["Time"].to_numpy()

xgb = joblib.load(XGB_MODEL)
y_proba_xgb_test = xgb.predict_proba(X_test)[:, 1]
y_proba_xgb_val  = xgb.predict_proba(X_val)[:, 1]

ae_errors_test = np.load(AE_TEST_ERR).astype(float)
ae_errors_val  = np.load(AE_VAL_ERR).astype(float)

with open(AE_THRESH) as f:
    ae_thresh = json.load(f)
AE_REVIEW = ae_thresh["review"]   # 0.6916
AE_BLOCK  = ae_thresh["block"]    # 4.8956

with open(XGB_METRICS) as f:
    xgb_meta = json.load(f)
XGB_THRESHOLD = xgb_meta["threshold"]  # 0.307

# decision-engine thresholds
XGB_T_LOW  = 0.05
XGB_T_HIGH = 0.80

# ── shared style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         200,
    "savefig.dpi":        200,
})

DPI = 200

# ── thesis colour palette (blue + orange only) ─────────────────────────
C_BLUE    = "#1f77b4"   # primary / legitimate / model curve
C_ORANGE  = "#ff7f0e"   # secondary / fraud / thresholds
C_GREY    = "#7f7f7f"   # baselines / reference lines

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 1 — XGBoost ROC Curve (Test Set)
# ═════════════════════════════════════════════════════════════════════════
fpr, tpr, _ = roc_curve(y_test, y_proba_xgb_test)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=C_BLUE, linewidth=1.8,
        label=f"XGBoost (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color=C_GREY, linestyle="--", linewidth=0.8,
        label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — XGBoost (Test Set)")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])

path = OUT / "ch12_xgb_roc_test.png"
fig.savefig(path, dpi=DPI)
plt.close(fig)
print(f"[1/8] {path.name}  (AUC={roc_auc:.4f})")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 2 — XGBoost Precision-Recall Curve (Test Set)
# ═════════════════════════════════════════════════════════════════════════
prec, rec, _ = precision_recall_curve(y_test, y_proba_xgb_test)
ap = average_precision_score(y_test, y_proba_xgb_test)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, color=C_ORANGE, linewidth=1.8,
        label=f"XGBoost (AP = {ap:.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curve — XGBoost (Test Set)")
ax.legend(loc="lower left", fontsize=10)
ax.set_xlim([-0.01, 1.05])
ax.set_ylim([-0.01, 1.05])

path = OUT / "ch12_xgb_pr_test.png"
fig.savefig(path, dpi=DPI)
plt.close(fig)
print(f"[2/8] {path.name}  (AP={ap:.4f})")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 3 — XGBoost Confusion Matrix (Test Set, threshold = 0.307)
# ═════════════════════════════════════════════════════════════════════════
y_pred_xgb = (y_proba_xgb_test >= XGB_THRESHOLD).astype(int)
cm = confusion_matrix(y_test, y_pred_xgb)

fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(cm, cmap="Blues", aspect="auto")

labels = ["Legitimate", "Fraud"]
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(f"Confusion Matrix — XGBoost (Test, t = {XGB_THRESHOLD})")

for i in range(2):
    for j in range(2):
        colour = "white" if cm[i, j] > 20_000 else "black"
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=colour)

path = OUT / "ch12_xgb_confusion_matrix.png"
fig.savefig(path, dpi=DPI)
plt.close(fig)
print(f"[3/8] {path.name}  CM={cm.ravel()}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 4 — F1 vs Decision Threshold (Validation Set)
# ═════════════════════════════════════════════════════════════════════════
thresholds = np.linspace(0.01, 0.99, 200)
f1_scores = [f1_score(y_val, (y_proba_xgb_val >= t).astype(int))
             for t in thresholds]

best_idx = int(np.argmax(f1_scores))
best_t   = thresholds[best_idx]
best_f1  = f1_scores[best_idx]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(thresholds, f1_scores, color=C_BLUE, linewidth=1.5)
ax.axvline(best_t, color=C_ORANGE, linestyle="--", linewidth=1,
           label=f"Best threshold = {best_t:.3f}  (F1 = {best_f1:.3f})")
ax.scatter([best_t], [best_f1], color=C_ORANGE, zorder=5, s=50)
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("F1 Score")
ax.set_title("F1 vs Decision Threshold — XGBoost (Validation Set)")
ax.legend(fontsize=9, loc="lower right")

path = OUT / "ch12_xgb_f1_vs_threshold_val.png"
fig.savefig(path, dpi=DPI)
plt.close(fig)
print(f"[4/8] {path.name}  best_t={best_t:.3f}  best_f1={best_f1:.3f}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 5 — AE Reconstruction Error Distribution (Legit vs Fraud, log)
# ═════════════════════════════════════════════════════════════════════════
legit_err = ae_errors_test[y_test == 0]
fraud_err = ae_errors_test[y_test == 1]

upper_clip = np.percentile(ae_errors_test, 99.9)
bins = np.logspace(np.log10(max(ae_errors_test.min(), 1e-3)),
                   np.log10(upper_clip), 100)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(legit_err, bins=bins, alpha=0.7, color=C_BLUE,
        label=f"Legitimate (n = {len(legit_err):,})", density=True)
ax.hist(fraud_err, bins=bins, alpha=0.85, color=C_ORANGE,
        label=f"Fraud (n = {len(fraud_err):,})", density=True)

ax.axvline(AE_REVIEW, color=C_ORANGE, linestyle="--", linewidth=1.2,
           label=f"Review threshold = {AE_REVIEW:.3f}")
ax.axvline(AE_BLOCK, color=C_ORANGE, linestyle="-.", linewidth=1.2,
           label=f"Block threshold = {AE_BLOCK:.3f}")

ax.set_xscale("log")
ax.set_xlabel("Reconstruction Error (log scale)")
ax.set_ylabel("Density")
ax.set_title("Autoencoder Reconstruction Error — Legitimate vs Fraud (Test Set)")
ax.legend(fontsize=8, loc="upper right")

path = OUT / "ch12_ae_error_dist_legit_vs_fraud.png"
fig.savefig(path, dpi=DPI)
plt.close(fig)
print(f"[5/8] {path.name}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 6 — AE Reconstruction Error (Legit Only) with Thresholds
# ═════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
upper_legit = np.percentile(legit_err, 99.8)
bins_legit = np.linspace(0, upper_legit, 120)

ax.hist(legit_err, bins=bins_legit, color=C_BLUE, edgecolor="white",
        linewidth=0.3, density=True)

ax.axvline(AE_REVIEW, color=C_ORANGE, linestyle="--", linewidth=1.4,
           label=f"Review threshold (p{ae_thresh['p_review']:.0f}) = {AE_REVIEW:.3f}")
ax.axvline(AE_BLOCK, color=C_ORANGE, linestyle="-.", linewidth=1.4,
           label=f"Block threshold (p{ae_thresh['p_block']:.0f}) = {AE_BLOCK:.3f}")

ax.set_xlabel("Reconstruction Error")
ax.set_ylabel("Density")
ax.set_title("Autoencoder Reconstruction Error — Legitimate Transactions Only (Test Set)")
ax.legend(fontsize=9, loc="upper right")

path = OUT / "ch12_ae_error_dist_legit_only.png"
fig.savefig(path, dpi=DPI)
plt.close(fig)
print(f"[6/8] {path.name}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 7 — AE Reconstruction Error Over Time (Transaction Index)
# ═════════════════════════════════════════════════════════════════════════
# time_col is the raw "Time" feature (seconds from first txn in dataset)
# Convert to hours for readability
time_hours = (time_col - time_col.min()) / 3600.0

fig, ax = plt.subplots(figsize=(8, 4.5))

# plot legitimate first (lighter), then fraud on top
legit_mask = y_test == 0
fraud_mask = y_test == 1

ax.scatter(time_hours[legit_mask], ae_errors_test[legit_mask],
           s=1, alpha=0.15, color=C_BLUE, label="Legitimate", rasterized=True)
ax.scatter(time_hours[fraud_mask], ae_errors_test[fraud_mask],
           s=18, alpha=0.9, color=C_ORANGE, marker="x", label="Fraud",
           zorder=5)

ax.axhline(AE_REVIEW, color=C_ORANGE, linestyle="--", linewidth=1,
           label=f"Review = {AE_REVIEW:.3f}")
ax.axhline(AE_BLOCK, color=C_ORANGE, linestyle="-.", linewidth=1,
           label=f"Block = {AE_BLOCK:.3f}")

ax.set_yscale("log")
ax.set_xlabel("Time (hours from start of test window)")
ax.set_ylabel("Reconstruction Error (log scale)")
ax.set_title("Autoencoder Reconstruction Error Over Time (Test Set)")
ax.legend(fontsize=8, loc="upper right", markerscale=2)

path = OUT / "ch12_ae_error_over_time.png"
fig.savefig(path, dpi=DPI)
plt.close(fig)
print(f"[7/8] {path.name}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Hybrid Decision Engine Triage Distribution
# ═════════════════════════════════════════════════════════════════════════
# Replicate the decision logic from run_decision_engine_eval.py
decisions = np.empty(len(y_test), dtype=object)

for i in range(len(y_test)):
    p   = y_proba_xgb_test[i]
    err = ae_errors_test[i]

    # AE hard-block gate
    if err >= AE_BLOCK:
        decisions[i] = "BLOCK"
    elif p >= XGB_T_HIGH:
        decisions[i] = "BLOCK"
    elif p <= XGB_T_LOW:
        decisions[i] = "APPROVE"
    else:
        decisions[i] = "REVIEW"

# counts
cats = ["APPROVE", "REVIEW", "BLOCK"]
total_counts = [int((decisions == c).sum()) for c in cats]
fraud_counts = [int(((decisions == c) & (y_test == 1)).sum()) for c in cats]
legit_counts = [t - f for t, f in zip(total_counts, fraud_counts)]

x = np.arange(len(cats))
width = 0.55

fig, ax = plt.subplots(figsize=(6, 4.8))
bars_l = ax.bar(x, legit_counts, width, label="Legitimate", color=C_BLUE)
bars_f = ax.bar(x, fraud_counts, width, bottom=legit_counts,
                label="Fraud", color=C_ORANGE, edgecolor="black", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=11)
ax.set_ylabel("Transaction Count")
ax.set_title("Hybrid Decision Engine — Triage Distribution (Test Set)")
ax.legend(fontsize=10, loc="upper right")

# annotate totals and fraud on each bar
for i, (tot, fr) in enumerate(zip(total_counts, fraud_counts)):
    # total count above bar
    ax.text(i, tot + tot * 0.08, f"n = {tot:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    # fraud count inside bar if visible
    if fr > 0:
        ax.text(i, legit_counts[i] + fr / 2, f"fraud = {fr}",
                ha="center", va="center", fontsize=8, fontweight="bold")

ax.set_yscale("log")
ax.set_ylim(1, 300_000)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

path = OUT / "ch12_hybrid_triage_distribution.png"
fig.savefig(path, dpi=DPI)
plt.close(fig)
print(f"[8/8] {path.name}  {dict(zip(cats, total_counts))}")

# ═════════════════════════════════════════════════════════════════════════
print("\n✅  All 8 Chapter 12 figures generated in:")
print(f"   {OUT}")
