#!/usr/bin/env python3
"""Generate Chapter 12 evaluation plots for thesis.

Plots produced (all monochrome / minimal colour):
  1. XGBoost confusion matrix heatmap (test set)
  2. Hybrid decision-engine triage distribution (stacked bar)
  3. AE reconstruction-error distribution overlay (legit vs fraud)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = Path("docs/figures/thesis_diagrams")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── shared style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ═══════════════════════════════════════════════════════════════════════
# 1.  XGBoost Confusion Matrix – Test Set
# ═══════════════════════════════════════════════════════════════════════
cm = np.array([[42_664, 6],
               [10,    42]])  # from artifacts/xgb_metrics.json -> test

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Greys", aspect="auto")

labels = ["Legitimate", "Fraud"]
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
ax.set_title("XGBoost Confusion Matrix — Test Set", fontsize=12, fontweight="bold")

for i in range(2):
    for j in range(2):
        colour = "white" if cm[i, j] > 20_000 else "black"
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=colour)

fig.tight_layout()
path = OUT_DIR / "ch12_xgb_confusion_matrix.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {path}")

# ═══════════════════════════════════════════════════════════════════════
# 2.  Hybrid Decision-Engine Triage Distribution
# ═══════════════════════════════════════════════════════════════════════
# From decision_engine_metrics.json
categories = ["APPROVE", "REVIEW", "BLOCK"]
total_counts = [42_529, 44, 149]
fraud_counts = [8, 7, 37]
legit_counts = [c - f for c, f in zip(total_counts, fraud_counts)]

x = np.arange(len(categories))
width = 0.55

fig, ax = plt.subplots(figsize=(6, 4.5))
bars_legit = ax.bar(x, legit_counts, width, label="Legitimate", color="#555555")
bars_fraud = ax.bar(x, fraud_counts, width, bottom=legit_counts,
                    label="Fraud", color="#cccccc", edgecolor="black", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylabel("Transaction Count", fontsize=11)
ax.set_title("Hybrid Decision-Engine — Triage Distribution (Test Set)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")

# annotate fraud counts on top
for i, (f, l) in enumerate(zip(fraud_counts, legit_counts)):
    ax.text(i, l + f + 200, f"fraud={f}", ha="center", va="bottom", fontsize=9,
            fontweight="bold")

# use log scale for y so REVIEW/BLOCK bars are visible
ax.set_yscale("log")
ax.set_ylim(1, 200_000)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

fig.tight_layout()
path = OUT_DIR / "ch12_hybrid_triage_distribution.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {path}")

# ═══════════════════════════════════════════════════════════════════════
# 3.  AE Reconstruction Error — Legit vs Fraud Overlay
# ═══════════════════════════════════════════════════════════════════════
import pandas as pd

test_csv = Path("data/processed/test.csv")
ae_errors_path = Path("artifacts/ae_errors/ae_test_errors.npy")

if test_csv.exists() and ae_errors_path.exists():
    df = pd.read_csv(test_csv)
    ae_err = np.load(ae_errors_path)

    legit_err = ae_err[df["Class"] == 0]
    fraud_err = ae_err[df["Class"] == 1]

    # Operational thresholds
    review_t = 0.6916
    block_t = 4.8956

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # clip upper range for readability
    upper = np.percentile(ae_err, 99.5)
    bins = np.linspace(0, upper, 80)

    ax.hist(legit_err, bins=bins, alpha=0.7, color="#888888",
            label=f"Legitimate (n={len(legit_err):,})", density=True)
    ax.hist(fraud_err, bins=bins, alpha=0.8, color="#000000",
            label=f"Fraud (n={len(fraud_err)})", density=True)

    ax.axvline(review_t, color="black", linestyle="--", linewidth=1.2,
               label=f"Review threshold = {review_t}")
    ax.axvline(block_t, color="black", linestyle="-.", linewidth=1.2,
               label=f"Block threshold = {block_t}")

    ax.set_xlabel("Reconstruction Error", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Autoencoder Reconstruction Error — Test Set",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    path = OUT_DIR / "ch12_ae_error_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")
else:
    print("[SKIP] AE plot – missing test.csv or ae_test_errors.npy")

print("\n[DONE] All Ch.12 plots generated.")
