#!/usr/bin/env python3
"""
Generate an improved AE threshold-selection plot for Section 12.6.2.

Shows the legitimate reconstruction-error distribution with:
  - Clear shaded regions for APPROVE / REVIEW / BLOCK zones
  - Annotated threshold lines with percentile labels
  - Inset or callout showing how many transactions fall in each zone
"""
from __future__ import annotations
import json, warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "docs" / "figures" / "thesis_diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# ── colours ──────────────────────────────────────────────────────────
C_BLUE   = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREY   = "#999999"
C_GREEN_LIGHT  = "#d4edda"   # approve zone fill
C_YELLOW_LIGHT = "#fff3cd"   # review zone fill
C_RED_LIGHT    = "#f8d7da"   # block zone fill
DPI = 200

# ── load data ────────────────────────────────────────────────────────
ae_thresh = json.loads((REPO / "artifacts" / "thresholds" / "ae_thresholds.json").read_text())
AE_REVIEW = ae_thresh["review"]
AE_BLOCK  = ae_thresh["block"]

# Use validation errors (threshold was selected on val set)
ae_val_errors = np.load(REPO / "artifacts" / "ae_errors" / "ae_val_errors.npy")

# Load val labels to get legit-only errors
import pandas as pd
val_df = pd.read_csv(REPO / "data" / "processed" / "val.csv")
y_val = val_df["Class"].values
legit_err = ae_val_errors[y_val == 0]

n_total = len(legit_err)
n_approve = int(np.sum(legit_err < AE_REVIEW))
n_review  = int(np.sum((legit_err >= AE_REVIEW) & (legit_err < AE_BLOCK)))
n_block   = int(np.sum(legit_err >= AE_BLOCK))

print(f"Legit transactions: {n_total}")
print(f"  APPROVE zone (< {AE_REVIEW:.3f}): {n_approve} ({100*n_approve/n_total:.1f}%)")
print(f"  REVIEW  zone ({AE_REVIEW:.3f} – {AE_BLOCK:.3f}): {n_review} ({100*n_review/n_total:.1f}%)")
print(f"  BLOCK   zone (≥ {AE_BLOCK:.3f}): {n_block} ({100*n_block/n_total:.2f}%)")

# ── plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))

# Histogram
upper = np.percentile(legit_err, 99.8)
bins = np.linspace(0, upper, 150)
counts, edges, patches = ax.hist(legit_err, bins=bins, color=C_BLUE,
                                  edgecolor="white", linewidth=0.3,
                                  density=True, alpha=0.75, zorder=2)

# Shade the three zones
ymax = ax.get_ylim()[1]
ax.axvspan(0, AE_REVIEW, alpha=0.12, color="green", zorder=1)
ax.axvspan(AE_REVIEW, AE_BLOCK, alpha=0.15, color=C_ORANGE, zorder=1)
ax.axvspan(AE_BLOCK, upper * 1.1, alpha=0.15, color="red", zorder=1)

# Threshold lines
ax.axvline(AE_REVIEW, color=C_ORANGE, linestyle="--", linewidth=2, zorder=3)
ax.axvline(AE_BLOCK, color="red", linestyle="-.", linewidth=2, zorder=3)

# Zone labels (positioned inside each zone)
zone_y = ymax * 0.88
ax.text(AE_REVIEW * 0.35, zone_y, "APPROVE\nzone",
        ha="center", va="top", fontsize=10, fontweight="bold",
        color="green", alpha=0.8)

review_mid = (AE_REVIEW + min(AE_BLOCK, upper)) / 2
ax.text(review_mid, zone_y, "REVIEW\nzone",
        ha="center", va="top", fontsize=10, fontweight="bold",
        color=C_ORANGE, alpha=0.9)

if AE_BLOCK < upper:
    ax.text(min(AE_BLOCK + (upper - AE_BLOCK) * 0.4, upper * 0.95), zone_y,
            "BLOCK\nzone",
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="red", alpha=0.8)

# Annotated threshold values
ax.annotate(f"p{ae_thresh['p_review']:.0f} = {AE_REVIEW:.3f}",
            xy=(AE_REVIEW, ymax * 0.65), xytext=(AE_REVIEW + 0.3, ymax * 0.72),
            fontsize=9, color=C_ORANGE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.2))

ax.annotate(f"p{ae_thresh['p_block']:.0f} = {AE_BLOCK:.3f}",
            xy=(AE_BLOCK, ymax * 0.3), xytext=(AE_BLOCK - 1.2, ymax * 0.55),
            fontsize=9, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2))

# Summary box
textstr = (
    f"APPROVE: {100*n_approve/n_total:.1f}% of legit\n"
    f"REVIEW:  {100*n_review/n_total:.1f}% of legit\n"
    f"BLOCK:   {100*n_block/n_total:.2f}% of legit"
)
props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=C_GREY, alpha=0.9)
ax.text(0.97, 0.95, textstr, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", horizontalalignment="right",
        bbox=props, family="monospace")

# Axis labels
ax.set_xlabel("Reconstruction Error", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Autoencoder Threshold Selection — Legitimate Transactions (Validation Set)",
             fontsize=11, fontweight="bold")
ax.set_xlim(0, upper * 1.05)

fig.tight_layout()
path = OUT / "ch12_ae_threshold_selection.png"
fig.savefig(path, dpi=DPI)
plt.close(fig)
print(f"\n[OK] saved → {path}")
