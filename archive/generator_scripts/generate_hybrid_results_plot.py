#!/usr/bin/env python3
"""
Generate a comprehensive hybrid system results figure for Section 12.7.

Shows a stacked/grouped bar chart with:
  - APPROVE / REVIEW / BLOCK counts
  - Fraud vs Legitimate breakdown within each bucket
  - Capture rates annotated
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "docs" / "figures" / "thesis_diagrams"
OUT.mkdir(parents=True, exist_ok=True)

C_BLUE   = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREY   = "#999999"
DPI = 200

# ── Real numbers from decision_engine_metrics.json ───────────────────
# Decision     Total    Fraud   Legit
data = {
    "APPROVE": {"total": 42529, "fraud": 8,  "legit": 42521},
    "REVIEW":  {"total": 44,    "fraud": 7,  "legit": 37},
    "BLOCK":   {"total": 149,   "fraud": 37, "legit": 112},
}
total_fraud = 52

categories = ["APPROVE", "REVIEW", "BLOCK"]
fraud_counts = [data[c]["fraud"] for c in categories]
legit_counts = [data[c]["legit"] for c in categories]
totals       = [data[c]["total"] for c in categories]

# ── Figure: Two-panel layout ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                gridspec_kw={"width_ratios": [2, 1.2]})

# ── Panel 1: Stacked bar chart (log scale for visibility) ───────────
x = np.arange(len(categories))
width = 0.5

bars_legit = ax1.bar(x, legit_counts, width, label="Legitimate", color=C_BLUE, alpha=0.8)
bars_fraud = ax1.bar(x, fraud_counts, width, bottom=legit_counts,
                     label="Fraud", color=C_ORANGE, alpha=0.9)

ax1.set_yscale("log")
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=12, fontweight="bold")
ax1.set_ylabel("Transaction Count (log scale)", fontsize=11)
ax1.set_title("Hybrid Triage — Fraud vs Legitimate", fontsize=12, fontweight="bold")
ax1.legend(fontsize=10, loc="upper right")

# Annotate totals above each bar
for i, cat in enumerate(categories):
    total = totals[i]
    fraud = fraud_counts[i]
    # Total count label
    ax1.text(i, total * 1.15, f"n = {total:,}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
    # Fraud count inside the orange part
    if fraud > 0:
        y_pos = legit_counts[i] + fraud / 2
        ax1.text(i, y_pos, f"{fraud} fraud",
                 ha="center", va="center", fontsize=8, fontweight="bold",
                 color="white" if fraud > 3 else C_ORANGE)

ax1.set_ylim(1, max(totals) * 3)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

# ── Panel 2: Fraud capture funnel ────────────────────────────────────
labels = ["Missed\n(APPROVE)", "Flagged\n(REVIEW)", "Blocked\n(BLOCK)"]
values = [8, 7, 37]
colors = [C_GREY, C_ORANGE, C_BLUE]

bars2 = ax2.barh(labels, values, color=colors, edgecolor="white", height=0.55)

# Percentage labels
for bar, val in zip(bars2, values):
    pct = 100 * val / total_fraud
    ax2.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
             f"{val}  ({pct:.1f}%)",
             va="center", fontsize=10, fontweight="bold")

ax2.set_xlabel("Fraud Transactions (of 52 total)", fontsize=11)
ax2.set_title("Fraud Capture Breakdown", fontsize=12, fontweight="bold")
ax2.set_xlim(0, 48)
ax2.invert_yaxis()

# Capture rate annotation box
textstr = (
    f"Flagged capture: {100*(7+37)/52:.1f}%\n"
    f"Auto-block rate: {100*37/52:.1f}%\n"
    f"Missed:  {100*8/52:.1f}%"
)
props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=C_GREY, alpha=0.9)
ax2.text(0.95, 0.05, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment="bottom", horizontalalignment="right",
         bbox=props, family="monospace")

fig.tight_layout(w_pad=3)
path = OUT / "ch12_hybrid_system_results.png"
fig.savefig(path, dpi=DPI)
plt.close(fig)
print(f"[OK] saved → {path}")
