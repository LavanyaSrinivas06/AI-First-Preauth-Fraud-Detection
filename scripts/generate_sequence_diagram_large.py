#!/usr/bin/env python3
"""
Generate a large, high-resolution UML sequence diagram for the
pre-authorisation fraud detection flow.

BLACK & WHITE ONLY — suitable for formal thesis printing.

Output: docs/figures/thesis_diagrams/fig_9_2_sequence_diagram_large.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "docs" / "figures" / "thesis_diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# ── Black & white only ───────────────────────────────────────────────
BLACK   = "#000000"
DKGREY  = "#444444"
GREY    = "#888888"
LTGREY  = "#e0e0e0"
WHITE   = "#ffffff"
DPI = 200

# ── Actors (x positions) ────────────────────────────────────────────
actors = [
    ("Customer",           1),
    ("Checkout\nPage",     3.5),
    ("Fraud Detection\nAPI", 6),
    ("XGBoost\nModel",     8.5),
    ("Autoencoder",       11),
    ("Analyst\nDashboard",13.5),
    ("Payment\nGateway",  16),
    ("SQLite\nDatabase",  18.5),
]

# ── Figure setup ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 34))
ax.set_xlim(-0.5, 20.5)
ax.set_ylim(-1, 52)
ax.invert_yaxis()
ax.axis("off")
fig.patch.set_facecolor(WHITE)

FONT_BOLD = {"fontsize": 11, "fontfamily": "sans-serif", "fontweight": "bold"}
FONT_NOTE = {"fontsize": 9.5, "fontfamily": "sans-serif", "fontstyle": "italic"}

# ── Draw actor boxes + lifelines ─────────────────────────────────────
ACTOR_Y = 0
LIFELINE_END = 50

for name, x in actors:
    box = mpatches.FancyBboxPatch(
        (x - 1.1, ACTOR_Y - 0.6), 2.2, 1.4,
        boxstyle="round,pad=0.15", facecolor=WHITE,
        edgecolor=BLACK, linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x, ACTOR_Y + 0.1, name, ha="center", va="center",
            **FONT_BOLD, color=BLACK)
    ax.plot([x, x], [ACTOR_Y + 0.8, LIFELINE_END], color=GREY,
            linestyle="--", linewidth=0.8, zorder=0)


def arrow(x1, x2, y, label, style="-|>", dashed=False, fontsize=10):
    ls = "--" if dashed else "-"
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle=style, color=BLACK, lw=1.4, linestyle=ls),
    )
    mid = (x1 + x2) / 2
    ax.text(mid, y - 0.3, label, ha="center", va="bottom",
            fontsize=fontsize, color=BLACK, fontfamily="sans-serif")


def note_box(x, y, text, width=3.5, height=0.7):
    box = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.12", facecolor=LTGREY,
        edgecolor=DKGREY, linewidth=1
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", **FONT_NOTE, color=BLACK)


def alt_box(y_start, y_end, label):
    w = 19.5; x0 = 0
    rect = mpatches.FancyBboxPatch(
        (x0, y_start), w, y_end - y_start,
        boxstyle="round,pad=0.1", facecolor="none",
        edgecolor=BLACK, linewidth=1.5
    )
    ax.add_patch(rect)
    tag = mpatches.FancyBboxPatch(
        (x0, y_start), 2.2, 0.7,
        boxstyle="round,pad=0.08", facecolor=LTGREY,
        edgecolor=BLACK, linewidth=1
    )
    ax.add_patch(tag)
    ax.text(x0 + 1.1, y_start + 0.35, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color=BLACK)


def divider(y, label, x0=0.2, x1=19.3):
    ax.plot([x0, x1], [y, y], color=DKGREY, linewidth=1, linestyle="--")
    ax.text(x0 + 0.3, y + 0.15, label, fontsize=9.5, color=DKGREY,
            fontstyle="italic")


# ── Shorthand ────────────────────────────────────────────────────────
xCust = 1;  xPage = 3.5;  xAPI = 6;  xXGB = 8.5
xAE   = 11; xDash = 13.5; xPG  = 16; xDB  = 18.5

# ── Title ────────────────────────────────────────────────────────────
ax.text(10, -0.8, "Pre-Authorisation Fraud Detection — Sequence Diagram",
        ha="center", va="center", fontsize=16, fontweight="bold",
        fontfamily="sans-serif", color=BLACK)

# ═══════════════════════════════════════════════════════════════════
# STEPS 1–4: Initial request flow
# ═══════════════════════════════════════════════════════════════════
y = 2.5;  arrow(xCust, xPage, y, "1.  Click Checkout")
y = 4.0;  arrow(xPage, xAPI,  y, "2.  POST /preauth/decision (JSON)")
y = 5.5;  arrow(xAPI,  xXGB,  y, "3.  Score Transaction")
y = 7.0;  arrow(xXGB,  xAPI,  y, "4.  Return XGBoost Score (p_xgb)", style="<|-")

# ═══════════════════════════════════════════════════════════════════
# ALT FRAME — main decision routing
# ═══════════════════════════════════════════════════════════════════
alt_box(8.5, 42, "alt")

# ── Path A: APPROVE ──────────────────────────────────────────────
y = 9.5;  note_box(3, y, "p_xgb < threshold_low", width=3.5)
y = 10.8
ax.text(xAPI + 0.4, y - 0.15, "5a. APPROVE — Transaction is Safe",
        ha="left", fontsize=10.5, color=BLACK, fontweight="bold")
y = 12.2; arrow(xAPI, xPG, y, "6a.  Send to Payment Gateway")
y = 13.6; arrow(xPG, xPage, y, "7a.  Payment Confirmed")

# ── Divider ──────────────────────────────────────────────────────
divider(15.0, "[HIGH RISK]  p_xgb > threshold_high")

# ── Path B: BLOCK ────────────────────────────────────────────────
y = 16.2; arrow(xAPI, xPage, y, "5b.  BLOCK — Stop Payment")

# ── Divider ──────────────────────────────────────────────────────
divider(17.8, "[UNCERTAIN]  threshold_low ≤ p_xgb ≤ threshold_high  →  Gray Zone")

# ── Path C: Gray zone — Autoencoder ──────────────────────────────
y = 19.5; arrow(xAPI, xAE, y, "5c.  Check How Unusual (Autoencoder)")
y = 21.0; arrow(xAE,  xAPI, y, "6c.  Return Reconstruction Error", style="<|-")

# ── Nested ALT: AE decision ─────────────────────────────────────
alt_box(22.5, 35.5, "alt")

# AE extreme → BLOCK
y = 23.8; note_box(3, y, "ae_error > block_threshold", width=3.8)
y = 25.0; arrow(xAPI, xPage, y, "7ca.  BLOCK — Stop Payment (AE extreme)")

divider(26.5, "[AE above review threshold]  →  REVIEW")

# AE review → analyst
y = 28.0; note_box(3, y, "ae_error > review_threshold", width=3.8)
y = 29.2; arrow(xAPI, xDash, y, "7cb.  Send for Review")
y = 30.8; arrow(xDash, xAPI, y, "8.  Analyst Decision (APPROVE / BLOCK)", style="<|-")

divider(32.0, "[AE below review threshold]  →  APPROVE")

# AE normal → approve
y = 33.5; note_box(3, y, "ae_error < review_threshold", width=3.8)
y = 34.5; arrow(xAPI, xPG, y, "7cc.  APPROVE — AE confirms safe")

# ═══════════════════════════════════════════════════════════════════
# STEP 9: Store outcome + feedback
# ═══════════════════════════════════════════════════════════════════
y = 37.5; arrow(xAPI, xDB, y, "9.  Store Outcome → Feedback Loop")
y = 39.0; note_box(xDB - 2, y, "Feed back to retrain XGBoost", width=4.2)
y = 40.2; arrow(xDB, xXGB, y, "Retrain signal", dashed=True)

# ── Footer ───────────────────────────────────────────────────────
y = 44
ax.text(10, y,
        "Fraud is detected before payment goes through — scored automatically.",
        ha="center", va="center", fontsize=12, fontstyle="italic",
        color=GREY, fontfamily="sans-serif")
ax.text(10, y + 1,
        "Gray-zone cases are escalated via autoencoder + analyst review.",
        ha="center", va="center", fontsize=12, fontstyle="italic",
        color=GREY, fontfamily="sans-serif")

# ── Save ─────────────────────────────────────────────────────────
path = OUT / "fig_9_2_sequence_diagram_large.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)
print(f"[OK] saved → {path}")
