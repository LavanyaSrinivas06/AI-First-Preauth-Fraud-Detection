#!/usr/bin/env python3
"""
Chapter 8 — Hybrid Fraud Decision Engine  (horizontal staged diagram).

Clean, border-only academic layout (no filled shapes):
  LEFT  ─ Stage 1: XGBoost triage  (3 risk bands)
  RIGHT ─ Stage 2: Autoencoder anomaly check (gray-zone only)
  OUTCOMES: Approve / Review / Block

All boxes are white-fill + black border — no colour fills.
No numeric threshold values shown (thesis text provides them).
AE extreme → BLOCK is dashed (very rare).

Matches  api/routers/preauth.py  exactly.
Serif font, 300 dpi.
Output: docs/figures/thesis_diagrams/fig_8_1_hybrid_decision_flow.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

OUT = Path("docs/figures/thesis_diagrams")
OUT.mkdir(parents=True, exist_ok=True)

FONT = "serif"

# ── Palette (white fill + light coloured borders) ─────────────────
BG       = "#FFFFFF"
BLACK    = "#1a1a1a"
GREY     = "#555555"
LGREY    = "#999999"
WHITE    = "#FFFFFF"

# Soft coloured borders (no fills — just outlines)
EC_PROC  = "#5b7ea8"   # blue-grey  — process / input boxes
EC_LOW   = "#4caf50"   # soft green — low risk / approve
EC_GRAY  = "#f5a623"   # warm amber — gray zone / review
EC_HIGH  = "#d9534f"   # soft red   — high risk / block
EC_AE    = "#7e57c2"   # muted purple — autoencoder

STAGE_BG = "#f9f9f9"   # barely-visible stage background


def _box(ax, x, y, w, h, text, fs=9, fw="normal", ec=BLACK, lw=1.0,
         ls=1.3):
    """Rounded box centred at (x, y) — white fill, black border."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.06",
        facecolor=WHITE, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, fontfamily=FONT, fontweight=fw,
            color=BLACK, zorder=4, linespacing=ls)


def _stage_bg(ax, x, y, w, h, label, label_y_offset=0):
    """Faint background rectangle for a stage."""
    from matplotlib.patches import Rectangle
    rect = Rectangle((x, y), w, h, facecolor=STAGE_BG, edgecolor=LGREY,
                      linewidth=0.6, linestyle="--", zorder=1, alpha=0.55)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h + 0.15 + label_y_offset, label,
            ha="center", va="bottom",
            fontsize=11, fontfamily=FONT, fontweight="bold",
            color=GREY, zorder=4)


def _arrow(ax, x1, y1, x2, y2, label="", lw=1.0, color=BLACK, ls="-",
           label_above=True, fs=7.5):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", color=color, lw=lw,
        linestyle=ls, zorder=2, mutation_scale=12,
    )
    ax.add_patch(a)
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        offset = 0.18 if label_above else -0.18
        va = "bottom" if label_above else "top"
        ax.text(mx, my + offset, label, ha="center", va=va,
                fontsize=fs, fontfamily=FONT, fontstyle="italic",
                color=GREY, zorder=4)


def generate():
    fig, ax = plt.subplots(figsize=(16, 8.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    # ── Coordinates ─────────────────────────────────────────────
    y_low  = 5.5     # top — low risk → approve
    y_gray = 3.5     # middle — gray zone → AE
    y_high = 1.5     # bottom — high risk → block

    x_txn     = 1.0
    x_xgb     = 4.5
    x_band    = 7.5
    x_ae      = 10.8
    x_out     = 14.0

    # ── Stage backgrounds ───────────────────────────────────────
    _stage_bg(ax, 2.8, 0.5, 5.8, 6.3,
              "Stage 1: XGBoost Triage")
    _stage_bg(ax, 9.2, 0.5, 3.5, 6.3,
              "Stage 2: Autoencoder Check")

    # ════════════════════════════════════════════════════════════
    #  TRANSACTION INPUT
    # ════════════════════════════════════════════════════════════
    _box(ax, x_txn, y_gray, 1.6, 1.2,
         "Incoming\nTransaction", fs=9.5, fw="bold", ec=EC_PROC)

    _arrow(ax, x_txn + 0.85, y_gray, x_xgb - 1.1, y_gray,
           label="102 features", fs=7)

    # ════════════════════════════════════════════════════════════
    #  XGBOOST SCORING
    # ════════════════════════════════════════════════════════════
    _box(ax, x_xgb, y_gray, 2.0, 1.3,
         "XGBoost\nClassifier\n→ p_fraud", fs=9.5, fw="bold", ec=EC_PROC)

    # ════════════════════════════════════════════════════════════
    #  THREE RISK BANDS
    # ════════════════════════════════════════════════════════════
    bw, bh = 2.2, 0.85

    _box(ax, x_band, y_low, bw, bh,
         "Low Risk", fs=9, fw="bold", ec=EC_LOW)

    _box(ax, x_band, y_gray, bw, bh,
         "Gray Zone", fs=9, fw="bold", ec=EC_GRAY)

    _box(ax, x_band, y_high, bw, bh,
         "High Risk", fs=9, fw="bold", ec=EC_HIGH)

    # Arrows from XGBoost → bands
    _arrow(ax, x_xgb + 1.05, y_gray + 0.45, x_band - bw / 2, y_low,
           lw=0.9)
    _arrow(ax, x_xgb + 1.05, y_gray, x_band - bw / 2, y_gray,
           lw=0.9)
    _arrow(ax, x_xgb + 1.05, y_gray - 0.45, x_band - bw / 2, y_high,
           lw=0.9)

    # ════════════════════════════════════════════════════════════
    #  AE CHECK (only for gray zone)
    # ════════════════════════════════════════════════════════════
    _arrow(ax, x_band + bw / 2, y_gray, x_ae - 1.15, y_gray,
           label="uncertain", fs=7)

    _box(ax, x_ae, y_gray, 2.1, 1.3,
         "Autoencoder\nReconstruction\nError", fs=9, fw="bold", ec=EC_AE)

    # AE bucket labels below
    ax.text(x_ae, y_gray - 0.85,
            "normal  /  elevated  /  extreme",
            ha="center", va="top", fontsize=7.5, fontfamily=FONT,
            color=GREY, linespacing=1.4, zorder=4)

    # ════════════════════════════════════════════════════════════
    #  OUTCOMES
    # ════════════════════════════════════════════════════════════
    ow, oh = 2.0, 0.75

    _box(ax, x_out, y_low, ow, oh, "APPROVE",
         fs=11, fw="bold", lw=1.5, ec=EC_LOW)

    _box(ax, x_out, y_gray, ow, oh, "REVIEW",
         fs=11, fw="bold", lw=1.5, ec=EC_GRAY)

    _box(ax, x_out, y_high, ow, oh, "BLOCK",
         fs=11, fw="bold", lw=1.5, ec=EC_HIGH)

    # ── Routing arrows to outcomes ──────────────────────────────

    # Low risk → APPROVE
    _arrow(ax, x_band + bw / 2, y_low, x_out - ow / 2, y_low,
           lw=1.1, color=EC_LOW)

    # High risk → BLOCK
    _arrow(ax, x_band + bw / 2, y_high, x_out - ow / 2, y_high,
           lw=1.1, color=EC_HIGH)

    # AE normal / elevated → REVIEW
    _arrow(ax, x_ae + 1.1, y_gray, x_out - ow / 2, y_gray,
           label="normal /\nelevated", fs=7, label_above=True,
           color=EC_GRAY)

    # AE extreme → BLOCK (down-right L-shape, DASHED — very rare)
    ae_right = x_ae + 1.1
    ax.plot([ae_right, ae_right + 0.5, ae_right + 0.5, x_out - ow / 2],
            [y_gray - 0.35, y_gray - 0.35, y_high, y_high],
            color=LGREY, lw=0.9, linestyle=(0, (4, 3)),
            zorder=2, solid_capstyle="round")
    ax.annotate("", xy=(x_out - ow / 2, y_high + 0.02),
                xytext=(x_out - ow / 2 - 0.01, y_high + 0.02),
                arrowprops=dict(arrowstyle="-|>", color=LGREY, lw=0.9,
                                linestyle="dashed"),
                zorder=2)
    ax.text(ae_right + 0.7, (y_gray - 0.35 + y_high) / 2,
            "extreme\n(very rare)",
            ha="left", va="center", fontsize=7, fontfamily=FONT,
            fontstyle="italic", color=LGREY, zorder=4)

    # ════════════════════════════════════════════════════════════
    #  Bottom note
    # ════════════════════════════════════════════════════════════
    ax.text(8.0, -0.1,
            "Gray-zone transactions are never auto-approved — "
            "outcomes are REVIEW or BLOCK only.    "
            "Source: api/routers/preauth.py",
            ha="center", va="top", fontsize=7.5, fontfamily=FONT,
            color=LGREY, zorder=4)

    # ── Limits ──────────────────────────────────────────────────
    ax.set_xlim(-0.3, 15.8)
    ax.set_ylim(-0.6, 8.2)

    fig.tight_layout()
    out_path = OUT / "fig_8_1_hybrid_decision_flow.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    generate()


