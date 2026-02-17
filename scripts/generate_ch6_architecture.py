#!/usr/bin/env python3
"""
Generate Chapter 6 – Conceptual Architecture of the Supervised
Gradient Boosting Model (generic, light palette, thesis-quality).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("docs/figures/thesis_diagrams")
OUT.mkdir(parents=True, exist_ok=True)

FONT = "DejaVu Sans"

# ── Light professional palette ──────────────────────────────────────
BG        = "#FFFFFF"
C_INPUT   = "#E8EAF6";  C_INPUT_E = "#7986CB"
C_TREE_BG = "#F5F7FF";  C_TREE    = "#D1D9F0";  C_TREE_E = "#7986CB"
C_SUM     = "#E0F2F1";  C_SUM_E   = "#4DB6AC"
C_SIGMOID = "#EDE7F6";  C_SIG_E   = "#9575CD"
C_THRESH  = "#FFF8E1";  C_THRESH_E= "#FFB74D"
C_LEGIT   = "#E8F5E9";  C_LEGIT_E = "#66BB6A"
C_FRAUD   = "#FFEBEE";  C_FRAUD_E = "#EF5350"
C_TXT     = "#37474F"
C_TXT_L   = "#78909C"
C_ARROW   = "#B0BEC5"
C_CORR    = "#FFAB91"


def _box(ax, x, y, w, h, text, fc, ec, tc=None, fs=9, bold=False, lw=1.2, zorder=2):
    if tc is None:
        tc = C_TXT
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.04",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=tc, fontfamily=FONT, fontweight="bold" if bold else "normal",
            zorder=zorder+1, linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.3):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=1)


def _draw_mini_tree(ax, cx, cy, scale=1.0, node_c=C_TREE_E, leaf_c=C_TREE):
    """Iconic binary-tree glyph."""
    s = scale
    r = 0.10 * s
    root = plt.Circle((cx, cy), r, fc=node_c, ec="white", lw=1.2, zorder=6)
    ax.add_patch(root)
    l1 = [(cx - 0.22*s, cy - 0.22*s), (cx + 0.22*s, cy - 0.22*s)]
    for nx, ny in l1:
        n = plt.Circle((nx, ny), r*0.8, fc=node_c, ec="white", lw=1, zorder=6, alpha=0.75)
        ax.add_patch(n)
        ax.plot([cx, nx], [cy - r, ny + r*0.8], color="white", lw=1, zorder=5)
    leaves = [
        (cx - 0.36*s, cy - 0.43*s), (cx - 0.10*s, cy - 0.43*s),
        (cx + 0.10*s, cy - 0.43*s), (cx + 0.36*s, cy - 0.43*s),
    ]
    for li, (lx, ly) in enumerate(leaves):
        parent = l1[0] if li < 2 else l1[1]
        ax.plot([parent[0], lx], [parent[1] - r*0.8, ly + 0.05*s],
                color="white", lw=0.8, zorder=5)
        leaf = FancyBboxPatch((lx - 0.055*s, ly - 0.055*s), 0.11*s, 0.11*s,
                              boxstyle="round,pad=0.01", fc=leaf_c, ec=node_c,
                              lw=0.7, zorder=6)
        ax.add_patch(leaf)


def generate():
    fig, ax = plt.subplots(figsize=(14, 9.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # ── Title ───────────────────────────────────────────────────────
    ax.text(7, 9.15, "Conceptual Architecture of the Supervised Gradient Boosting Model",
            ha="center", va="top", fontsize=13.5, fontfamily=FONT,
            fontweight="bold", color=C_TXT)

    # ════════════════════════════════════════════════════════════════
    # 1)  INPUT FEATURE VECTOR
    # ════════════════════════════════════════════════════════════════
    ix = 1.6
    _box(ax, ix, 5.0, 2.5, 4.6, "", C_INPUT, C_INPUT_E, lw=1.3)
    ax.text(ix, 7.1, "Transaction\nFeature Vector", ha="center", va="center",
            fontsize=10, fontfamily=FONT, fontweight="bold", color=C_INPUT_E, zorder=3)

    rows = [
        "Numerical features",
        "Categorical features",
        "Velocity signals",
        "Device indicators",
        "Network context",
        "Temporal patterns",
    ]
    fy = 6.15
    for row in rows:
        _box(ax, ix, fy, 2.1, 0.32, row,
             fc="#FFFFFF", ec=C_INPUT_E, tc=C_TXT, fs=7.5, lw=0.7)
        fy -= 0.44

    ax.text(ix, fy + 0.08, "preprocessed and scaled", ha="center", va="center",
            fontsize=7, fontfamily=FONT, fontstyle="italic", color=C_TXT_L, zorder=3)

    # ════════════════════════════════════════════════════════════════
    # 2)  SEQUENTIAL DECISION TREE ENSEMBLE
    # ════════════════════════════════════════════════════════════════
    ex = 7.2
    _box(ax, ex, 5.75, 7.0, 5.0, "", C_TREE_BG, C_TREE_E, lw=1.3)
    ax.text(ex, 8.05, "Sequential Decision Tree Ensemble",
            ha="center", va="center", fontsize=11, fontfamily=FONT,
            fontweight="bold", color=C_TREE_E, zorder=3)
    ax.text(ex, 7.6, "each tree corrects the residual errors of the previous trees",
            ha="center", va="center", fontsize=8, fontfamily=FONT,
            fontstyle="italic", color=C_TXT_L, zorder=3)

    # Tree cards
    trees = [
        (4.5,  6.0, "Tree 1"),
        (6.1,  6.0, "Tree 2"),
        (7.7,  6.0, "Tree 3"),
        (9.9,  6.0, "Tree K"),
    ]
    for tx, ty, label in trees:
        _box(ax, tx, ty, 1.25, 1.85, "", C_TREE, C_TREE_E, lw=0.8)
        _draw_mini_tree(ax, tx, ty + 0.25, scale=0.9)
        ax.text(tx, ty - 0.7, label, ha="center", va="center",
                fontsize=8.5, fontfamily=FONT, fontweight="bold",
                color=C_TREE_E, zorder=7)

    # Ellipsis dots
    for dx in [8.5, 8.85, 9.2]:
        ax.plot(dx, 6.0, 'o', color=C_TREE_E, markersize=3.5, alpha=0.45, zorder=3)

    # Dashed sequential arrows
    for a, b in [(4.5, 6.1), (6.1, 7.7)]:
        ax.annotate("", xy=(b - 0.62, 6.0), xytext=(a + 0.62, 6.0),
                    arrowprops=dict(arrowstyle="-|>", color=C_CORR, lw=1.2,
                                    linestyle=(0, (4, 3))), zorder=3)

    # Input → ensemble
    _arrow(ax, 2.85, 5.0, 3.8, 6.0, color=C_INPUT_E, lw=1.5)

    # Per-tree output arrows
    for tx, ty, _ in trees:
        _arrow(ax, tx, ty - 0.93, tx, 3.95, color=C_TREE_E, lw=1)

    # ════════════════════════════════════════════════════════════════
    # 3)  WEIGHTED SUM
    # ════════════════════════════════════════════════════════════════
    _box(ax, ex, 3.4, 5.4, 0.65,
         "Weighted Sum of All Tree Predictions",
         C_SUM, C_SUM_E, fs=10, bold=True)

    # ════════════════════════════════════════════════════════════════
    # 4)  SIGMOID → PROBABILITY
    # ════════════════════════════════════════════════════════════════
    _box(ax, ex, 2.4, 4.2, 0.60,
         "Sigmoid Activation   -->   Fraud Probability",
         C_SIGMOID, C_SIG_E, fs=9.5, bold=True)
    _arrow(ax, ex, 3.4 - 0.325, ex, 2.4 + 0.30, color=C_SUM_E, lw=1.5)

    # ════════════════════════════════════════════════════════════════
    # 5)  THRESHOLD
    # ════════════════════════════════════════════════════════════════
    _box(ax, ex, 1.45, 3.4, 0.55,
         "Decision Threshold",
         C_THRESH, C_THRESH_E, fs=10, bold=True)
    _arrow(ax, ex, 2.4 - 0.30, ex, 1.45 + 0.275, color=C_SIG_E, lw=1.5)

    # ════════════════════════════════════════════════════════════════
    # 6)  OUTCOMES
    # ════════════════════════════════════════════════════════════════
    _box(ax, 5.0, 0.38, 2.6, 0.60,
         "Legitimate",
         C_LEGIT, C_LEGIT_E, fs=10, bold=True, tc=C_LEGIT_E)
    _box(ax, 9.4, 0.38, 2.6, 0.60,
         "Fraudulent",
         C_FRAUD, C_FRAUD_E, fs=10, bold=True, tc=C_FRAUD_E)

    _arrow(ax, 6.1, 1.45 - 0.275, 5.0, 0.38 + 0.30, color=C_LEGIT_E, lw=1.5)
    _arrow(ax, 8.3, 1.45 - 0.275, 9.4, 0.38 + 0.30, color=C_FRAUD_E, lw=1.5)

    # Subtle outcome labels
    ax.text(5.0, -0.08, "score below threshold", ha="center", va="center",
            fontsize=7.5, fontfamily=FONT, fontstyle="italic", color=C_TXT_L)
    ax.text(9.4, -0.08, "score at or above threshold", ha="center", va="center",
            fontsize=7.5, fontfamily=FONT, fontstyle="italic", color=C_TXT_L)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(OUT / "fig_6_2_xgboost_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [+] fig_6_2_xgboost_architecture.png")


if __name__ == "__main__":
    print("Generating Chapter 6 architecture diagram ...")
    generate()
    print(f"Done – saved to {OUT.resolve()}")
