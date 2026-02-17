#!/usr/bin/env python3
"""
Chapter 7 — Denoising Autoencoder Architecture.

Classic ML-paper style: each layer is a tall vertical bar whose HEIGHT
is proportional to the neuron count, producing the iconic symmetric
hourglass / funnel silhouette that visually communicates compression
and reconstruction.

Trapezoid connectors between bars show the information flow, and
implementation-specific details (BatchNorm, Dropout, loss, etc.)
are annotated alongside.

Thesis-quality: serif font, black outlines, faint fills, 300 dpi.
Output: docs/figures/thesis_diagrams/fig_7_1_autoencoder_architecture.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
import matplotlib.patches as mpatches

OUT = Path("docs/figures/thesis_diagrams")
OUT.mkdir(parents=True, exist_ok=True)

FONT = "serif"

# ── Palette ─────────────────────────────────────────────────────────
BG      = "#FFFFFF"
BLACK   = "#1a1a1a"
GREY    = "#444444"
LGREY   = "#888888"
ENC_C   = "#b8cce4"   # encoder bar
ENC_C2  = "#92b0d1"   # encoder bar darker
BN_C    = "#c4b7d9"   # bottleneck
DEC_C   = "#a7d5a7"   # decoder bar
DEC_C2  = "#8bc48b"   # decoder bar darker
IO_C    = "#d0d0d0"   # input / output
NOISE_C = "#f5d6a8"   # noise
TRAP_C  = "#e8e8e8"   # trapezoid fill (connections)


# ── Helper: draw one vertical bar ──────────────────────────────────

def _bar(ax, cx, bar_h, bar_w, fc, ec=BLACK, lw=1.1):
    """Draw a rounded vertical bar centred at cx, centred vertically at y=0."""
    box = FancyBboxPatch(
        (cx - bar_w / 2, -bar_h / 2), bar_w, bar_h,
        boxstyle="round,pad=0.04",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(box)


def _trapezoid(ax, x1, h1, w1, x2, h2, w2, fc=TRAP_C, alpha=0.45):
    """Draw a filled trapezoid connecting two bars (information flow)."""
    verts = [
        (x1 + w1 / 2,  h1 / 2),
        (x2 - w2 / 2,  h2 / 2),
        (x2 - w2 / 2, -h2 / 2),
        (x1 + w1 / 2, -h1 / 2),
    ]
    trap = Polygon(verts, closed=True, facecolor=fc,
                   edgecolor=LGREY, linewidth=0.4, alpha=alpha, zorder=2)
    ax.add_patch(trap)


def _bracket_top(ax, x_left, x_right, y_base, label, fs=10):
    tick = 0.3
    ax.plot([x_left, x_left, x_right, x_right],
            [y_base, y_base + tick, y_base + tick, y_base],
            color=GREY, lw=0.9, clip_on=False, zorder=4)
    mid = (x_left + x_right) / 2
    ax.text(mid, y_base + tick + 0.2, label, ha="center", va="bottom",
            fontsize=fs, fontfamily=FONT, fontweight="bold",
            color=GREY, zorder=5)


# ── Main figure ─────────────────────────────────────────────────────

def generate():
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    # ================================================================
    #  Layer spec — neurons map to visual bar height (proportional).
    #  Actual arch: Input(102) → GaussianNoise(0.05)
    #    → Dense(128)+BN+ReLU+Dropout(0.2) → Dense(64)+BN+ReLU
    #    → Dense(32,ReLU) [bottleneck]
    #    → Dense(64,ReLU) → Dense(128,ReLU) → Dense(102,Linear)
    # ================================================================

    # Scale: 1 neuron ≈ 0.055 visual units height
    scale = 0.055
    bar_w = 0.85       # width of each vertical bar
    gap   = 1.85       # horizontal spacing centre-to-centre

    #  (label, neurons, colour, annotation_lines)
    layer_spec = [
        ("Input\n102",       102, IO_C,    []),
        ("+ Noise\n\u03c3=0.05", 102, NOISE_C, ["GaussianNoise"]),
        ("Dense\n128",       128, ENC_C,   ["BatchNorm", "ReLU", "Dropout 0.2"]),
        ("Dense\n64",         64, ENC_C2,  ["BatchNorm", "ReLU"]),
        ("Dense\n32",         32, BN_C,    ["ReLU"]),
        ("Dense\n64",         64, DEC_C2,  ["ReLU"]),
        ("Dense\n128",       128, DEC_C,   ["ReLU"]),
        ("Output\n102",     102, IO_C,    ["Linear"]),
    ]

    n = len(layer_spec)
    # x positions: centred around 0
    xs = [i * gap - (n - 1) * gap / 2 for i in range(n)]
    heights = [spec[1] * scale for spec in layer_spec]

    # ── Trapezoid connectors (draw first, behind bars) ──────────
    for i in range(n - 1):
        _trapezoid(ax, xs[i], heights[i], bar_w,
                       xs[i + 1], heights[i + 1], bar_w)

    # ── Vertical bars ───────────────────────────────────────────
    for i, (label, neurons, colour, annot) in enumerate(layer_spec):
        _bar(ax, xs[i], heights[i], bar_w, fc=colour)

        # Layer label inside bar
        ax.text(xs[i], 0, label, ha="center", va="center",
                fontsize=8.5, fontfamily=FONT, fontweight="bold",
                color=BLACK, zorder=5, linespacing=1.25)

        # Annotation lines below bar
        if annot:
            annot_text = "\n".join(annot)
            ax.text(xs[i], -heights[i] / 2 - 0.35, annot_text,
                    ha="center", va="top", fontsize=7, fontfamily=FONT,
                    color=GREY, linespacing=1.3, zorder=5)

    # ── Arrow at very start & end ───────────────────────────────
    arr_kw = dict(arrowstyle="-|>", color=BLACK, lw=1.2,
                  mutation_scale=14)
    # Input arrow (from left)
    ax.annotate("", xy=(xs[0] - bar_w / 2, 0),
                xytext=(xs[0] - bar_w / 2 - 1.1, 0),
                arrowprops=arr_kw, zorder=4)
    ax.text(xs[0] - bar_w / 2 - 1.3, 0, r"$\mathbf{x}$",
            ha="right", va="center", fontsize=14, fontfamily=FONT,
            color=BLACK, zorder=5)

    # Output arrow (to right)
    ax.annotate("", xy=(xs[-1] + bar_w / 2 + 1.1, 0),
                xytext=(xs[-1] + bar_w / 2, 0),
                arrowprops=arr_kw, zorder=4)
    ax.text(xs[-1] + bar_w / 2 + 1.3, 0, r"$\hat{\mathbf{x}}$",
            ha="left", va="center", fontsize=14, fontfamily=FONT,
            color=BLACK, zorder=5)

    # ── Top brackets: Encoder / Bottleneck / Decoder ────────────
    max_h = max(heights)
    brk_y = max_h / 2 + 0.5

    # Encoder bracket (Noise → Dense 64)
    _bracket_top(ax, xs[1] - bar_w / 2, xs[3] + bar_w / 2, brk_y, "Encoder")

    # Bottleneck label
    ax.text(xs[4], brk_y + 0.7, "Bottleneck",
            ha="center", va="bottom", fontsize=9.5, fontfamily=FONT,
            fontstyle="italic", color=GREY, zorder=5)

    # Decoder bracket (Dense 64 → Dense 128)
    _bracket_top(ax, xs[5] - bar_w / 2, xs[6] + bar_w / 2, brk_y, "Decoder")

    # ── Loss / Optimiser annotation ─────────────────────────────
    bottom_y = -max_h / 2 - 2.2
    ax.text(0, bottom_y,
            "Loss: Huber (\u03b4 = 1.0)            "
            "Optimiser: Adam (lr = 10\u207b\u00b3, clipnorm = 1.0)",
            ha="center", va="center", fontsize=9, fontfamily=FONT,
            color=GREY, zorder=5)

    # ── Neuron count dimension labels (right side of each bar) ──
    for i, (label, neurons, colour, annot) in enumerate(layer_spec):
        ax.text(xs[i] + bar_w / 2 + 0.12, heights[i] / 2,
                str(neurons), ha="left", va="top",
                fontsize=7, fontfamily=FONT, color=LGREY,
                fontstyle="italic", zorder=5)

    # ── Set limits ──────────────────────────────────────────────
    ax.set_xlim(xs[0] - 2.8, xs[-1] + 2.8)
    ax.set_ylim(bottom_y - 0.8, brk_y + 1.8)
    ax.set_aspect("equal")

    # ════════════════════════════════════════════════════════════
    fig.tight_layout()
    out_path = OUT / "fig_7_1_autoencoder_architecture.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    generate()
