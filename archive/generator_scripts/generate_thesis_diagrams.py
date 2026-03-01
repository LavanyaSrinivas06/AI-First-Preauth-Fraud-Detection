#!/usr/bin/env python3
"""
Generate thesis-quality PNG diagrams for sections 4.3–4.8.
Output: docs/figures/thesis_diagrams/fig_4_3_*.png … fig_4_8_*.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("docs/figures/thesis_diagrams")
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared styling ──────────────────────────────────────────────────────────
FONT = {"family": "sans-serif", "size": 10}
matplotlib.rc("font", **FONT)
COLORS = {
    "blue":   "#bbdefb",  "blue_e":   "#1565c0",
    "green":  "#c8e6c9",  "green_e":  "#2e7d32",
    "orange": "#ffe0b2",  "orange_e": "#e65100",
    "red":    "#ffcdd2",  "red_e":    "#c62828",
    "purple": "#e1bee7",  "purple_e": "#7b1fa2",
    "yellow": "#fff9c4",  "yellow_e": "#f9a825",
    "grey":   "#f5f5f5",  "grey_e":   "#616161",
    "white":  "#ffffff",  "white_e":  "#bdbdbd",
}


def _box(ax, x, y, w, h, text, fc, ec, fs=9, bold=False, ha="center"):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15", fc=fc, ec=ec, lw=1.5,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, weight=weight,
            wrap=True, linespacing=1.4)


def _arrow(ax, x1, y1, x2, y2, color="#616161"):
    """Simple arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 4.4 Synthetic Feature Enrichment Workflow (the main one)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_4_4_enrichment():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(7, 10.1, "Synthetic Feature Enrichment Workflow",
            ha="center", va="center", fontsize=14, weight="bold")

    # Row 1: Base dataset
    _box(ax, 7, 9.2, 5, 0.8,
         "Kaggle Base Dataset\n284,807 rows × 31 cols (Time, V1–V28, Amount, Class)",
         COLORS["blue"], COLORS["blue_e"], fs=10, bold=True)

    # Arrow down
    _arrow(ax, 7, 8.8, 7, 8.2)

    # Row 2: Seed
    _box(ax, 7, 7.8, 3.5, 0.6,
         "Deterministic Seed (seed = 42)",
         COLORS["yellow"], COLORS["yellow_e"], fs=9, bold=True)

    # Arrows fanning out
    xs = [2.0, 5.0, 7.0, 9.0, 12.0]
    for xd in xs:
        _arrow(ax, 7, 7.5, xd, 6.8)

    # Row 3: Five feature groups
    groups = [
        ("Device &\nBrowser",       "device_id\ndevice_os\nbrowser\nis_new_device",         COLORS["purple"], COLORS["purple_e"]),
        ("Network &\nIP",           "ip_country\nis_proxy_vpn\nip_reputation",               COLORS["red"],    COLORS["red_e"]),
        ("Velocity &\nBehavioral",  "txn_count_5m\ntxn_count_30m\ntxn_count_60m\navg_amount_7d", COLORS["orange"], COLORS["orange_e"]),
        ("Profile &\nAccount",      "account_age_days\ntoken_age_days\navg_spend_user_30d",  COLORS["green"],  COLORS["green_e"]),
        ("Geo &\nAddress",          "billing_country\nshipping_country\ngeo_distance_km\ncountry_mismatch", COLORS["blue"], COLORS["blue_e"]),
    ]
    for i, (title, feats, fc, ec) in enumerate(groups):
        x = xs[i]
        # Group header
        _box(ax, x, 6.4, 2.4, 0.6, title, fc, ec, fs=9, bold=True)
        # Feature list
        _box(ax, x, 5.2, 2.4, 1.4, feats, COLORS["white"], ec, fs=8)

    # Arrows converging to Derived
    for xd in xs:
        _arrow(ax, xd, 4.5, 7, 3.8)

    # Row 4: Derived / Temporal
    _box(ax, 7, 3.4, 4.5, 0.7,
         "Derived & Temporal Features\namount_zscore  ·  night_txn  ·  weekend_txn",
         COLORS["orange"], COLORS["orange_e"], fs=9, bold=True)

    # Arrow down
    _arrow(ax, 7, 3.0, 7, 2.4)

    # Row 5: Schema validation
    _box(ax, 7, 2.0, 4.5, 0.6,
         "Schema Validation (schema_enriched.json) · Null Check · Column Ordering",
         COLORS["red"], COLORS["red_e"], fs=8)

    # Arrow down
    _arrow(ax, 7, 1.7, 7, 1.1)

    # Row 6: Output
    _box(ax, 7, 0.7, 5, 0.7,
         "Enriched Dataset — 284,807 rows × 52 cols\ndata/processed/enriched.csv",
         COLORS["green"], COLORS["green_e"], fs=10, bold=True)

    fig.savefig(OUT / "fig_4_4_synthetic_enrichment_workflow.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {OUT / 'fig_4_4_synthetic_enrichment_workflow.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — 4.5 Preprocessing Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def fig_4_5_preprocessing():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6, 8.1, "Data Preprocessing Pipeline", ha="center", fontsize=14, weight="bold")

    # Input
    _box(ax, 6, 7.3, 5, 0.7,
         "Enriched Dataset (52 cols)", COLORS["blue"], COLORS["blue_e"], fs=10, bold=True)
    _arrow(ax, 6, 6.9, 6, 6.4)

    # Feature grouping
    _box(ax, 6, 6.0, 4, 0.6,
         "Feature Grouping\n(identify numerical vs categorical)", COLORS["yellow"], COLORS["yellow_e"], fs=9)

    # Branch left / right
    _arrow(ax, 6, 5.7, 3.5, 5.1)
    _arrow(ax, 6, 5.7, 8.5, 5.1)

    # Numerical
    _box(ax, 3.5, 4.6, 4.0, 0.8,
         "39 Numerical Features\nV1–V28, Amount, ip_reputation,\ntxn_count_*, account_age_days, …",
         COLORS["blue"], COLORS["blue_e"], fs=8)
    _arrow(ax, 3.5, 4.2, 3.5, 3.6)
    _box(ax, 3.5, 3.2, 3.0, 0.6,
         "StandardScaler\n(mean=0, std=1)", COLORS["purple"], COLORS["purple_e"], fs=9, bold=True)

    # Categorical
    _box(ax, 8.5, 4.6, 4.0, 0.8,
         "10 Categorical Features\ndevice_os, browser, ip_country,\nbilling/shipping_country, …",
         COLORS["orange"], COLORS["orange_e"], fs=8)
    _arrow(ax, 8.5, 4.2, 8.5, 3.6)
    _box(ax, 8.5, 3.2, 3.5, 0.6,
         "OneHotEncoder\n(handle_unknown='ignore')", COLORS["purple"], COLORS["purple_e"], fs=9, bold=True)

    # Converge
    _arrow(ax, 3.5, 2.9, 6, 2.3)
    _arrow(ax, 8.5, 2.9, 6, 2.3)

    _box(ax, 6, 1.9, 4.5, 0.6,
         "ColumnTransformer — Fit on TRAIN only",
         COLORS["red"], COLORS["red_e"], fs=9, bold=True)
    _arrow(ax, 6, 1.6, 6, 1.1)

    _box(ax, 6, 0.7, 4, 0.6,
         "102 Model-Ready Features",
         COLORS["green"], COLORS["green_e"], fs=11, bold=True)

    # Side artifacts
    _box(ax, 10.5, 1.9, 2.2, 0.5,
         "preprocess.joblib", COLORS["grey"], COLORS["grey_e"], fs=8)
    _arrow(ax, 8.25, 1.9, 9.4, 1.9, color=COLORS["grey_e"])
    _box(ax, 10.5, 1.2, 2.2, 0.5,
         "features.json", COLORS["grey"], COLORS["grey_e"], fs=8)
    _arrow(ax, 8.0, 0.7, 9.4, 1.2, color=COLORS["grey_e"])

    fig.savefig(OUT / "fig_4_5_preprocessing_pipeline.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {OUT / 'fig_4_5_preprocessing_pipeline.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — 4.6 SMOTE Class Balancing
# ═══════════════════════════════════════════════════════════════════════════════
def fig_4_6_smote():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6, 7.1, "Handling Class Imbalance — SMOTE", ha="center", fontsize=14, weight="bold")

    # Input
    _box(ax, 6, 6.3, 5, 0.7,
         "Training Set — 199,364 rows\nClass 0: 198,980 (99.81%)  |  Class 1: 384 (0.19%)",
         COLORS["red"], COLORS["red_e"], fs=9, bold=True)

    # Branch
    _arrow(ax, 4, 5.9, 3.5, 5.2)
    _arrow(ax, 8, 5.9, 8.5, 5.2)

    # Left: SMOTE path
    _box(ax, 3.5, 4.8, 3.5, 0.6,
         "SMOTE (random_state=42)\nSynthetic oversampling", COLORS["purple"], COLORS["purple_e"], fs=9, bold=True)
    _arrow(ax, 3.5, 4.5, 3.5, 3.9)
    _box(ax, 3.5, 3.4, 4.0, 0.8,
         "Balanced Training Set\n397,960 rows\nClass 0: 198,980 (50%)\nClass 1: 198,980 (50%)",
         COLORS["green"], COLORS["green_e"], fs=9, bold=True)
    _arrow(ax, 3.5, 3.0, 3.5, 2.4)
    _box(ax, 3.5, 2.0, 3.5, 0.6,
         "→ Used by XGBoost\ntrain.csv", COLORS["blue"], COLORS["blue_e"], fs=9)

    # Right: NO-SMOTE path
    _box(ax, 8.5, 4.8, 3.5, 0.6,
         "NO-SMOTE Copy\nOriginal distribution kept", COLORS["yellow"], COLORS["yellow_e"], fs=9, bold=True)
    _arrow(ax, 8.5, 4.5, 8.5, 3.9)
    _box(ax, 8.5, 3.4, 4.0, 0.8,
         "Original Training Set\n199,364 rows\nClass 0: 198,980 (99.81%)\nClass 1: 384 (0.19%)",
         COLORS["orange"], COLORS["orange_e"], fs=9)
    _arrow(ax, 8.5, 3.0, 8.5, 2.4)
    _box(ax, 8.5, 2.0, 3.5, 0.6,
         "→ Used by Autoencoder\ntrain_nosmote.csv", COLORS["blue"], COLORS["blue_e"], fs=9)

    # Bottom note
    _box(ax, 6, 0.8, 6, 0.7,
         "Validation (42,721 rows) & Test (42,722 rows)\nNOT resampled — kept imbalanced for realistic evaluation",
         COLORS["grey"], COLORS["grey_e"], fs=9)

    fig.savefig(OUT / "fig_4_6_smote_class_balancing.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {OUT / 'fig_4_6_smote_class_balancing.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — 4.7 Train-Validation-Test Split
# ═══════════════════════════════════════════════════════════════════════════════
def fig_4_7_split():
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6.5, 4.1, "Time-Based Train / Validation / Test Split",
            ha="center", fontsize=14, weight="bold")

    # Timeline arrow
    ax.annotate("", xy=(12.5, 2.6), xytext=(0.5, 2.6),
                arrowprops=dict(arrowstyle="-|>", color="#616161", lw=2))
    ax.text(0.3, 2.3, "Oldest", fontsize=8, color="#9e9e9e")
    ax.text(12.0, 2.3, "Newest", fontsize=8, color="#9e9e9e")

    # Three blocks on the timeline
    # Training 70%
    train_box = FancyBboxPatch((0.8, 2.8), 7.0, 1.0,
                                boxstyle="round,pad=0.1",
                                fc="#bbdefb", ec="#1565c0", lw=2)
    ax.add_patch(train_box)
    ax.text(4.3, 3.3, "Training Set — 70%\n199,364 rows",
            ha="center", va="center", fontsize=10, weight="bold")

    # Validation 15%
    val_box = FancyBboxPatch((8.0, 2.8), 2.0, 1.0,
                              boxstyle="round,pad=0.1",
                              fc="#fff9c4", ec="#f9a825", lw=2)
    ax.add_patch(val_box)
    ax.text(9.0, 3.3, "Val — 15%\n42,721",
            ha="center", va="center", fontsize=9, weight="bold")

    # Test 15%
    test_box = FancyBboxPatch((10.2, 2.8), 2.0, 1.0,
                               boxstyle="round,pad=0.1",
                               fc="#c8e6c9", ec="#2e7d32", lw=2)
    ax.add_patch(test_box)
    ax.text(11.2, 3.3, "Test — 15%\n42,722",
            ha="center", va="center", fontsize=9, weight="bold")

    # Descriptions below
    ax.text(4.3, 1.7, "Model training\n+ SMOTE applied here",
            ha="center", fontsize=9, color="#1565c0")
    ax.text(9.0, 1.7, "Threshold tuning\n& comparison",
            ha="center", fontsize=9, color="#f9a825")
    ax.text(11.2, 1.7, "Final evaluation\n(used once)",
            ha="center", fontsize=9, color="#2e7d32")

    # Dashed lines
    for xx in [8.0, 10.2]:
        ax.plot([xx, xx], [1.4, 3.8], ls="--", color="#bdbdbd", lw=1)

    # Time label
    ax.text(6.5, 0.8, "← Transactions sorted by Time →",
            ha="center", fontsize=10, style="italic", color="#757575")

    fig.savefig(OUT / "fig_4_7_train_val_test_split.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {OUT / 'fig_4_7_train_val_test_split.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — 4.8 Data Quality & Leakage Prevention
# ═══════════════════════════════════════════════════════════════════════════════
def fig_4_8_quality():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6, 6.1, "Data Quality Checks & Leakage Prevention",
            ha="center", fontsize=14, weight="bold")

    # Left column: Quality
    _box(ax, 3, 5.2, 4.0, 0.6,
         "Data Quality Checks", COLORS["blue"], COLORS["blue_e"], fs=11, bold=True)

    checks = [
        "✓ No missing values (null check enforced)",
        "✓ No duplicate transactions",
        "✓ Time order preserved (sorted)",
        "✓ Correlation heatmap reviewed",
    ]
    for i, txt in enumerate(checks):
        y = 4.3 - i * 0.7
        _box(ax, 3, y, 4.5, 0.5, txt, COLORS["white"], COLORS["blue_e"], fs=8)

    # Right column: Leakage
    _box(ax, 9, 5.2, 4.0, 0.6,
         "Leakage Prevention", COLORS["red"], COLORS["red_e"], fs=11, bold=True)

    leaks = [
        "✓ Features use only past information",
        "✓ Pipeline fit on TRAIN data only",
        "✓ SMOTE applied AFTER split",
        "✓ Val/Test never resampled",
    ]
    for i, txt in enumerate(leaks):
        y = 4.3 - i * 0.7
        _box(ax, 9, y, 4.5, 0.5, txt, COLORS["white"], COLORS["red_e"], fs=8)

    # Bottom
    _box(ax, 6, 0.8, 8, 0.7,
         "All checks passed — dataset is ready for model training and evaluation",
         COLORS["green"], COLORS["green_e"], fs=10, bold=True)

    _arrow(ax, 3, 2.0, 5, 1.2, color=COLORS["blue_e"])
    _arrow(ax, 9, 2.0, 7, 1.2, color=COLORS["red_e"])

    fig.savefig(OUT / "fig_4_8_data_quality_leakage.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {OUT / 'fig_4_8_data_quality_leakage.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — 4.3 Need for Synthetic Enrichment (Gap Analysis)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_4_3_gap():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6.5, 6.1, "Need for Synthetic Enrichment — Gap Analysis",
            ha="center", fontsize=14, weight="bold")

    # Base dataset
    _box(ax, 2.5, 5.0, 4.0, 0.8,
         "Kaggle Base Dataset\n284,807 rows × 31 cols",
         COLORS["blue"], COLORS["blue_e"], fs=10, bold=True)

    # Gaps
    gaps = [
        "No device / browser info",
        "No network / IP data",
        "No velocity / behavioral signals",
        "No account / profile context",
        "No geo / address fields",
        "Limited temporal features",
    ]
    for i, g in enumerate(gaps):
        y = 5.0 - i * 0.65
        _box(ax, 7.0, y, 3.5, 0.45, "[X]  " + g, COLORS["red"], COLORS["red_e"], fs=8)
        _arrow(ax, 4.5, 5.0 - i * 0.15, 5.25, y, color=COLORS["red_e"])

    # Arrow to solution
    _arrow(ax, 8.75, 1.7, 10.5, 1.7, color=COLORS["purple_e"])

    # Solution
    _box(ax, 10.5, 3.2, 3.5, 0.8,
         "Synthetic\nEnrichment\n(seed = 42)",
         COLORS["purple"], COLORS["purple_e"], fs=10, bold=True)

    for i in range(6):
        y = 5.0 - i * 0.65
        _arrow(ax, 8.75, y, 9.5, 3.2 + (2.5 - i) * 0.08, color=COLORS["purple_e"])

    _arrow(ax, 10.5, 2.8, 10.5, 1.3)

    # Output
    _box(ax, 10.5, 0.8, 3.5, 0.8,
         "Enriched Dataset\n284,807 × 52 cols\n+21 new features",
         COLORS["green"], COLORS["green_e"], fs=10, bold=True)

    fig.savefig(OUT / "fig_4_3_need_for_enrichment.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {OUT / 'fig_4_3_need_for_enrichment.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig_4_3_gap()
    fig_4_4_enrichment()
    fig_4_5_preprocessing()
    fig_4_6_smote()
    fig_4_7_split()
    fig_4_8_quality()
    print(f"\n🎉 All diagrams saved to: {OUT.resolve()}")
