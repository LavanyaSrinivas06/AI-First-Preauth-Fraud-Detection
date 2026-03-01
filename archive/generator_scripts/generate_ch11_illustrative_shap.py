"""
Generate illustrative SHAP beeswarm-style plots for Chapter 11.

Each plot represents a single *example transaction* with hand-crafted
feature values and SHAP contributions that clearly demonstrate one
reason-code category.  The dots use the same blue → red colour gradient
as the standard SHAP summary_plot (blue = low feature value, red = high).

These are thesis illustrations, not real model outputs.

Output: docs/figures/thesis_diagrams/shap_illustrative_*.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

OUT_DIR = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "docs" / "figures" / "thesis_diagrams"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour setup (matches your reference image exactly) ───────────────
CMAP = matplotlib.colormaps["coolwarm"]       # blue → white → red


# ── scenario definitions ──────────────────────────────────────────────
# Each scenario has:
#   txn_id   – human-readable transaction label for the title
#   title    – reason code category
#   features – list of (feature_name, shap_value, feature_value_normalised)
#              feature_value_normalised: 0 = low (blue), 1 = high (red)
#   The list is ordered top → bottom (most important first).

SCENARIOS = [
    # ── 1. High Transaction Velocity ──────────────────────────────────
    {
        "txn_id": "rev_a3f8842c1b9e04d7",
        "title": "Reason Code: High Transaction Frequency",
        "tag": "velocity",
        "features": [
            ("Txn count (5 min)",          +0.182, 1.00),
            ("Txn count (30 min)",         +0.134, 0.92),
            ("Txn count (60 min)",         +0.097, 0.85),
            ("Account age (days)",         -0.061, 0.15),
            ("V14",                        +0.048, 0.70),
            ("V12",                        -0.044, 0.22),
            ("Transaction amount",         +0.031, 0.58),
            ("IP reputation score",        -0.028, 0.35),
            ("V17",                        +0.022, 0.62),
            ("Geo distance (km)",          +0.018, 0.55),
            ("New device = True",          +0.015, 1.00),
            ("VPN / Proxy = False",        -0.011, 0.00),
            ("V10",                        +0.009, 0.48),
            ("Amount z-score",             +0.007, 0.60),
            ("Browser: Chrome",            -0.004, 0.00),
        ],
    },
    # ── 2. VPN / Proxy Detected ───────────────────────────────────────
    {
        "txn_id": "rev_3157e6b0ca4f28d1",
        "title": "Reason Code: Suspicious Network (VPN / Proxy)",
        "tag": "proxy",
        "features": [
            ("VPN / Proxy = True",         +0.210, 1.00),
            ("IP reputation score",        +0.145, 0.90),
            ("VPN / Proxy = False",        -0.130, 0.00),
            ("V14",                        +0.052, 0.72),
            ("Geo distance (km)",          +0.044, 0.68),
            ("Account age (days)",         -0.038, 0.20),
            ("V12",                        -0.033, 0.18),
            ("Transaction amount",         +0.025, 0.55),
            ("V17",                        +0.019, 0.60),
            ("Txn count (5 min)",          +0.014, 0.50),
            ("Country mismatch",           +0.012, 1.00),
            ("New device = True",          +0.008, 0.00),
            ("V10",                        -0.006, 0.30),
            ("Amount z-score",             +0.005, 0.52),
            ("Browser: Firefox",           -0.003, 0.00),
        ],
    },
    # ── 3. New / Unseen Device ────────────────────────────────────────
    {
        "txn_id": "rev_5501d9f7e82a36bc",
        "title": "Reason Code: New Device Detected",
        "tag": "new_device",
        "features": [
            ("New device = True",          +0.195, 1.00),
            ("Geo distance (km)",          +0.128, 0.88),
            ("Country mismatch",           +0.095, 1.00),
            ("Account age (days)",         -0.058, 0.12),
            ("V14",                        +0.042, 0.65),
            ("IP reputation score",        -0.035, 0.30),
            ("V12",                        -0.030, 0.25),
            ("Transaction amount",         +0.024, 0.54),
            ("VPN / Proxy = True",         +0.018, 0.00),
            ("Txn count (60 min)",         +0.013, 0.45),
            ("V17",                        +0.011, 0.58),
            ("Amount z-score",             +0.008, 0.50),
            ("V10",                        -0.006, 0.32),
            ("Browser: Safari",            -0.004, 0.00),
            ("Device: iPhone",             +0.003, 1.00),
        ],
    },
    # ── 4. Unusual Transaction Amount ─────────────────────────────────
    {
        "txn_id": "rev_7290b4ac05e1f83d",
        "title": "Reason Code: Unusual Transaction Amount",
        "tag": "amount",
        "features": [
            ("Transaction amount",         +0.225, 1.00),
            ("Amount z-score",             +0.168, 0.95),
            ("Avg amount (7 day)",         +0.102, 0.88),
            ("V14",                        +0.055, 0.70),
            ("Account age (days)",         -0.045, 0.18),
            ("V12",                        -0.038, 0.20),
            ("IP reputation score",        -0.029, 0.35),
            ("Geo distance (km)",          +0.021, 0.50),
            ("V17",                        +0.018, 0.62),
            ("Txn count (5 min)",          +0.013, 0.48),
            ("New device = True",          +0.009, 0.00),
            ("VPN / Proxy = False",        -0.007, 0.00),
            ("Country mismatch",           +0.005, 0.00),
            ("V10",                        -0.004, 0.38),
            ("Browser: Chrome",            -0.002, 0.00),
        ],
    },
    # ── 5. Atypical Behavioural Pattern ───────────────────────────────
    {
        "txn_id": "rev_4618c7de93fa50b2",
        "title": "Reason Code: Atypical Behavioural Pattern",
        "tag": "behaviour",
        "features": [
            ("V14",                        +0.198, 0.92),
            ("V12",                        -0.155, 0.08),
            ("V10",                        +0.112, 0.85),
            ("V17",                        +0.088, 0.80),
            ("V7",                         +0.065, 0.78),
            ("Account age (days)",         -0.042, 0.22),
            ("Transaction amount",         +0.030, 0.55),
            ("IP reputation score",        -0.025, 0.32),
            ("Geo distance (km)",          +0.018, 0.50),
            ("Txn count (30 min)",         +0.014, 0.48),
            ("New device = True",          +0.010, 0.00),
            ("VPN / Proxy = True",         +0.007, 0.00),
            ("Amount z-score",             +0.005, 0.52),
            ("Country mismatch",           +0.004, 0.00),
            ("Browser: Chrome",            -0.002, 0.00),
        ],
    },
]


# ── plot generator ────────────────────────────────────────────────────

def _generate(scenario: dict):
    feats   = scenario["features"]
    n       = len(feats)
    names   = [f[0] for f in feats]
    shaps   = np.array([f[1] for f in feats])
    normals = np.array([f[2] for f in feats])   # 0 → blue, 1 → red

    # Reverse so highest-impact feature is at top of plot
    names   = names[::-1]
    shaps   = shaps[::-1]
    normals = normals[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.42 * n + 1.6))

    rng = np.random.default_rng(seed=17)

    for i in range(n):
        base_shap   = shaps[i]
        base_normal = normals[i]

        # Create a small cloud of ~50 dots around the true value
        # to mimic the beeswarm distribution
        n_dots = 55
        dot_shaps   = base_shap + rng.normal(0, abs(base_shap) * 0.25 + 0.003, n_dots)
        dot_normals = np.clip(base_normal + rng.normal(0, 0.12, n_dots), 0, 1)
        jitter_y    = rng.uniform(-0.30, 0.30, n_dots)

        colours = CMAP(dot_normals)

        ax.scatter(
            dot_shaps,
            np.full(n_dots, i) + jitter_y,
            c=colours,
            s=14,
            alpha=0.78,
            edgecolors="none",
            zorder=3,
        )

    # Axis labels & styling
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    ax.set_title(
        f"{scenario['title']}  —  {scenario['txn_id']}",
        fontsize=11.5, fontweight="bold", pad=14,
    )
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--", zorder=1)
    ax.set_ylim(-0.8, n - 0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.82)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / f"shap_illustrative_{scenario['tag']}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}  ({out.stat().st_size / 1024:.0f} KB)")
    return out


# ── main ──────────────────────────────────────────────────────────────
def main():
    print(f"Output: {OUT_DIR}\n")
    paths = []
    for sc in SCENARIOS:
        print(f"▸ {sc['title']}  ({sc['txn_id']})")
        paths.append(_generate(sc))
    print(f"\nDone — {len(paths)} illustrative SHAP plots generated.")


if __name__ == "__main__":
    main()
