"""
Generate per-reason-code SHAP beeswarm (summary) plots for Chapter 11.

Uses ALL 345 demo payloads to compute SHAP values, then for each reason
code produces a beeswarm plot showing only the top features relevant to
that code.  Colour gradient: blue (low feature value) → red (high feature
value) — same style as the standard SHAP summary_plot.

Output: docs/figures/thesis_diagrams/shap_reason_*.png  (5 plots)
"""
from __future__ import annotations

import json, glob
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import shap

# ── paths ──────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
ART    = ROOT / "artifacts"
DEMO   = ROOT / "demo_payloads"
OUT    = ROOT / "docs" / "figures" / "thesis_diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# ── load model + feature schema ───────────────────────────────────────
model = joblib.load(ART / "models" / "xgb_model.pkl")
feature_names: list[str] = json.loads((ART / "features.json").read_text())
N_FEAT = len(feature_names)

# ── collect ALL demo payloads ─────────────────────────────────────────
def _collect_all_payloads() -> tuple[np.ndarray, np.ndarray]:
    """Return (X, feature_matrix) each shape (n_samples, n_features)."""
    rows = []
    for jf in sorted(DEMO.rglob("*.json")):
        try:
            raw = json.loads(jf.read_text())
            feats = raw["features"]
            row = [float(feats.get(f, 0.0)) for f in feature_names]
            rows.append(row)
        except Exception:
            continue
    X = np.array(rows, dtype=np.float64)
    return X

print("Loading all demo payloads …")
X_all = _collect_all_payloads()
print(f"  {X_all.shape[0]} samples × {X_all.shape[1]} features")

# ── compute SHAP for ALL samples (one pass) ───────────────────────────
print("Computing SHAP values (this may take a moment) …")
sv = None
try:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_all)
except Exception:
    try:
        booster = model.get_booster()
        explainer = shap.TreeExplainer(booster)
        sv = explainer.shap_values(X_all)
    except Exception:
        fn = lambda arr: model.predict_proba(arr)[:, 1]
        explainer = shap.Explainer(fn, masker=shap.maskers.Independent(X_all[:50]))
        result = explainer(X_all)
        sv = getattr(result, "values", result)

SHAP_ALL = np.asarray(sv)       # (n_samples, n_features)
print(f"  SHAP matrix: {SHAP_ALL.shape}")

# ── reason-code scenarios ─────────────────────────────────────────────
SCENARIOS = [
    {
        "label": "High Transaction Frequency",
        "tag": "velocity",
        "features": ["num__txn_count_5m", "num__txn_count_30m", "num__txn_count_60m"],
        "top_n": 15,
    },
    {
        "label": "Suspicious Network (VPN / Proxy)",
        "tag": "proxy",
        "features": ["cat__is_proxy_vpn_True", "cat__is_proxy_vpn_False",
                      "num__ip_reputation"],
        "top_n": 15,
    },
    {
        "label": "New Device Detected",
        "tag": "new_device",
        "features": ["cat__is_new_device_True", "cat__is_new_device_False",
                      "num__geo_distance_km"],
        "top_n": 15,
    },
    {
        "label": "Unusual Transaction Amount",
        "tag": "amount",
        "features": ["num__Amount", "num__amount_zscore", "num__avg_amount_7d"],
        "top_n": 15,
    },
    {
        "label": "Atypical Behavioural Pattern",
        "tag": "behaviour",
        "features": ["num__V12", "num__V17", "num__V14", "num__V10", "num__V11"],
        "top_n": 15,
    },
]

# ── friendly names ────────────────────────────────────────────────────
_RENAMES = {
    "num__txn_count_5m": "Txn count (5 min)",
    "num__txn_count_30m": "Txn count (30 min)",
    "num__txn_count_60m": "Txn count (60 min)",
    "num__Amount": "Transaction amount",
    "num__amount_zscore": "Amount z-score",
    "num__avg_amount_7d": "Avg amount (7 day)",
    "num__avg_spend_user_30d": "Avg spend (30 day)",
    "num__geo_distance_km": "Geo distance (km)",
    "num__ip_reputation": "IP reputation score",
    "num__account_age_days": "Account age (days)",
    "num__token_age_days": "Token age (days)",
    "cat__is_proxy_vpn_True": "VPN / Proxy = True",
    "cat__is_proxy_vpn_False": "VPN / Proxy = False",
    "cat__is_new_device_True": "New device = True",
    "cat__is_new_device_False": "New device = False",
    "cat__country_mismatch_True": "Country mismatch",
    "cat__night_txn_True": "Night transaction",
    "cat__weekend_txn_True": "Weekend transaction",
}

def _friendly(feat: str) -> str:
    if feat in _RENAMES:
        return _RENAMES[feat]
    if feat.startswith("num__V"):
        return feat.replace("num__", "").upper()
    for prefix, label in [("cat__device_os_", "Device: "),
                           ("cat__browser_", "Browser: "),
                           ("cat__ip_country_", "IP country: "),
                           ("cat__billing_country_", "Billing: "),
                           ("cat__shipping_country_", "Shipping: ")]:
        if feat.startswith(prefix):
            return label + feat[len(prefix):]
    return feat


# ── beeswarm plot generator ──────────────────────────────────────────
def generate_beeswarm(scenario: dict) -> Path:
    """
    Create a beeswarm plot (like shap.summary_plot) for a subset of
    features relevant to a specific reason code.

    Dots are coloured by actual feature value: blue (low) → red (high).
    """
    # Select top-N features by mean(|SHAP|), but always include the
    # scenario's key features.
    key_idx = [feature_names.index(f) for f in scenario["features"]
               if f in feature_names]
    mean_abs = np.mean(np.abs(SHAP_ALL), axis=0)
    top_n = scenario.get("top_n", 15)
    top_idx_global = np.argsort(mean_abs)[::-1][:top_n].tolist()
    # merge: key features first, then fill from top global
    selected = list(dict.fromkeys(key_idx + top_idx_global))[:top_n]
    selected = selected[::-1]  # bottom → top (highest at top)

    n_sel  = len(selected)
    n_samp = SHAP_ALL.shape[0]

    # Colour map: blue → red (matching the reference image)
    cmap = plt.cm.get_cmap("coolwarm")

    fig, ax = plt.subplots(figsize=(8, 0.45 * n_sel + 1.2))

    for row_i, feat_i in enumerate(selected):
        shap_col = SHAP_ALL[:, feat_i]
        feat_col = X_all[:, feat_i]

        # Normalise feature value to [0, 1] for colour mapping
        fmin, fmax = feat_col.min(), feat_col.max()
        if fmax - fmin > 1e-12:
            normed = (feat_col - fmin) / (fmax - fmin)
        else:
            normed = np.full_like(feat_col, 0.5)

        # Add small vertical jitter so overlapping dots spread out
        jitter = np.random.default_rng(42).uniform(-0.3, 0.3, size=n_samp)

        colours = cmap(normed)

        ax.scatter(
            shap_col,
            np.full(n_samp, row_i) + jitter,
            c=colours,
            s=8,
            alpha=0.75,
            edgecolors="none",
            rasterized=True,
        )

    # Labels
    y_labels = [_friendly(feature_names[i]) for i in selected]
    ax.set_yticks(range(n_sel))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    ax.set_title(f"Reason Code: {scenario['label']}", fontsize=12,
                 fontweight="bold", pad=12)
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--")
    ax.set_ylim(-0.8, n_sel - 0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Colour bar (blue → red)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", fontsize=9)

    plt.tight_layout()
    out_path = OUT / f"shap_reason_{scenario['tag']}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path.name}  ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


# ── main ──────────────────────────────────────────────────────────────
def main():
    print(f"\nModel: {N_FEAT} features, {X_all.shape[0]} samples")
    print(f"Output: {OUT}\n")
    paths = []
    for sc in SCENARIOS:
        print(f"▸ {sc['label']}")
        p = generate_beeswarm(sc)
        paths.append(p)
    print(f"\nDone — {len(paths)} beeswarm plots saved to {OUT}")


if __name__ == "__main__":
    main()
