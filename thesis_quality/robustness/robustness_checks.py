#!/usr/bin/env python3
"""
thesis_quality/robustness/robustness_checks.py
Offline robustness checks — loads XGBoost + Autoencoder models directly
and validates resilience to:
  1. Feature perturbation  (noise injection ±20 %)
  2. Extreme / out-of-range values  (×100, −×100, zero, NaN)
  3. Missing features  (zeroed-out subsets)
  4. Decision-boundary stability  (how much jitter changes a decision?)
Writes results to thesis_quality/robustness/results/robustness_report.json
"""
from __future__ import annotations

import copy, json, math, random, warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
SNAP_DIR = REPO / "artifacts" / "snapshots" / "feature_snapshots"
RESULTS  = REPO / "thesis_quality" / "robustness" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ── load models ──────────────────────────────────────────────────────
import joblib, xgboost as xgb
from tensorflow import keras  # type: ignore

XGB_MODEL_PATH = REPO / "artifacts" / "models" / "xgb_model.pkl"
AE_MODEL_PATH  = REPO / "artifacts" / "models" / "autoencoder_model.keras"
FEATURES_PATH  = REPO / "artifacts" / "preprocess" / "features.json"
AE_THRESH_PATH = REPO / "artifacts" / "thresholds" / "ae_thresholds.json"
XGB_METRICS    = REPO / "artifacts" / "xgb_metrics.json"

xgb_model = joblib.load(XGB_MODEL_PATH)
ae_model  = keras.models.load_model(AE_MODEL_PATH, compile=False)

with open(FEATURES_PATH) as f:
    model_features = json.load(f)["feature_names_after_preprocessing"]
with open(AE_THRESH_PATH) as f:
    ae_cfg = json.load(f)
with open(XGB_METRICS) as f:
    xgb_cfg = json.load(f)

AE_REVIEW  = ae_cfg["review"]
AE_BLOCK   = ae_cfg["block"]
XGB_T_LOW  = 0.05
XGB_T_HIGH = 0.80
XGB_THRESHOLD = xgb_cfg["threshold"]

# ── helpers ──────────────────────────────────────────────────────────
def load_snapshots(n: int = 50) -> List[Dict[str, float]]:
    snaps = sorted(SNAP_DIR.glob("rev_*.json"))[:n]
    out = []
    for s in snaps:
        obj = json.loads(s.read_text())
        out.append(obj)
    return out

def score(features: Dict[str, float]) -> Dict[str, Any]:
    """Score a single feature dict → {p_xgb, ae_error, decision}."""
    vec = [features.get(f, 0.0) for f in model_features]
    X = pd.DataFrame([vec], columns=model_features)

    p_xgb = float(xgb_model.predict_proba(X)[:, 1][0])

    X_np = np.array(vec, dtype=np.float32).reshape(1, -1)
    recon = ae_model.predict(X_np, verbose=0)
    ae_err = float(np.mean((X_np - recon) ** 2))

    # decision logic
    if p_xgb < XGB_T_LOW:
        decision = "APPROVE"
    elif p_xgb >= XGB_T_HIGH:
        decision = "BLOCK"
    elif ae_err >= AE_BLOCK:
        decision = "BLOCK"
    else:
        decision = "REVIEW"

    return {"p_xgb": round(p_xgb, 6), "ae_error": round(ae_err, 4), "decision": decision}


# ── Test 1: Feature perturbation (noise injection) ───────────────────
def test_perturbation(snapshots: List[dict], strength: float = 0.20, trials: int = 5):
    """Perturb each snapshot ±strength and check if decision flips."""
    flips, total = 0, 0
    details: List[dict] = []

    for snap in snapshots:
        base = score(snap)
        for _ in range(trials):
            perturbed = {}
            for k, v in snap.items():
                if isinstance(v, (int, float)) and not math.isnan(v):
                    noise = random.uniform(-strength, strength)
                    perturbed[k] = float(v) * (1.0 + noise)
                else:
                    perturbed[k] = v
            result = score(perturbed)
            total += 1
            if result["decision"] != base["decision"]:
                flips += 1
                details.append({
                    "base_decision": base["decision"],
                    "perturbed_decision": result["decision"],
                    "base_p_xgb": base["p_xgb"],
                    "perturbed_p_xgb": result["p_xgb"],
                })

    return {
        "test": "feature_perturbation",
        "strength": strength,
        "snapshots": len(snapshots),
        "trials_per_snapshot": trials,
        "total_trials": total,
        "decision_flips": flips,
        "flip_rate": round(flips / max(total, 1), 4),
        "stable_rate": round(1 - flips / max(total, 1), 4),
        "sample_flips": details[:5],
    }


# ── Test 2: Extreme / out-of-range values ───────────────────────────
def test_extreme_values(snapshots: List[dict]):
    """Inject extreme values (×100, −×100, 0, NaN) and check the system
    still returns a valid decision without crashing."""
    cases = {
        "scale_100x": lambda v: v * 100,
        "scale_neg100x": lambda v: v * -100,
        "all_zeros": lambda _: 0.0,
        "nan_injection": lambda _: float("nan"),
    }
    results = {}
    for case_name, transform in cases.items():
        crashes, valid, total = 0, 0, 0
        for snap in snapshots[:20]:
            mutated = {}
            for k, v in snap.items():
                if isinstance(v, (int, float)):
                    mutated[k] = transform(v)
                else:
                    mutated[k] = v
            total += 1
            try:
                r = score(mutated)
                if r["decision"] in {"APPROVE", "REVIEW", "BLOCK"}:
                    valid += 1
                else:
                    crashes += 1
            except Exception as e:
                crashes += 1

        results[case_name] = {
            "total": total,
            "valid_responses": valid,
            "crashes": crashes,
            "pass": crashes == 0,
        }
    return {"test": "extreme_values", "cases": results}


# ── Test 3: Missing features (zero-fill subsets) ─────────────────────
def test_missing_features(snapshots: List[dict]):
    """Drop progressively more features (set to 0) and check decision
    stability and that the system does not crash."""
    drop_fracs = [0.10, 0.25, 0.50, 0.75]
    results = {}
    for frac in drop_fracs:
        flips, total, crashes = 0, 0, 0
        for snap in snapshots[:20]:
            base = score(snap)
            keys = [k for k in snap if isinstance(snap[k], (int, float))]
            n_drop = max(1, int(len(keys) * frac))
            dropped = copy.deepcopy(snap)
            for k in random.sample(keys, n_drop):
                dropped[k] = 0.0
            total += 1
            try:
                r = score(dropped)
                if r["decision"] != base["decision"]:
                    flips += 1
            except Exception:
                crashes += 1
        results[f"drop_{int(frac*100)}pct"] = {
            "total": total,
            "decision_flips": flips,
            "flip_rate": round(flips / max(total, 1), 4),
            "crashes": crashes,
        }
    return {"test": "missing_features_zeroed", "cases": results}


# ── Test 4: Decision-boundary jitter ─────────────────────────────────
def test_boundary_stability(snapshots: List[dict]):
    """For transactions near the XGB threshold, measure how much noise
    is needed to flip the decision — a bigger ε → more stable."""
    near_boundary = []
    for snap in snapshots:
        r = score(snap)
        if 0.02 < r["p_xgb"] < 0.12:  # near the xgb_t_low=0.05
            near_boundary.append((snap, r))

    if not near_boundary:
        return {"test": "boundary_stability", "note": "no snapshots near boundary in sample"}

    epsilons = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
    results = {}
    for eps in epsilons:
        flips, trials = 0, 0
        for snap, base in near_boundary:
            for _ in range(10):
                perturbed = {
                    k: float(v) * (1 + random.uniform(-eps, eps))
                    if isinstance(v, (int, float)) else v
                    for k, v in snap.items()
                }
                r = score(perturbed)
                trials += 1
                if r["decision"] != base["decision"]:
                    flips += 1
        results[f"eps_{eps}"] = {
            "boundary_samples": len(near_boundary),
            "trials": trials,
            "flips": flips,
            "flip_rate": round(flips / max(trials, 1), 4),
        }
    return {"test": "boundary_stability", "results": results}


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("[robustness] loading snapshots …")
    snapshots = load_snapshots(50)
    print(f"[robustness] loaded {len(snapshots)} snapshots")

    report = {}

    print("[1/4] Feature perturbation (±20 %) …")
    report["perturbation"] = test_perturbation(snapshots)
    print(f"      flip rate = {report['perturbation']['flip_rate']}")

    print("[2/4] Extreme / out-of-range values …")
    report["extreme"] = test_extreme_values(snapshots)
    for c, r in report["extreme"]["cases"].items():
        print(f"      {c}: pass={r['pass']}")

    print("[3/4] Missing features (zeroed) …")
    report["missing"] = test_missing_features(snapshots)
    for c, r in report["missing"]["cases"].items():
        print(f"      {c}: flip_rate={r['flip_rate']}, crashes={r['crashes']}")

    print("[4/4] Boundary stability …")
    report["boundary"] = test_boundary_stability(snapshots)

    out_path = RESULTS / "robustness_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
