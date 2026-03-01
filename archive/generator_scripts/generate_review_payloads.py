import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

from api.core.config import get_settings
from api.services.model_service import predict_from_processed_102

OUT_DIR = Path("demo_payloads/review_legit")
TEST_DATA_PATH = Path("data/processed/test.csv")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _drop_non_features(d: Dict[str, Any]) -> Dict[str, Any]:
    d.pop("Class", None)
    d.pop("xgb_probability", None)
    return d


def set_onehot(row: Dict[str, Any], prefix: str, chosen_key: str) -> None:
    keys = [k for k in row.keys() if k.startswith(prefix)]
    for k in keys:
        row[k] = 0.0
    if chosen_key in row:
        row[chosen_key] = 1.0


def build_payload(features: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    return {
        "meta": {"source": "demo_sampling", "scenario": scenario_name, "category": "review_legit"},
        "features": features,
    }


def score(settings, features: Dict[str, Any]) -> float:
    p_xgb, _, _, _, _ = predict_from_processed_102(settings, features)
    return float(p_xgb)


def main():
    parser = argparse.ArgumentParser(description="Generate review (gray) demo payloads")
    parser.add_argument("--count", type=int, default=100, help="How many payloads to generate (default: 100)")
    args = parser.parse_args()

    settings = get_settings()
    t_low = settings.xgb_t_low
    t_high = settings.xgb_t_high
    mid = (t_low + t_high) / 2

    df = pd.read_csv(TEST_DATA_PATH)

    probs: List[float] = []
    for _, r in df.iterrows():
        probs.append(score(settings, _drop_non_features(r.to_dict())))
    df["xgb_probability"] = probs

    # base candidates: inside gray zone
    gray = df[(df["xgb_probability"] >= t_low) & (df["xgb_probability"] < t_high)]
    # prioritize near mid to keep stable in REVIEW; take a larger head to support many variants
    gray = gray.iloc[(gray["xgb_probability"] - mid).abs().argsort()].head(max(1200, args.count * 6))

    templates: List[Tuple[str, Dict[str, Any]]] = [
        ("review_travel_new_device", {"new_device_true": True, "mismatch_true": True, "geo_boost": 1.3}),
        ("review_vpn_false_positive", {"proxy_true": True, "mismatch_false": True, "geo_reduce": True}),
        ("review_late_night_purchase", {"night_true": True}),
        ("review_weekend_purchase", {"weekend_true": True}),
        ("review_browser_change_firefox", {"browser": "cat__browser_Firefox"}),
        ("review_browser_change_edge", {"browser": "cat__browser_Edge"}),
        ("review_device_change_android", {"device_os": "cat__device_os_Android", "new_device_true": True}),
        ("review_device_change_macos", {"device_os": "cat__device_os_MacOS", "new_device_true": True}),
        ("review_crossborder_shipping", {"force_shipping": "cat__shipping_country_NL", "mismatch_true": True}),
        ("review_crossborder_billing", {"force_billing": "cat__billing_country_NL", "mismatch_true": True}),
        ("review_velocity_spike_small", {"velocity_boost": 1.2}),
        ("review_amount_spike_moderate", {"amount_boost": 1.15}),
        ("review_geo_distance_spike", {"geo_boost": 1.6}),
        ("review_low_account_age", {"account_age_reduce": True}),
        ("review_token_recent", {"token_age_reduce": True}),
    ]

    _ensure_dir(OUT_DIR)

    saved = 0
    used_hash = set()

    for _, base in gray.iterrows():
        base_features = _drop_non_features(base.to_dict())

        key = json.dumps(base_features, sort_keys=True)
        if key in used_hash:
            continue
        used_hash.add(key)

        for name, cfg in templates:
            if saved >= args.count:
                break

            f = dict(base_features)

            if cfg.get("new_device_true"):
                set_onehot(f, "cat__is_new_device_", "cat__is_new_device_True")
            if cfg.get("mismatch_true"):
                set_onehot(f, "cat__country_mismatch_", "cat__country_mismatch_True")
            if cfg.get("mismatch_false"):
                set_onehot(f, "cat__country_mismatch_", "cat__country_mismatch_False")
            if cfg.get("proxy_true"):
                set_onehot(f, "cat__is_proxy_vpn_", "cat__is_proxy_vpn_True")
            if cfg.get("night_true"):
                set_onehot(f, "cat__night_txn_", "cat__night_txn_True")
            if cfg.get("weekend_true"):
                set_onehot(f, "cat__weekend_txn_", "cat__weekend_txn_True")
            if cfg.get("device_os"):
                set_onehot(f, "cat__device_os_", cfg["device_os"])
            if cfg.get("browser"):
                set_onehot(f, "cat__browser_", cfg["browser"])
            if cfg.get("force_billing"):
                set_onehot(f, "cat__billing_country_", cfg["force_billing"])
            if cfg.get("force_shipping"):
                set_onehot(f, "cat__shipping_country_", cfg["force_shipping"])

            if cfg.get("geo_reduce") and "num__geo_distance_km" in f:
                f["num__geo_distance_km"] = float(f["num__geo_distance_km"]) * 0.8
            if "geo_boost" in cfg and "num__geo_distance_km" in f:
                f["num__geo_distance_km"] = float(f["num__geo_distance_km"]) * float(cfg["geo_boost"])
            if cfg.get("velocity_boost") and "num__txn_count_60m" in f:
                f["num__txn_count_5m"] = float(f.get("num__txn_count_5m", 0.0)) * float(cfg["velocity_boost"])
                f["num__txn_count_30m"] = float(f.get("num__txn_count_30m", 0.0)) * float(cfg["velocity_boost"])
                f["num__txn_count_60m"] = float(f.get("num__txn_count_60m", 0.0)) * float(cfg["velocity_boost"])
            if "amount_boost" in cfg and "num__Amount" in f:
                f["num__Amount"] = float(f["num__Amount"]) * float(cfg["amount_boost"])
            if cfg.get("account_age_reduce") and "num__account_age_days" in f:
                f["num__account_age_days"] = float(f["num__account_age_days"]) * 0.7
            if cfg.get("token_age_reduce") and "num__token_age_days" in f:
                f["num__token_age_days"] = float(f["num__token_age_days"]) * 0.7

            p = score(settings, f)
            if t_low <= p < t_high:
                saved += 1
                out = OUT_DIR / f"review_{saved:03d}_{name}.json"
                out.write_text(json.dumps(build_payload(f, name), indent=2), encoding="utf-8")
                print(f"[REVIEW] Saved {out} (p_xgb={p:.4f})")

        if saved >= args.count:
            break

    if saved < args.count:
        print(f"WARNING: Only generated {saved}/{args.count} REVIEW scenarios. Increase candidate pool or relax edits.")
    else:
        print(f"Done: {args.count} REVIEW scenarios ready.")


if __name__ == "__main__":
    main()
