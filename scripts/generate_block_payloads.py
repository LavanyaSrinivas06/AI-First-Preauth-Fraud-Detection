import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

from api.core.config import get_settings
from api.services.model_service import predict_from_processed_102

OUT_DIR = Path("demo_payloads/block_fraud")
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
        "meta": {"source": "demo_sampling", "scenario": scenario_name, "category": "block_fraud"},
        "features": features,
    }


def score(settings, features: Dict[str, Any]) -> float:
    p_xgb, _, _, _, _ = predict_from_processed_102(settings, features)
    return float(p_xgb)


def main():
    parser = argparse.ArgumentParser(description="Generate block (fraud) demo payloads")
    parser.add_argument("--count", type=int, default=100, help="How many payloads to generate (default: 100)")
    args = parser.parse_args()

    settings = get_settings()
    t_high = settings.xgb_t_high

    df = pd.read_csv(TEST_DATA_PATH)

    probs: List[float] = []
    for _, r in df.iterrows():
        probs.append(score(settings, _drop_non_features(r.to_dict())))
    df["xgb_probability"] = probs

    # base candidates: clearly high risk
    candidates = df[df["xgb_probability"] >= t_high].sort_values("xgb_probability", ascending=False).head(max(1000, args.count * 5))

    templates: List[Tuple[str, Dict[str, Any]]] = [
        ("block_proxy_new_device", {"proxy_true": True, "new_device_true": True}),
        ("block_country_mismatch", {"mismatch_true": True}),
        ("block_night_proxy", {"night_true": True, "proxy_true": True}),
        ("block_velocity_burst", {"velocity_boost": 1.6}),
        ("block_geo_distance_extreme", {"geo_boost": 2.0}),
        ("block_ip_risk_high", {"ip_risk_boost": 1.6}),
        ("block_amount_spike", {"amount_boost": 1.5}),
        ("block_crossborder_shipping", {"force_shipping": "cat__shipping_country_RU", "mismatch_true": True}),
        ("block_crossborder_billing", {"force_billing": "cat__billing_country_RU", "mismatch_true": True}),
        ("block_device_unknown_new", {"device_os": "cat__device_os_Android", "new_device_true": True}),
        ("block_browser_change", {"browser": "cat__browser_Firefox"}),
        ("block_weekend_night", {"weekend_true": True, "night_true": True}),
        ("block_short_account_age", {"account_age_reduce": True}),
        ("block_recent_token", {"token_age_reduce": True}),
        ("block_combined_signals", {"proxy_true": True, "mismatch_true": True, "geo_boost": 1.5}),
    ]

    _ensure_dir(OUT_DIR)

    saved = 0
    used_hash = set()

    for _, base in candidates.iterrows():
        base_features = _drop_non_features(base.to_dict())

        key = json.dumps(base_features, sort_keys=True)
        if key in used_hash:
            continue
        used_hash.add(key)

        for name, cfg in templates:
            if saved >= args.count:
                break

            f = dict(base_features)

            if cfg.get("proxy_true"):
                set_onehot(f, "cat__is_proxy_vpn_", "cat__is_proxy_vpn_True")
            if cfg.get("new_device_true"):
                set_onehot(f, "cat__is_new_device_", "cat__is_new_device_True")
            if cfg.get("mismatch_true"):
                set_onehot(f, "cat__country_mismatch_", "cat__country_mismatch_True")
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

            if "geo_boost" in cfg and "num__geo_distance_km" in f:
                f["num__geo_distance_km"] = float(f["num__geo_distance_km"]) * float(cfg["geo_boost"])
            if "velocity_boost" in cfg and "num__txn_count_60m" in f:
                f["num__txn_count_5m"] = float(f.get("num__txn_count_5m", 0.0)) * float(cfg["velocity_boost"])
                f["num__txn_count_30m"] = float(f.get("num__txn_count_30m", 0.0)) * float(cfg["velocity_boost"])
                f["num__txn_count_60m"] = float(f.get("num__txn_count_60m", 0.0)) * float(cfg["velocity_boost"])
            if "ip_risk_boost" in cfg and "num__ip_reputation" in f:
                f["num__ip_reputation"] = float(f["num__ip_reputation"]) * float(cfg["ip_risk_boost"])
            if "amount_boost" in cfg and "num__Amount" in f:
                f["num__Amount"] = float(f["num__Amount"]) * float(cfg["amount_boost"])
            if cfg.get("account_age_reduce") and "num__account_age_days" in f:
                f["num__account_age_days"] = float(f["num__account_age_days"]) * 0.6
            if cfg.get("token_age_reduce") and "num__token_age_days" in f:
                f["num__token_age_days"] = float(f["num__token_age_days"]) * 0.6

            p = score(settings, f)
            if p >= t_high:
                saved += 1
                out = OUT_DIR / f"block_{saved:03d}_{name}.json"
                out.write_text(json.dumps(build_payload(f, name), indent=2), encoding="utf-8")
                print(f"[BLOCK] Saved {out} (p_xgb={p:.4f})")

        if saved >= args.count:
            break

    if saved < args.count:
        print(f"WARNING: Only generated {saved}/{args.count} BLOCK scenarios. Increase candidate pool or adjust edits.")
    else:
        print(f"Done: {args.count} BLOCK scenarios ready.")


if __name__ == "__main__":
    main()
