import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

from api.core.config import get_settings
from api.services.model_service import predict_from_processed_102

OUT_DIR = Path("demo_payloads/approve")
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
        "meta": {"source": "demo_sampling", "scenario": scenario_name, "category": "approve"},
        "features": features,
    }


def score(settings, features: Dict[str, Any]) -> float:
    p_xgb, _, _, _, _ = predict_from_processed_102(settings, features)
    return float(p_xgb)


def main():
    parser = argparse.ArgumentParser(description="Generate approve demo payloads")
    parser.add_argument("--count", type=int, default=100, help="How many payloads to generate (default: 100)")
    args = parser.parse_args()

    settings = get_settings()
    t_low = settings.xgb_t_low

    df = pd.read_csv(TEST_DATA_PATH)

    # score all rows
    probs: List[float] = []
    for _, r in df.iterrows():
        probs.append(score(settings, _drop_non_features(r.to_dict())))
    df["xgb_probability"] = probs

    # base candidates: clearly low-risk
    # pick a larger candidate pool to allow many perturbations
    candidates = df[df["xgb_probability"] < t_low].sort_values("xgb_probability").head(max(1000, args.count * 5))

    # APPROVE scenario templates (benign toggles)
    templates: List[Tuple[str, Dict[str, Any]]] = [
        ("approve_repeat_customer", {"account_age_boost": 0.8, "token_age_boost": 0.6}),
        ("approve_domestic_no_proxy", {"force_no_proxy": True, "force_no_mismatch": True}),
        ("approve_mobile_safari", {"device_os": "cat__device_os_iOS", "browser": "cat__browser_Safari"}),
        ("approve_desktop_chrome", {"device_os": "cat__device_os_Windows", "browser": "cat__browser_Chrome"}),
        ("approve_weekend_purchase", {"weekend_true": True}),
        ("approve_daytime_purchase", {"night_false": True}),
        ("approve_low_velocity", {"velocity_reduce": True}),
        ("approve_low_geo_distance", {"geo_reduce": True}),
        ("approve_low_ip_risk", {"ip_risk_reduce": True}),
        ("approve_same_billing_shipping", {"force_billing": "cat__billing_country_DE", "force_shipping": "cat__shipping_country_DE"}),
        ("approve_known_device", {"new_device_false": True}),
        ("approve_regular_spend_pattern", {"spend_reduce": True}),
        ("approve_small_amount_signal", {"amount_reduce": True}),
        ("approve_stable_behavior", {"noise_reduce": True}),
        ("approve_standard_browser_edge", {"browser": "cat__browser_Edge"}),
    ]

    _ensure_dir(OUT_DIR)

    saved = 0
    used_hash = set()

    for _, base in candidates.iterrows():
        base_features = _drop_non_features(base.to_dict())

        # Avoid duplicates using hash of sorted keys/values string
        key = json.dumps(base_features, sort_keys=True)
        if key in used_hash:
            continue
        used_hash.add(key)

        for name, cfg in templates:
            if saved >= args.count:
                break

            f = dict(base_features)

            # Apply benign edits
            if cfg.get("force_no_proxy"):
                set_onehot(f, "cat__is_proxy_vpn_", "cat__is_proxy_vpn_False")
            if cfg.get("force_no_mismatch"):
                set_onehot(f, "cat__country_mismatch_", "cat__country_mismatch_False")
            if cfg.get("new_device_false"):
                set_onehot(f, "cat__is_new_device_", "cat__is_new_device_False")
            if cfg.get("night_false"):
                set_onehot(f, "cat__night_txn_", "cat__night_txn_False")
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

            # Mild numeric nudges (kept small to stay realistic)
            if cfg.get("geo_reduce") and "num__geo_distance_km" in f:
                f["num__geo_distance_km"] = float(f["num__geo_distance_km"]) * 0.7
            if cfg.get("velocity_reduce") and "num__txn_count_60m" in f:
                f["num__txn_count_5m"] = float(f.get("num__txn_count_5m", 0.0)) * 0.7
                f["num__txn_count_30m"] = float(f.get("num__txn_count_30m", 0.0)) * 0.7
                f["num__txn_count_60m"] = float(f.get("num__txn_count_60m", 0.0)) * 0.7
            if cfg.get("ip_risk_reduce") and "num__ip_reputation" in f:
                f["num__ip_reputation"] = float(f["num__ip_reputation"]) * 0.7
            if cfg.get("amount_reduce") and "num__Amount" in f:
                f["num__Amount"] = float(f["num__Amount"]) * 0.85
            if cfg.get("spend_reduce") and "num__avg_spend_user_30d" in f:
                f["num__avg_spend_user_30d"] = float(f["num__avg_spend_user_30d"]) * 0.8
            if cfg.get("noise_reduce"):
                # reduce a couple of latent magnitudes slightly (safe, not huge)
                for k in ["num__V7", "num__V10", "num__V14"]:
                    if k in f:
                        f[k] = float(f[k]) * 0.9
            if "account_age_boost" in cfg and "num__account_age_days" in f:
                f["num__account_age_days"] = float(f["num__account_age_days"]) + float(cfg["account_age_boost"])
            if "token_age_boost" in cfg and "num__token_age_days" in f:
                f["num__token_age_days"] = float(f["num__token_age_days"]) + float(cfg["token_age_boost"])

            p = score(settings, f)
            if p < t_low:
                saved += 1
                out = OUT_DIR / f"approve_{saved:03d}_{name}.json"
                out.write_text(json.dumps(build_payload(f, name), indent=2), encoding="utf-8")
                print(f"[APPROVE] Saved {out} (p_xgb={p:.4f})")

        if saved >= args.count:
            break
    if saved < args.count:
        print(f"WARNING: Only generated {saved}/{args.count} APPROVE scenarios. Increase candidate pool or relax edits.")
    else:
        print(f"Done: {args.count} APPROVE scenarios ready.")


if __name__ == "__main__":
    main()
