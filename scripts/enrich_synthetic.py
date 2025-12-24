#scripts/enrich_synthetic.py
#!/usr/bin/env python3
"""
Synthetic enrichment for Kaggle creditcard.csv (CSV-first).
Adds device, network, velocity, profile, geo, and derived features.
Deterministic with --seed. Defaults to CSV output to avoid extra deps.

Usage:
  python scripts/enrich_synthetic.py \
    --input data/raw/creditcard.csv \
    --output data/processed/enriched.csv \
    --seed 42
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

SCHEMA_PATH = Path("docs/schema_enriched.json")
ANCHOR_TS = pd.Timestamp("2013-09-01 00:00:00", tz="UTC")

# Countries + rough centroids (lat, lon)
COUNTRY_COORDS: Dict[str, Tuple[float, float]] = {
    "DE": (51.1657, 10.4515),
    "US": (37.0902, -95.7129),
    "GB": (55.3781, -3.4360),
    "FR": (46.2276, 2.2137),
    "NL": (52.1326, 5.2913),
    "ES": (40.4637, -3.7492),
    "IT": (41.8719, 12.5674),
    "IN": (20.5937, 78.9629),
    "BR": (-14.2350, -51.9253),
    "CA": (56.1304, -106.3468),
    "AU": (-25.2744, 133.7751),
    "SE": (60.1282, 18.6435),
    "PL": (51.9194, 19.1451),
    "PT": (39.3999, -8.2245),
    "IE": (53.1424, -7.6921),
}
COUNTRIES = list(COUNTRY_COORDS.keys())
DEVICE_OSES = ["Android", "iOS", "Windows", "MacOS"]
BROWSERS = ["Chrome", "Safari", "Edge", "Firefox"]


def deterministic_device_id(time_series: pd.Series) -> pd.Series:
    # Deterministic pseudo device id derived from Time with higher cardinality
    # (reduces collisions vs %10000)
    t = time_series.astype(np.int64)
    dev = (t * 1103515245 + 12345) % 2_000_000  # deterministic LCG
    return ("dev_" + dev.astype(str)).astype("string")


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance (km)."""
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def enrich(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- sanity on base columns
    base_cols = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing base columns: {missing}")

    # sort for rolling/velocity correctness
    df = df.sort_values("Time").reset_index(drop=True)

    # pseudo device & timestamp
    df["device_id"] = deterministic_device_id(df["Time"])
    ts = ANCHOR_TS + pd.to_timedelta(df["Time"], unit="s")
    df["_ts"] = ts

    # Precompute unix seconds once (avoids Series.view deprecation inside groupby)
    df["_ts_s"] = (df["_ts"].astype("int64") // 10**9).astype("int64")


    # --- Device/Browser
    df["device_os"] = rng.choice(DEVICE_OSES, size=len(df))
    df["browser"] = rng.choice(BROWSERS, size=len(df))
    p_new = np.where(df["Class"].values == 1, 0.25, 0.05)
    df["is_new_device"] = (rng.random(len(df)) < p_new)

    # --- Network
    df["ip_country"] = rng.choice(COUNTRIES, size=len(df))
    p_vpn = np.where(df["Class"].values == 1, 0.35, 0.07)
    df["is_proxy_vpn"] = (rng.random(len(df)) < p_vpn)

    # fraud skew higher ip reputation
    base_rep = rng.uniform(0, 1, size=len(df))
    boost = rng.uniform(0.2, 0.6, size=len(df))
    df["ip_reputation"] = np.clip(base_rep + boost * (df["Class"].values == 1), 0, 1).round(3)

    # --- Velocity (per device_id, only up-to-current row)
    def compute_velocity(g: pd.DataFrame) -> pd.DataFrame:
        arr = g["_ts_s"].to_numpy()
        idx = np.arange(len(g))


        def win(seconds: int):
            left = np.searchsorted(arr, arr - seconds, side="left")
            return idx - left + 1  # inclusive count

        g["txn_count_5m"] = win(5 * 60)
        g["txn_count_30m"] = win(30 * 60)
        g["txn_count_60m"] = win(60 * 60)
        # rolling avg amount (proxy for 7d)
        g["avg_amount_7d"] = g["Amount"].rolling(50, min_periods=1).mean().round(2)
        return g
    print(f"Computing velocity features over {df['device_id'].nunique()} devices...")
    df = df.groupby("device_id", group_keys=False, sort=False).apply(compute_velocity)
    print("Velocity features done.")
    # --- Fraud velocity bursts (post-velocity computation)
    burst = (df["Class"].values == 1) & (rng.random(len(df)) < 0.25)

    df.loc[burst, "txn_count_5m"] += rng.integers(2, 6, size=burst.sum())
    df.loc[burst, "txn_count_30m"] += rng.integers(3, 8, size=burst.sum())
    df.loc[burst, "txn_count_60m"] += rng.integers(4, 12, size=burst.sum())

    # --- Profile
    df["account_age_days"] = rng.integers(0, 1001, size=len(df))
    fraud_mask = df["Class"].values == 1
    df.loc[fraud_mask, "account_age_days"] = rng.integers(0, 300, size=fraud_mask.sum())
    df["token_age_days"] = rng.integers(0, 366, size=len(df))
    df["avg_spend_user_30d"] = np.exp(rng.normal(3.0, 1.0, size=len(df))).round(2)

    # --- Geo + distance
    df["billing_country"] = rng.choice(COUNTRIES, size=len(df))
    df["shipping_country"] = rng.choice(COUNTRIES, size=len(df))
    lat_b = df["billing_country"].map(lambda c: COUNTRY_COORDS[c][0]).astype(float)
    lon_b = df["billing_country"].map(lambda c: COUNTRY_COORDS[c][1]).astype(float)
    lat_s = df["shipping_country"].map(lambda c: COUNTRY_COORDS[c][0]).astype(float)
    lon_s = df["shipping_country"].map(lambda c: COUNTRY_COORDS[c][1]).astype(float)
    df["geo_distance_km"] = haversine_km(lat_b, lon_b, lat_s, lon_s).round(1)
    # Fraud tends to have higher country mismatch
    flip = (rng.random(len(df)) < np.where(df["Class"].values == 1, 0.25, 0.02))
    df.loc[flip, "shipping_country"] = rng.choice(COUNTRIES, size=int(flip.sum()))
    df["country_mismatch"] = df["billing_country"] != df["shipping_country"]

    # --- Fraud more likely cross-border / distant
    geo_boost = (df["Class"].values == 1) & (rng.random(len(df)) < 0.30)
    df.loc[geo_boost, "geo_distance_km"] *= rng.uniform(1.3, 2.0, size=geo_boost.sum())

    # --- Derived
    def zscore(s: pd.Series) -> pd.Series:
        mu, sd = s.mean(), s.std(ddof=0)
        return (s - mu) / (sd + 1e-9)

    df["amount_zscore"] = df.groupby("device_id")["Amount"].transform(zscore).round(3)
    hour = df["_ts"].dt.hour
    df["night_txn"] = hour.isin([0, 1, 2, 3, 4, 5, 23])
    df["weekend_txn"] = df["_ts"].dt.weekday.isin([5, 6])

    # cleanup temp)
    df = df.drop(columns=["_ts", "_ts_s"])


    # Optional: align to schema order if present
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r") as f:
            schema = json.load(f)
        desired = [e["field"] for e in schema]
        missing_out = [c for c in desired if c not in df.columns]
        if missing_out:
            raise ValueError(f"Output missing schema fields: {missing_out}")
        df = df[desired]  # keep extras out to stay strict

    # Final null check (Kaggle base has no NaNs; our gen shouldn't add any)
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Nulls found after enrichment in: {bad}")

    return df


def main():
    p = argparse.ArgumentParser(description="Synthetic enrichment generator (CSV-first).")
    p.add_argument("--input", type=Path, default=Path("data/raw/creditcard.csv"))
    p.add_argument("--output", type=Path, default=Path("data/processed/enriched.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--format", choices=["csv", "parquet"], default="csv",
                   help="Output format. 'csv' recommended (no extra deps).")
    args = p.parse_args()

    assert args.input.exists(), f"Input not found: {args.input}"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.input)
    enriched = enrich(base, seed=args.seed)

    if args.format == "csv":
        enriched.to_csv(args.output, index=False)
        print(f"✅ Saved CSV: {args.output}")
    else:
        # Parquet path (optional): only if pyarrow is available
        try:
            enriched.to_parquet(args.output, index=False, engine="pyarrow")
            print(f"✅ Saved Parquet: {args.output}")
        except Exception as e:
            fallback = args.output.with_suffix(".csv")
            enriched.to_csv(fallback, index=False)
            print(f"⚠️ Parquet save failed ({e}). Fallback CSV saved: {fallback}")

    # Simple reproducibility signature
    sig = int(pd.util.hash_pandas_object(enriched.head(200), index=True).sum())
    print(f"Reproducibility signature (head200): {sig}")


if __name__ == "__main__":
    main()
