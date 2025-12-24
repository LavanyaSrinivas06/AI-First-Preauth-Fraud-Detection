from __future__ import annotations

import json
import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = Path("data/processed/train.csv")
OUT_DIR = Path("payloads/incoming")

N_PAYLOADS = 50
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# -----------------------------
# Setup
# -----------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Class"])

means = X.mean()
stds = X.std().replace(0, 1e-6)


# -----------------------------
# Metadata
# -----------------------------
def random_meta(i: int) -> dict:
    ts = datetime.utcnow() - timedelta(minutes=random.randint(0, 10000))
    return {
        "txn_id": f"txn_{i:05d}",
        "timestamp": ts.isoformat() + "Z",
        "merchant_id": f"m_{random.randint(100,999)}",
        "user_id": f"u_{random.randint(1000,9999)}",
        "country": random.choice(["US", "DE", "FR", "NL"]),
        "currency": "EUR",
        "ip_country": random.choice(["US", "DE", "FR", "RU"]),
    }


# -----------------------------
# Feature generators
# -----------------------------
def gen_legit():
    noise = np.random.normal(0, 0.5, size=len(means))
    return (means + noise * stds).to_dict()


def gen_borderline():
    x = means.copy()
    for k in x.sample(10).index:
        x[k] += np.random.choice([2, -2]) * stds[k]
    return x.to_dict()


def gen_extreme():
    x = means.copy()

    for k in x.index:
        if k.startswith("num__V"):
            x[k] += np.random.choice([6, -6]) * stds[k]

    for k in [
        "num__txn_count_5m",
        "num__txn_count_30m",
        "num__txn_count_60m",
        "num__amount_zscore",
    ]:
        if k in x:
            x[k] += np.random.uniform(6, 10) * stds[k]

    return x.to_dict()


# -----------------------------
# MAIN
# -----------------------------
def main():
    for i in range(N_PAYLOADS):
        regime = random.choices(
            ["legit", "borderline", "extreme"],
            weights=[0.7, 0.2, 0.1],
            k=1,
        )[0]

        if regime == "legit":
            features = gen_legit()
        elif regime == "borderline":
            features = gen_borderline()
        else:
            features = gen_extreme()

        payload = {
            "meta": random_meta(i),
            "features": features,
        }

        p = OUT_DIR / f"payload_{i:04d}.json"
        p.write_text(json.dumps(payload, indent=2))

    print(f"✅ Generated {N_PAYLOADS} mixed payloads in {OUT_DIR}")


if __name__ == "__main__":
    main()
