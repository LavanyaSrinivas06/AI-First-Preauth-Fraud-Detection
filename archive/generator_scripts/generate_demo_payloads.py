"""
Generate demo simulation payloads by sampling rows
from the processed test dataset.

This script creates:
- SAFE   example (low XGB probability)
- GRAY   example (borderline XGB probability)
- RISKY  example (high XGB probability)
"""

import json
from pathlib import Path

import pandas as pd
import joblib

from api.core.config import get_settings
from api.services.model_service import predict_from_processed_102


OUTPUT_DIR = Path("demo_payloads")
TEST_DATA_PATH = Path("data/processed/test.csv")


def build_payload(row: pd.Series) -> dict:
    """Convert a processed row into API payload format."""
    features = row.to_dict()
    return {
        "meta": {
            "source": "demo_sampling",
            "note": "Sampled from processed test set"
        },
        "features": features,
    }


def main():
    settings = get_settings()

    print("Loading processed test dataset...")
    df = pd.read_csv(TEST_DATA_PATH)

    print(f"Loaded {len(df)} rows")

    # Score all rows with XGBoost only (fast)
    scores = []
    for _, row in df.iterrows():
        p_xgb, _, _, _, _ = predict_from_processed_102(settings, row.to_dict())
        scores.append(p_xgb)

    df["xgb_probability"] = scores

    # Thresholds
    t_low = settings.xgb_t_low
    t_high = settings.xgb_t_high

    # SAFE: clearly low risk
    safe_row = df[df["xgb_probability"] < t_low].sort_values("xgb_probability").iloc[0]

    # RISKY: clearly high risk
    risky_row = df[df["xgb_probability"] >= t_high].sort_values("xgb_probability", ascending=False).iloc[0]

    # GRAY: closest to midpoint
    mid = (t_low + t_high) / 2
    gray_row = df.iloc[(df["xgb_probability"] - mid).abs().argsort().iloc[0]]

    payloads = {
        "safe": safe_row,
        "gray": gray_row,
        "risky": risky_row,
    }

    for name, row in payloads.items():
        payload = build_payload(row)

        out_path = OUTPUT_DIR / name / f"sim_{name}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved {name.upper()} payload → {out_path}")

        print(f"{name.upper()} XGB probability: {row['xgb_probability']:.4f}")

    print("\nDone. Demo payloads are ready.")


if __name__ == "__main__":
    main()