# scripts/make_processed_payloads.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def row_to_payload(df: pd.DataFrame, idx: int) -> dict:
    row = df.iloc[idx].to_dict()
    # ensure JSON-serializable floats
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def main():
    data_path = Path("data/processed/test.csv")
    if not data_path.exists():
        raise SystemExit(f"Missing: {data_path}")

    df = pd.read_csv(data_path)
    if "Class" not in df.columns:
        raise SystemExit("test.csv must contain Class column")

    # Only 102 processed features (drop Class)
    X = df.drop(columns=["Class"])
    assert X.shape[1] == 102, f"Expected 102 features, got {X.shape[1]}"

    # Use your model score distribution? We'll do simple heuristics:
    # Pick a few random rows; you can later replace with "top prob rows" if you want.
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(X), size=6, replace=False).tolist()

    out_dir = Path("tmp_payloads_processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(idxs, start=1):
        payload = row_to_payload(X, idx)
        p = out_dir / f"req_{i}.json"
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved {p} (row={idx})")

    print("\nNow test with:")
    print("curl -X POST http://127.0.0.1:8000/preauth/decision -H 'Content-Type: application/json' --data @tmp_payloads_processed/req_1.json")


if __name__ == "__main__":
    main()
