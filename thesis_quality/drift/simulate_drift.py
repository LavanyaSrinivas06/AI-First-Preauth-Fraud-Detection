from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def simulate_drift(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()

    # 1) Amount shift (if exists)
    for col in ["num__Amount", "amount", "Amount"]:
        if col in out.columns:
            x = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy()
            x = x * 1.5 + rng.normal(0, np.std(x) * 0.05 + 1e-6, size=len(x))
            out[col] = x
            break

    # 2) Add noise to V features if present
    v_cols = [c for c in out.columns if isinstance(c, str) and c.startswith("num__V")]
    if v_cols:
        for c in v_cols[:30]:  # enough for demo
            x = pd.to_numeric(out[c], errors="coerce").fillna(0.0).to_numpy()
            out[c] = x + rng.normal(0, 0.15, size=len(x))

    # 3) Slightly increase rare binary flags (if any)
    bin_like = []
    for c in out.columns:
        s = out[c]
        if s.dropna().isin([0, 1]).mean() > 0.98:
            bin_like.append(c)
    for c in bin_like[:50]:
        s = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
        flip = rng.random(len(s)) < 0.01
        s[flip] = 1
        out[c] = s

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", default="thesis_quality/drift/results/current_simulated.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    inp = Path(args.in_csv)
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    drifted = simulate_drift(df, seed=args.seed)
    drifted.to_csv(outp, index=False)
    print("Wrote:", outp)


if __name__ == "__main__":
    main()
