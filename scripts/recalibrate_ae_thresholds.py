#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


@dataclass(frozen=True)
class Cfg:
    val_csv: Path = Path("data/processed/val.csv")  # processed
    target_col: str = "Class"

    preprocess_path: Path = Path("artifacts/preprocess.joblib")  # not used if val already processed
    ae_model_path: Path = Path("artifacts/autoencoder_model.keras")

    out_thresholds: Path = Path("artifacts/ae_thresholds.json")
    out_baseline: Path = Path("artifacts/ae_baseline_legit_errors.npy")

    p_review: float = 95.0
    p_block: float = 99.5


def reconstruction_error(ae, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X).astype("float32")
    X_rec = ae.predict(X, verbose=0)
    err = np.mean((X - X_rec) ** 2, axis=1)
    return err.astype("float64")


def main() -> None:
    cfg = Cfg()

    if not cfg.val_csv.exists():
        raise FileNotFoundError(f"Missing {cfg.val_csv}")

    print(f"Loading AE: {cfg.ae_model_path}")
    ae = load_model(cfg.ae_model_path, compile=False)

    print(f"Loading val: {cfg.val_csv}")
    df = pd.read_csv(cfg.val_csv)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target col '{cfg.target_col}' in {cfg.val_csv}")

    # val.csv is already processed in your repo (102 features + Class)
    X = df.drop(columns=[cfg.target_col]).values
    y = df[cfg.target_col].astype(int).values

    legit_mask = (y == 0)
    if legit_mask.sum() == 0:
        raise RuntimeError("No legit (Class=0) rows found in val.csv for calibration.")

    X_legit = X[legit_mask]
    print(f"Legit samples for calibration: {X_legit.shape[0]}")

    err_legit = reconstruction_error(ae, X_legit)

    review_thr = float(np.percentile(err_legit, cfg.p_review))
    block_thr = float(np.percentile(err_legit, cfg.p_block))

    cfg.out_thresholds.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_baseline.parent.mkdir(parents=True, exist_ok=True)

    # Save thresholds
    thresholds = {
        "review": review_thr,
        "block": block_thr,
        "p_review": cfg.p_review,
        "p_block": cfg.p_block,
        "n_legit": int(err_legit.shape[0]),
        "source": str(cfg.val_csv),
    }
    cfg.out_thresholds.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    # Save baseline errors (for percentiles later)
    np.save(cfg.out_baseline, np.sort(err_legit))

    print("✅ AE calibration complete")
    print(f"  review threshold (p{cfg.p_review}): {review_thr}")
    print(f"  block  threshold (p{cfg.p_block}): {block_thr}")
    print(f"  wrote: {cfg.out_thresholds}")
    print(f"  wrote: {cfg.out_baseline}")


if __name__ == "__main__":
    main()
