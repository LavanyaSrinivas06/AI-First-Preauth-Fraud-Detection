#!/usr/bin/env python3
"""Simple demo tool: run the model on a single payload (JSON) and show decision + metrics.

Usage examples:
  # processed payload (contains num__/cat__ features)
  scripts/demo_predict.py --json '{"num__V1": 1.2, "cat__x": 1}'

  # raw payload in a file (will attempt to run preprocessing if available)
  scripts/demo_predict.py --file payload.json

  # with a ground-truth label (APPROVE or BLOCK) to check correctness
  scripts/demo_predict.py --json '{...}' --label APPROVE

This is a thin convenience wrapper around the project's model loader and scoring code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def load_settings_and_artifacts():
    # ensure repo root is on sys.path
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from api.core.config import get_settings
    from api.services.model_service import ensure_loaded

    settings = get_settings()
    artifacts = ensure_loaded(settings)
    return settings, artifacts


def try_preprocess(settings, raw_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Attempt to run the saved preprocessing pipeline to produce processed features.
    Returns a dict mapping processed feature names -> values, or None if preprocessing not available.
    """
    preproc_path = settings.abs_preprocess_path()
    if not preproc_path.exists():
        return None

    try:
        import joblib
        import pandas as pd

        pre = joblib.load(preproc_path)
        # Try to predict feature names
        if hasattr(pre, "get_feature_names_out"):
            # Create one-row dataframe from raw payload and transform
            df = pd.DataFrame([raw_payload])
            X = pre.transform(df)
            names = list(pre.get_feature_names_out())
            # If transform returns a 1-d array, wrap appropriately
            import numpy as np

            X = np.asarray(X)
            if X.ndim == 1:
                vals = X.tolist()
            else:
                vals = X[0].tolist()

            return {n: float(v) for n, v in zip(names, vals)}
        else:
            print("Preprocessor exists but does not expose get_feature_names_out(); skipping preprocess.")
            return None
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return None


def decision_from_scores(p_xgb: float, ae_bkt: Optional[str], settings) -> str:
    if p_xgb < settings.xgb_t_low:
        return "APPROVE"
    if p_xgb >= settings.xgb_t_high:
        return "BLOCK"
    # gray zone
    if ae_bkt == "extreme":
        return "BLOCK"
    return "REVIEW"


def run_single(payload: Dict[str, Any], label: Optional[str] = None):
    settings, artifacts = load_settings_and_artifacts()

    # detect processed payload
    is_processed = any(isinstance(k, str) and (k.startswith("num__") or k.startswith("cat__")) for k in payload.keys())

    from api.services.model_service import predict_from_processed_102

    processed = None
    if is_processed:
        processed = payload
    else:
        processed = try_preprocess(settings, payload)

    if processed is None:
        print("Could not obtain processed features for this payload. Provide processed features (num__/cat__) or run preprocessing first.")
        return

    try:
        p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = predict_from_processed_102(settings, processed)
    except Exception as e:
        print(f"Prediction failed: {type(e).__name__}: {e}")
        return

    decision = decision_from_scores(p_xgb, ae_bkt, settings)

    print("--- Prediction result ---")
    print(f"payload_hash: {payload_hash}")
    print(f"xgb_prob: {p_xgb:.6f}")
    print(f"ae_error: {ae_err}")
    print(f"ae_percentile: {ae_pct}")
    print(f"ae_bucket: {ae_bkt}")
    print(f"decision: {decision}")

    if label is not None:
        lab = str(label).upper()
        ok = (lab == decision) or (lab == "BLOCK" and decision == "BLOCK")
        print(f"label: {lab} -> match: {ok}")


def main():
    p = argparse.ArgumentParser(description="Demo: run model on a payload")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="Inline JSON payload string")
    group.add_argument("--file", help="Path to JSON payload file")
    p.add_argument("--label", help="Optional ground-truth label (APPROVE or BLOCK)")
    args = p.parse_args()

    if args.json:
        payload = json.loads(args.json)
    else:
        with open(args.file) as f:
            payload = json.load(f)

    run_single(payload, label=args.label)


if __name__ == "__main__":
    main()
