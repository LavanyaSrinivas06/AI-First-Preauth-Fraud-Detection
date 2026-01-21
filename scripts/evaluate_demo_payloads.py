#!/usr/bin/env python3
"""Evaluate model decisions on demo payloads under demo_payloads/.

Scans all JSON files under demo_payloads/, expects each file to have:
  {
    "meta": {"category": "approve"|"block_fraud"|...},
    "features": { <processed num__/cat__ features> }
  }

Mapping of categories to decisions (heuristic):
  approve, safe -> APPROVE
  block_fraud, risky -> BLOCK
  gray, review_legit -> REVIEW

This script uses the project's model loader (local) and will require artifacts to be present.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict


def load_settings_and_predictor():
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from api.core.config import get_settings
    from api.services.model_service import predict_from_processed_102

    settings = get_settings()
    return settings, predict_from_processed_102


def canonical_label(category: str) -> str:
    c = str(category).lower()
    if c in {"approve", "safe"}:
        return "APPROVE"
    if c in {"block_fraud", "risky"}:
        return "BLOCK"
    if c in {"gray", "review_legit", "review"}:
        return "REVIEW"
    # fallback: try keywords
    if "block" in c or "fraud" in c:
        return "BLOCK"
    if "approve" in c or "safe" in c:
        return "APPROVE"
    return "REVIEW"


def decision_from_scores(p_xgb, ae_bkt, settings):
    if p_xgb < settings.xgb_t_low:
        return "APPROVE"
    if p_xgb >= settings.xgb_t_high:
        return "BLOCK"
    if ae_bkt == "extreme":
        return "BLOCK"
    return "REVIEW"


def main():
    base = Path("demo_payloads")
    if not base.exists():
        print("demo_payloads/ not found")
        return 2

    settings, predictor = load_settings_and_predictor()

    files = list(base.rglob("*.json"))
    if not files:
        print("No JSON payloads found under demo_payloads/")
        return 2

    tot = 0
    correct = 0
    conf = defaultdict(Counter)  # conf[true][pred]

    details = []

    for f in sorted(files):
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Failed reading {f}: {e}")
            continue

        meta = obj.get("meta") or {}
        cat = meta.get("category") or meta.get("scenario") or "unknown"
        true = canonical_label(cat)

        features = obj.get("features") or obj.get("payload_min") or {}
        if not features:
            print(f"Skipping {f} missing features")
            continue

        try:
            p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = predictor(settings, features)
            pred = decision_from_scores(p_xgb, ae_bkt, settings)
        except Exception as e:
            pred = f"ERROR:{type(e).__name__}"

        tot += 1
        if pred == true:
            correct += 1
        conf[true][pred] += 1
        details.append((f.name, true, pred, p_xgb if not isinstance(pred, str) or not pred.startswith("ERROR") else None, ae_bkt))

    # report
    print(f"Evaluated {tot} demo payloads")
    if tot:
        acc = 100.0 * correct / tot
        print(f"Accuracy (simple exact-match): {acc:.2f}% ({correct}/{tot})")

        print("\nConfusion matrix (rows=true, cols=pred):")
        all_labels = sorted({*conf.keys(), *(k for d in conf.values() for k in d.keys())})
        print("\t" + "\t".join(all_labels))
        for t in all_labels:
            row = [str(conf[t].get(p, 0)) for p in all_labels]
            print(f"{t}\t" + "\t".join(row))

    # optionally write details
    out = Path("demo_eval_results.csv")
    with out.open("w", encoding="utf-8") as fh:
        fh.write("file,true,pred,xgb,ae_bkt\n")
        for fn, t, p, xgb, aeb in details:
            fh.write(f"{fn},{t},{p},{'' if xgb is None else xgb},{'' if aeb is None else aeb}\n")

    print(f"Wrote detailed results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
