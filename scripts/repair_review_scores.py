"""
Recompute model scores (XGB probability + AE) for reviews that have suspect/default values.

Usage:
  .venv/bin/python scripts/repair_review_scores.py

This will scan recent reviews (or all) and for rows where score_xgb == 0.5 or ae_bucket in (NULL,'unknown'),
will attempt to recompute using `predict_from_processed_102` and call `set_review_scores` to persist values.
"""
import sys
import os
from pathlib import Path
import json

sys.path.insert(0, os.path.abspath('.'))
from api.core.config import get_settings
from api.services.store import get_review_by_id, set_review_scores
from api.services.store import load_review_queue
from api.services.model_service import predict_from_processed_102, ensure_loaded


def main(limit=200):
    s = get_settings()
    sqlite_path = s.abs_sqlite_path()

    # load recent reviews (all statuses)
    items = load_review_queue(sqlite_path, limit=limit, status='all')
    art = ensure_loaded(s)
    repaired = []
    for it in items:
        rid = it['id']
        score = it.get('score_xgb')
        ae_bkt = it.get('ae_bucket')
        processed = it.get('processed_features_json')
        if isinstance(processed, str):
            try:
                processed = json.loads(processed)
            except Exception:
                processed = None

        needs = False
        if score is None or float(score) == 0.5:
            needs = True
        if not ae_bkt or ae_bkt in (None, '', 'unknown'):
            needs = True

        if not needs:
            continue

        if not isinstance(processed, dict):
            print(f"Skipping {rid}: no processed features to recompute")
            continue

        try:
            p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = predict_from_processed_102(s, processed)
            ok = set_review_scores(sqlite_path, rid, p_xgb, ae_err, ae_pct, ae_bkt)
            repaired.append((rid, ok, p_xgb, ae_err, ae_pct, ae_bkt))
            print(f"Repaired {rid}: p_xgb={p_xgb:.3f}, ae_err={ae_err}, ae_pct={ae_pct}, ae_bkt={ae_bkt}")
        except Exception as e:
            print(f"Failed to repair {rid}: {e}")

    print(f"\nRepaired {len([r for r in repaired if r[1]])} reviews (attempted {len(repaired)}).")


if __name__ == '__main__':
    main()
