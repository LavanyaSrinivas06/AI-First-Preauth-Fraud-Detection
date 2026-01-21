"""
Generate SHAP images for up to N demo review payloads and persist their shap_path in the reviews DB.

Usage:
  .venv/bin/python scripts/generate_and_store_shap.py

The script will look for JSON files under `demo_payloads/review/*.json` (defaults) and fall back to other demo_payloads subfolders.
It will save each payload as a REVIEW (via `save_review_if_needed`) with `processed_features_json` when possible,
then call `generate_shap_png` to create PNG and write a `.input.json` alongside it, and finally persist `shap_path` into the reviews table.

"""
import sys
import os
from pathlib import Path
import json
import uuid
import time

# ensure project root on sys.path
sys.path.insert(0, os.path.abspath("."))

from api.core.config import get_settings
from api.services.store import save_review_if_needed, set_review_shap_path
from api.services import store as store_module
from api.services.model_service import ensure_loaded, predict_from_processed_102
from dashboard.utils.explainability import generate_shap_png, shap_path


def find_payload_files(limit=15):
    root = Path("demo_payloads")
    files = list(root.glob("review/*.json"))
    if not files:
        # fallback: gather across subfolders
        files = [p for p in root.rglob("*.json") if p.is_file()]
    files = sorted(files)[:limit]
    return files


def load_features_from_payload(p: Path):
    obj = json.loads(p.read_text())
    # common shapes: {"features": {...}} or {"payload_min": {...}} or full payload
    for k in ("features", "payload_min", "payload", "features_full"):
        if k in obj and isinstance(obj[k], dict):
            return obj[k]
    # if the file itself is the features dict
    if isinstance(obj, dict) and any(k.startswith("num__") or k.startswith("cat__") for k in obj.keys()):
        return obj
    # as last resort, try top-level items
    return obj


def main(limit=15, pause_between=0.5):
    s = get_settings()
    sqlite_path = s.abs_sqlite_path()
    print(f"Using sqlite: {sqlite_path}")

    payload_files = find_payload_files(limit=limit)
    if not payload_files:
        print("No payload files found under demo_payloads/ — aborting.")
        return

    print(f"Found {len(payload_files)} payload files, will process up to {limit}.")

    # load artifacts once
    art = ensure_loaded(s)
    model = getattr(art, 'xgb', None)
    if model is None:
        print("Warning: no xgb model found in loaded artifacts — generate_shap may fail.")

    created = []

    for p in payload_files:
        try:
            features = load_features_from_payload(p)
            # ensure we have a dict for processed features JSON
            processed_json = features if isinstance(features, dict) else None
            # If this is a processed_102 features dict, compute model scores
            p_xgb = None
            ae_err = None
            ae_pct = None
            ae_bkt = None
            payload_hash = str(uuid.uuid4())
            try:
                if isinstance(processed_json, dict):
                    p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = predict_from_processed_102(s, processed_json)
            except Exception as e:
                print(f"Warning: could not compute scores for {p.name}: {e}")

            rid = save_review_if_needed(
                sqlite_path=sqlite_path,
                decision='REVIEW',
                payload=features,
                meta={'source_file': str(p.name)},
                p_xgb=float(p_xgb) if p_xgb is not None else 0.5,
                ae_err=float(ae_err) if ae_err is not None else None,
                payload_hash=payload_hash,
                reason_codes=[],
                ae_percentile=float(ae_pct) if ae_pct is not None else None,
                ae_bucket=ae_bkt if ae_bkt is not None else 'unknown',
                feature_path=None,
                model_version=getattr(art, 'model_version', 'unknown'),
                processed_features_json=json.dumps(processed_json) if processed_json is not None else None,
            )
            print(f"Saved review {rid} for payload {p.name}")

            # generate shap png (this will also write a .input.json next to it)
            path = generate_shap_png(rid, model, features)
            print(f"Generated SHAP png: {path}")

            # persist shap path in DB
            try:
                set_review_shap_path(sqlite_path, rid, str(path))
                print(f"Persisted shap_path for review {rid}")
            except Exception as e:
                print(f"Failed to persist shap_path for {rid}: {e}")

            created.append((rid, p.name, str(path)))

            # small pause to avoid overwhelming resources
            time.sleep(pause_between)

        except Exception as e:
            print(f"ERROR processing {p}: {e}")

    print("\nSummary:")
    for rid, name, path in created:
        print(f" - review {rid} from {name} -> {path}")

    print(f"Processed {len(created)} payloads.")


if __name__ == '__main__':
    main()
