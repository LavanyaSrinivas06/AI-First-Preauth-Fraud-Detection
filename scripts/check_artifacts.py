"""Check presence of required artifact and data files for a reproducible demo.

Prints the expected paths (derived from api.core.config.Settings) and whether each file exists.
Run from the repository root: python3 scripts/check_artifacts.py
"""
import sys
from pathlib import Path

# Ensure repository root is on sys.path so `api` can be imported when running this script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.core.config import get_settings


def check():
    s = get_settings()
    checks = []

    # artifact files
    checks.append((s.abs_xgb_model_path(), "XGBoost model (xgb_model.pkl)"))
    checks.append((s.artifacts_path() / "models" / "active_xgb.json", "active_xgb.json (model registry)"))
    checks.append((s.abs_ae_model_path(), "Autoencoder model"))
    checks.append((s.abs_features_path(), "features.json (feature list)"))
    checks.append((s.artifacts_path() / s.ae_thresholds_path, "AE thresholds JSON"))
    checks.append((s.artifacts_path() / s.ae_baseline_path, "AE baseline npy"))

    # data files
    checks.append((Path(s.root_path()) / "data" / "processed" / "train.csv", "processed train.csv"))
    checks.append((Path(s.root_path()) / "data" / "processed" / "val.csv", "processed val.csv"))

    print("Checking required artifacts and data files:\n")
    missing = []
    for p, label in checks:
        p = Path(p)
        ok = p.exists()
        print(f"- {label}: {p} -> {'FOUND' if ok else 'MISSING'}")
        if not ok:
            missing.append((label, p))

    if missing:
        print("\nSome required files are missing. See REPRODUCE.md for instructions to generate artifacts and processed data.")
        return 2
    print("\nAll required artifacts appear present.")
    return 0


if __name__ == '__main__':
    raise SystemExit(check())
