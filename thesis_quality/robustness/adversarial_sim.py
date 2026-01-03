# thesis_quality/robustness/adversarial_sim.py
from __future__ import annotations

import json
import random
import requests
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SNAP_DIR = REPO / "artifacts" / "snapshots" / "feature_snapshots"
RESULTS = REPO / "thesis_quality" / "robustness" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

def load_any_snapshot() -> dict:
    snap = next(iter(SNAP_DIR.glob("rev_*.json")), None)
    if snap is None:
        raise FileNotFoundError(f"No snapshots in {SNAP_DIR}")
    obj = json.loads(snap.read_text(encoding="utf-8"))
    feats = obj.get("features") or obj.get("data") or obj
    return feats

def perturb(features: dict, strength: float = 0.2) -> dict:
    out = dict(features)
    keys = [k for k, v in out.items() if isinstance(v, (int, float))]
    for k in random.sample(keys, k=min(8, len(keys))):
        out[k] = float(out[k]) * (1.0 + random.uniform(-strength, strength))
    return out

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8010")
    p.add_argument("--endpoint", default="/preauth/decision")
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()

    base = load_any_snapshot()
    url = args.base_url.rstrip("/") + args.endpoint

    rows = []
    for i in range(args.n):
        feats = perturb(base, strength=0.3)
        payload = {"transaction_id": f"adv_{i}_{random.randint(0,10**9)}", "features": feats, "meta": {"no_store": True}}
        r = requests.post(url, json=payload, timeout=10)
        rows.append({"i": i, "status": r.status_code, "body": r.text[:200]})

    out = RESULTS / "adversarial_results.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    main()
