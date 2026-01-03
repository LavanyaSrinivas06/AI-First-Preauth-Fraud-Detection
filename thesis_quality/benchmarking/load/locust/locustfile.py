# thesis_quality/benchmarking/load/locust/locustfile.py
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from locust import HttpUser, task, between, events

REPO = Path(__file__).resolve().parents[4]
SNAP_DIR = REPO / "artifacts" / "snapshots" / "feature_snapshots"

# Pick any snapshot you already have (full 102 processed features)
DEFAULT_SNAPSHOT = next(iter(SNAP_DIR.glob("rev_*.json")), None)

NO_STORE = False


def _load_snapshot() -> dict:
    if DEFAULT_SNAPSHOT is None:
        raise FileNotFoundError(f"No snapshots found under {SNAP_DIR}")
    obj = json.loads(DEFAULT_SNAPSHOT.read_text(encoding="utf-8"))
    # snapshots are usually: {"transaction_id":..., "data": {...}} or {"features": {...}}
    # We normalize to API expected: {"transaction_id":..., "features": {...}}
    if "features" in obj:
        feats = obj["features"]
    elif "data" in obj:
        feats = obj["data"]
    else:
        # if snapshot is raw structure, adjust here
        feats = obj
    return feats


FEATURES = _load_snapshot()


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--no-store", action="store_true", help="Disable DB writes (prevents SQLite UNIQUE collisions).")


@events.init.add_listener
def on_init(environment, **kwargs):
    global NO_STORE
    opts = environment.parsed_options
    NO_STORE = bool(getattr(opts, "no_store", False))


class WebsiteUser(HttpUser):
    wait_time = between(0.0, 0.02)

    @task
    def preauth_decision(self):
        txid = f"locust_{int(time.time()*1000)}_{random.randint(0, 10**9)}"
        payload = {"transaction_id": txid, "features": FEATURES}
        if NO_STORE:
            payload["meta"] = {"no_store": True}

        with self.client.post("/preauth/decision", json=payload, catch_response=True) as resp:
            if resp.status_code >= 500:
                resp.failure(resp.text[:200])
            else:
                resp.success()
