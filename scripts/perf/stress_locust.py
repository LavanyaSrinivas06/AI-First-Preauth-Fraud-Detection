# scripts/perf/stress_locust.py
from __future__ import annotations

import json
from pathlib import Path

from locust import HttpUser, task, between

PAYLOAD_PATH = Path("payloads/incoming/payload_0000.json")
PAYLOAD = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))

class ApiUser(HttpUser):
    wait_time = between(0.0, 0.02)

    @task
    def preauth_decision(self):
        with self.client.post("/preauth/decision", json=PAYLOAD, catch_response=True) as r:
            if not (200 <= r.status_code < 300):
                r.failure(f"status={r.status_code} body={r.text[:120]}")
            else:
                r.success()
