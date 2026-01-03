import time
import json
import uuid
import statistics
import requests
import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_URL = "http://127.0.0.1:8010"
ENDPOINT = "/preauth/decision"
CSV_PATH = "data/processed/test.csv"
N_REQUESTS = 300
TIMEOUT = 5

OUT_DIR = Path("thesis_quality/benchmarking/latency/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH)
df = df.drop(columns=["Class"], errors="ignore")

rows = df.to_dict(orient="records")

latencies = []
errors = 0

# ---------------- WARMUP ----------------
for _ in range(10):
    payload = {
        "transaction_id": f"warmup_{uuid.uuid4().hex}",
        "features": rows[0]
    }
    requests.post(BASE_URL + ENDPOINT, json=payload, timeout=TIMEOUT)

# ---------------- BENCHMARK ----------------
start_total = time.time()

for i in range(N_REQUESTS):
    payload = {
        "transaction_id": f"bench_{uuid.uuid4().hex}",
        "features": rows[i % len(rows)]
    }

    t0 = time.perf_counter()
    r = requests.post(BASE_URL + ENDPOINT, json=payload, timeout=TIMEOUT)
    dt = (time.perf_counter() - t0) * 1000  # ms

    if r.status_code == 200:
        latencies.append(dt)
    else:
        errors += 1

total_time = time.time() - start_total

# ---------------- METRICS ----------------
report = {
    "requests": N_REQUESTS,
    "errors": errors,
    "error_rate": errors / N_REQUESTS,
    "latency_ms": {
        "min": min(latencies),
        "p50": statistics.quantiles(latencies, n=100)[49],
        "p90": statistics.quantiles(latencies, n=100)[89],
        "p99": statistics.quantiles(latencies, n=100)[98],
        "mean": statistics.mean(latencies),
        "max": max(latencies),
    },
    "throughput_rps": N_REQUESTS / total_time,
    "total_time_sec": total_time,
}

# ---------------- SAVE ----------------
json_path = OUT_DIR / "latency_report.json"
md_path = OUT_DIR / "latency_report.md"

json_path.write_text(json.dumps(report, indent=2))

md_path.write_text(
    f"""# Latency Benchmark Report

**Requests:** {N_REQUESTS}  
**Errors:** {errors}  
**Error Rate:** {report['error_rate']:.4f}  

## Latency (ms)
- Min: {report['latency_ms']['min']:.2f}
- P50: {report['latency_ms']['p50']:.2f}
- P90: {report['latency_ms']['p90']:.2f}
- P99: {report['latency_ms']['p99']:.2f}
- Mean: {report['latency_ms']['mean']:.2f}
- Max: {report['latency_ms']['max']:.2f}

## Throughput
- {report['throughput_rps']:.2f} requests/sec

## Proposal Check
- **Latency target ≤ 150 ms:** {"PASS" if report['latency_ms']['p99'] <= 150 else "FAIL"}
"""
)

print("Latency benchmark complete")
print(json.dumps(report, indent=2))
