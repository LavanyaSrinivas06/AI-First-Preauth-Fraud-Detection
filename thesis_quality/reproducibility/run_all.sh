#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8010}"
ENDPOINT="${ENDPOINT:-/preauth/decision}"

echo "[INFO] Using BASE_URL=$BASE_URL ENDPOINT=$ENDPOINT"
python -V

echo "[STEP 1/4] Evaluation (models + hybrid + plots + table)"
python thesis_quality/evaluation/run_evaluation.py

echo "[STEP 2/4] Drift PSI report"
python thesis_quality/drift/psi.py

echo "[STEP 3/4] Latency benchmark (NO-STORE to avoid SQLite collisions)"
python thesis_quality/benchmarking/latency/run_latency.py \
  --base-url "$BASE_URL" \
  --endpoint "$ENDPOINT" \
  --n 2000 --concurrency 10 \
  --no-store

echo "[STEP 4/4] Load test (k6 if installed, else locust)"
python thesis_quality/benchmarking/load/run_load.py \
  --base-url "$BASE_URL" \
  --endpoint "$ENDPOINT" \
  --seconds 30 \
  --vus 50 \
  --no-store

echo "[OK] Thesis-quality run completed."
