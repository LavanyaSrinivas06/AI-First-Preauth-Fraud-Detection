# Load Test Report

- Generated: `2026-01-03 18:00:16`
- Runs detected: `3`

## Summary

| run | status | requests | failures | failure_rate | rps | p50_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run1 | degraded | 25034 | 13100 | 52.33% | 210.17 | 1100 | 2000 | 2600 | 5337 |
| run2 | healthy | 23930 | 0 | 0.00% | 200.88 | 980 | 1100 | 1100 | 1323 |
| run3 | degraded | 108163 | 12385 | 11.45% | 905.66 | 160 | 1400 | 1700 | 7861 |

## Run details

### run1

- Endpoint: `UNKNOWN Aggregated`
- Status: **degraded**
- Requests: `25034`  | Failures: `13100`  | Failure rate: `52.33%`
- Throughput: `210.17 req/s`
- Latency (ms): min `0` | avg `923` | p50 `1100` | p90 `1700` | p95 `2000` | p99 `2600` | max `5337`
- Notes:
  - Failure rate 52.33% > 1% threshold

### run2

- Endpoint: `UNKNOWN Aggregated`
- Status: **healthy**
- Requests: `23930`  | Failures: `0`  | Failure rate: `0.00%`
- Throughput: `200.88 req/s`
- Latency (ms): min `52` | avg `969` | p50 `980` | p90 `1000` | p95 `1100` | p99 `1100` | max `1323`

### run3

- Endpoint: `UNKNOWN Aggregated`
- Status: **degraded**
- Requests: `108163`  | Failures: `12385`  | Failure rate: `11.45%`
- Throughput: `905.66 req/s`
- Latency (ms): min `0` | avg `417` | p50 `160` | p90 `1300` | p95 `1400` | p99 `1700` | max `7861`
- Notes:
  - Failure rate 11.45% > 1% threshold
