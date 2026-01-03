# Latency + Stress Summary
- URL: `http://127.0.0.1:8010/preauth/decision`
- Requests: **2000**
- Concurrency: **10**
- Total time: **16.1017s**
- Throughput: **124.21 rps**
- Error rate: **0.001**

## Latency (all responses)
- min: 6.063 ms
- p50: 74.801 ms
- p90: 138.237 ms
- p99: 219.278 ms
- max: 443.393 ms
- mean: 80.326 ms

## Latency (2xx only)
- min: 6.063 ms
- p50: 74.789 ms
- p90: 138.253 ms
- p99: 219.295 ms
- max: 443.393 ms
- mean: 80.331 ms

## Status counts
```json
{
  "2xx": 1997,
  "4xx": 0,
  "5xx": 3,
  "other": 0,
  "exceptions": 0
}
```

## Error samples (first 10)
```json
[
  {
    "status": 500,
    "body": "{\"error\":{\"type\":\"invalid_request_error\",\"code\":\"internal_error\",\"message\":\"IntegrityError: UNIQUE constraint failed: decisions.id\"}}",
    "latency_ms": 128.55329201556742
  },
  {
    "status": 500,
    "body": "{\"error\":{\"type\":\"invalid_request_error\",\"code\":\"internal_error\",\"message\":\"IntegrityError: UNIQUE constraint failed: decisions.id\"}}",
    "latency_ms": 20.011625019833446
  },
  {
    "status": 500,
    "body": "{\"error\":{\"type\":\"invalid_request_error\",\"code\":\"internal_error\",\"message\":\"IntegrityError: UNIQUE constraint failed: decisions.id\"}}",
    "latency_ms": 81.47187484428287
  }
]
```
