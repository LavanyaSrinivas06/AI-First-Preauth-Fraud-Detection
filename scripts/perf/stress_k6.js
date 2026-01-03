// scripts/perf/stress_k6.js
import http from 'k6/http';
import { check, sleep } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://127.0.0.1:8010';
const ENDPOINT = __ENV.ENDPOINT || '/preauth/decision';

// Keep payload tiny: we load one sample processed payload.
// If you want raw checkout payloads, swap this to your sample_request.json.
const payload = JSON.parse(open('./payloads/incoming/payload_0000.json'));

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '20s', target: 20 },
        { duration: '40s', target: 50 },
        { duration: '30s', target: 50 },
        { duration: '20s', target: 0 },
      ],
      gracefulRampDown: '10s',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],      // <1% failures
    http_req_duration: ['p(95)<250'],    // p95 under 250ms (tune per your target)
  },
};

export default function () {
  const url = `${BASE_URL}${ENDPOINT}`;
  const res = http.post(url, JSON.stringify(payload), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '5s',
  });

  check(res, {
    'status is 2xx': (r) => r.status >= 200 && r.status < 300,
  });

  sleep(0.01);
}
