import http from "k6/http";
import { check, sleep } from "k6";

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8010";
const ENDPOINT = __ENV.ENDPOINT || "/preauth/decision";
const NO_STORE = __ENV.NO_STORE === "1";

const FEATURES_URL = `${BASE_URL}${ENDPOINT}`;

export const options = {
  vus: __ENV.VUS ? parseInt(__ENV.VUS) : 50,
  duration: __ENV.DURATION || "30s",
};

function uuidLike() {
  // good enough uniqueness for load tests
  return `${Date.now()}_${Math.random().toString(16).slice(2)}_${__VU}_${__ITER}`;
}

export default function () {
  // Minimal payload: processed features must be present
  // For k6, we do NOT load CSV. We send a small static valid-like vector.
  // Your API validates required fields; so you should swap this with a real snapshot if needed.
  // If your API requires all 102 features strictly, use Locust instead (it loads snapshot JSON).
  const features = {
    "num__V1": 0.0,
    "num__V2": 0.0,
    "num__V3": 0.0,
    "num__V4": 0.0,
    "num__V5": 0.0,
    "num__V6": 0.0,
    "num__V7": 0.0,
    "num__V8": 0.0,
    "num__V9": 0.0,
    "num__V10": 0.0,
  };

  const payload = {
    transaction_id: `k6_${uuidLike()}`,
    features: features,
    meta: { no_store: NO_STORE },
  };

  const res = http.post(FEATURES_URL, JSON.stringify(payload), {
    headers: { "Content-Type": "application/json" },
    timeout: "10s",
  });

  check(res, {
    "status is 2xx or 4xx": (r) => r.status >= 200 && r.status < 500,
  });

  sleep(0.01);
}
