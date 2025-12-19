from fastapi.testclient import TestClient
from api.main import app


def test_health_ok():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert isinstance(body["preprocess_loaded"], bool)
        assert isinstance(body["version"], str) and body["version"]


def test_predict_valid_payload_returns_placeholders():
    payload = {
        "amount": 10.5,
        "ip_reputation": 0.2,
        "velocity_1h": 0.0,
        "velocity_24h": 1.0,
        "device_os": "iOS",
        "browser": "Safari",
        "ip_country": "de",
        "billing_country": "DE",
        "shipping_country": "DE",
        "hour_of_day": 10,
    }

    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert "X-Latency-ms" in r.headers
        float(r.headers["X-Latency-ms"])

        body = r.json()
        assert body["label"] in {"approve", "review", "block"}
        assert body["score_xgb"] is None
        assert body["score_ae"] is None
        assert body["ensemble_score"] is None
        assert isinstance(body["reason_codes"], list)


def test_predict_missing_required_field_returns_422():
    payload = {
        "ip_reputation": 0.2,
        "velocity_1h": 0.0,
        "velocity_24h": 1.0,
        "device_os": "iOS",
        "browser": "Safari",
        "ip_country": "DE",
        "billing_country": "DE",
        "shipping_country": "DE",
    }
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 422


def test_predict_extra_fields_forbidden_returns_422():
    payload = {
        "amount": 10.5,
        "ip_reputation": 0.2,
        "velocity_1h": 0.0,
        "velocity_24h": 1.0,
        "device_os": "iOS",
        "browser": "Safari",
        "ip_country": "DE",
        "billing_country": "DE",
        "shipping_country": "DE",
        "unexpected": "nope",
    }
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 422
