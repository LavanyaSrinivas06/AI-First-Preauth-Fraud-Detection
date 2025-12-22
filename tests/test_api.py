# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app


def _full_valid_payload(overrides=None):
    base = {
        # categorical_features
        "device_id": "dev_1",
        "device_os": "iOS",
        "browser": "Safari",
        "is_new_device": False,
        "ip_country": "DE",
        "is_proxy_vpn": False,
        "billing_country": "DE",
        "shipping_country": "DE",
        "night_txn": False,
        "weekend_txn": False,

        # numerical_features V1..V28
        **{f"V{i}": 0.0 for i in range(1, 29)},

        # IMPORTANT: we send "amount" (alias) and it maps to field "Amount"
        "amount": 10.5,
        "ip_reputation": 0.2,

        "txn_count_5m": 0.0,
        "txn_count_30m": 0.0,
        "txn_count_60m": 0.0,
        "avg_amount_7d": 0.0,
        "account_age_days": 100.0,
        "token_age_days": 10.0,
        "avg_spend_user_30d": 50.0,
        "geo_distance_km": 1.0,
        "amount_zscore": 0.0,
    }
    if overrides:
        base.update(overrides)
    return base


def test_health_ok():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert isinstance(body["preprocess_loaded"], bool)
        assert isinstance(body["version"], str) and body["version"]


def test_predict_valid_payload_returns_scores():
    payload = _full_valid_payload()

    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert "X-Latency-ms" in r.headers
        float(r.headers["X-Latency-ms"])

        body = r.json()
        assert body["label"] in {"approve", "review", "block"}

        # for FPN-12 these should be populated floats
        assert body["score_xgb"] is None or (0.0 <= float(body["score_xgb"]) <= 1.0)
        assert body["score_ae"] is None or (0.0 <= float(body["score_ae"]) <= 1.0)
        assert body["ensemble_score"] is None or (0.0 <= float(body["ensemble_score"]) <= 1.0)
        assert isinstance(body["reason_codes"], list)


def test_predict_missing_required_field_returns_422():
    payload = _full_valid_payload()
    payload.pop("device_id")  # remove required field

    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 422


def test_predict_extra_fields_forbidden_returns_422():
    payload = _full_valid_payload({"unexpected": "nope"})

    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 422


@pytest.mark.parametrize("bad_key,bad_value", [
    ("ip_country", "D"),          # invalid len
    ("billing_country", "123"),   # invalid
    ("amount", -1.0),             # invalid gt=0
    ("ip_reputation", 2.0),       # invalid >1
])
def test_predict_validation_errors(bad_key, bad_value):
    payload = _full_valid_payload({bad_key: bad_value})

    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 422
