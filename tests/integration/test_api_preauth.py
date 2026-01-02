# tests/integration/test_api_preauth.py
from __future__ import annotations

from api.routers import preauth as preauth_router


def test_preauth_decision_happy_path(client, monkeypatch):
    # Patch the *router-level* imported function (not model_service module)
    monkeypatch.setattr(
        preauth_router,
        "predict_from_processed_102",
        lambda settings, features: (0.9, None, "hash123", None, "n/a"),
    )

    class DummyArt:
        model_version = "xgb-feedback-2026w01"
        ae_review = 0.1
        ae_block = 0.2
        # not needed when predict_from_processed_102 is patched, but harmless:
        model_features = ["num__V1", "cat__x"]

    monkeypatch.setattr(preauth_router, "ensure_loaded", lambda settings: DummyArt())

    body = {"meta": {"txn_id": "t1"}, "features": {"num__V1": 0.0, "cat__x": 1}}
    r = client.post("/preauth/decision", json=body)

    assert r.status_code == 200, r.text
    data = r.json()
    assert data["decision"] == "BLOCK"
    assert data["request"]["payload_hash"] == "hash123"
