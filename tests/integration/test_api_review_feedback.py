#tests/integration/test_api_review_feedback.py
from __future__ import annotations

from api.routers import preauth as preauth_router


def test_review_queue_and_close_flow(client, test_settings, init_test_db, monkeypatch):
    # Force a REVIEW decision
    monkeypatch.setattr(
        preauth_router,
        "predict_from_processed_102",
        lambda settings, features: (0.5, 0.2, "hash999", 99.0, "elevated"),
    )

    class DummyArt:
        model_version = "xgb-feedback-2026w01"
        ae_review = 0.1
        ae_block = 0.3
        model_features = ["num__V1", "cat__x"]

    monkeypatch.setattr(preauth_router, "ensure_loaded", lambda settings: DummyArt())

    # 1) Create REVIEW via preauth
    body = {"meta": {"txn_id": "t2"}, "features": {"num__V1": 0.0, "cat__x": 1}}
    r = client.post("/preauth/decision", json=body)
    assert r.status_code == 200, r.text
    dec = r.json()
    assert dec["decision"] == "REVIEW"
    review_id = dec["request"]["review_id"]
    assert review_id

    # 2) Review queue should contain it
    q = client.get("/review/queue")
    assert q.status_code == 200, q.text
    items = q.json().get("items", [])
    assert any(it.get("id") == review_id for it in items)

    # 3) Close review
    close_payload = {"analyst_decision": "BLOCK", "analyst": "tester", "notes": "fraud confirmed"}
    c = client.post(f"/review/{review_id}/close", json=close_payload)
    assert c.status_code == 200, c.text
    out = c.json()
    assert out.get("status") == "ok"
    assert out.get("review_id") == review_id
    assert out.get("closed_as") == "BLOCK"
    assert out.get("feedback_id")

    # 4) Export should include the closed review
    ex = client.get("/feedback/export", params={"limit": 200})
    assert ex.status_code == 200, ex.text
    export_items = ex.json().get("items", [])
    assert any(it.get("id") == review_id or it.get("review_id") == review_id for it in export_items)
