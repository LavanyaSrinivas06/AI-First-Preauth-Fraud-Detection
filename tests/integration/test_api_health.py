# tests/integration/test_api_health.py
def test_health_model(client, monkeypatch):
    from api.routers import health as health_router

    class DummyArt:
        model_version = "xgb-feedback-2026w01"
        has_ae = True

    monkeypatch.setattr(health_router, "ensure_loaded", lambda settings: DummyArt())

    r = client.get("/health/model")
    assert r.status_code == 200
    data = r.json()

    # Your endpoint returns active_model_version
    assert data.get("active_model_version") == "xgb-feedback-2026w01"
